import numpy as np
import pandas as pd
import os
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.request import urlretrieve
from PIL import Image
import ast

# Constants
DATA_PATH = "/content/drive/MyDrive/MovieGenre/archive"
CSV_PATH = os.path.join(DATA_PATH, "MovieGenre.csv")
SAVE_LOCATION = os.path.join(DATA_PATH, "imdb_posters")
SAMPLE_SIZE = 500

# Ensure save directory exists
os.makedirs(SAVE_LOCATION, exist_ok=True)

def download_image(url, save_path):
    try:
        urlretrieve(url, save_path)
        image = Image.open(save_path).convert('RGB')
        image.save(save_path)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)  # Remove corrupted image file
        return False

# Step 1: Web Scraping to get poster URLs
movie = pd.read_csv(CSV_PATH, encoding='latin1')
movie = movie.sample(n=SAMPLE_SIZE, random_state=42)
print(f"CSV loaded successfully. Number of rows: {len(movie)}")

movie['imdb_link'] = movie['Imdb Link']
imdbURLS = movie['imdb_link'].tolist()
imdbIDS = movie['imdbId'].tolist()
records = []

for x in tqdm(imdbURLS):
    imdbID = imdbIDS[imdbURLS.index(x)]
    r = requests.get(x)
    soup = BeautifulSoup(r.text, 'html.parser')
    results = soup.find_all('div', attrs={'class':'poster'})
    if results:
        first_result = results[0]
        postername = first_result.find('img')['alt']
        imgurl = first_result.find('img')['src']
        records.append((x, postername, imgurl))
    else:
        movie = movie[movie.imdbId != imdbID]

poster_df = pd.DataFrame(records, columns=['imdb_link', 'postername', 'poster_link'])
df_movietotal = pd.merge(movie, poster_df, on='imdb_link')
df_movietotal.to_csv('movie_metadataWithPoster.csv', sep='\t')

# Step 2: Posters Download
df_movietotal = pd.read_csv("movie_metadataWithPoster.csv", sep='\t')
genres = []
for entry in df_movietotal["Genre"]:
    genres.append(entry)
df_movietotal["genres"] = genres
df_movietotal['genres'].replace('', np.nan, inplace=True)
df_movietotal.dropna(inplace=True)

df_poster = df_movietotal[['imdbId', 'poster_link']]
not_found = []

for index, row in tqdm(df_poster.iterrows(), total=df_poster.shape[0]):
    url = row['poster_link']
    if "https://m.media-amazon.com/" in str(url):
        id = row['imdbId']
        jpgname = os.path.join(SAVE_LOCATION, f'{id}.jpg')
        if not download_image(url, jpgname):
            not_found.append(index)
    else:
        not_found.append(index)

# Check for corrupt images and remove them
for filename in os.listdir(SAVE_LOCATION):
    if filename.endswith('.jpg'):
        try:
            img = Image.open(os.path.join(SAVE_LOCATION, filename))
            img.verify()
        except (IOError, SyntaxError) as e:
            print('Bad file:', filename)
            os.remove(os.path.join(SAVE_LOCATION, filename))

# Remove entries with missing or corrupt images from the DataFrame
df_movietotal.drop(df_movietotal.index[not_found], inplace=True)
columns_to_drop = [col for col in df_movietotal.columns if "Unnamed" in col]
df_movietotal.drop(columns_to_drop, axis=1, inplace=True)
df_movietotal.to_csv('movie_metadataWithPoster.csv', sep='\t')

print(f"Successfully downloaded {len(df_movietotal) - len(not_found)} images, failed to download {len(not_found)} images.")
