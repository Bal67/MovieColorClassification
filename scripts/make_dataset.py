import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from urllib.request import urlretrieve
import ast
from PIL import Image

# Constants
CSV_PATH = "/content/drive/MyDrive/MovieGenre/archive/MovieGenre.csv"
SAVE_LOCATION = "/content/drive/MyDrive/MovieGenre/archive/TrainPosters"
PROCESSED_DATA_DIR = "data/processed"
FINAL_DATA_PATH = "movie_metadataWithPoster.csv"
IMAGE_SIZE = (128, 128)

# Create directories if not exist
os.makedirs(SAVE_LOCATION, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Step 1: Load Dataset
def load_dataset(csv_path):
    movie = pd.read_csv(csv_path, encoding='latin1')
    return movie

# Step 2: Scrape IMDb Pages for Posters
def fetch_poster_url(imdb_url):
    try:
        r = requests.get(imdb_url)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            poster_tag = soup.find('div', class_='poster')
            if poster_tag:
                img_tag = poster_tag.find('img')
                if img_tag and 'src' in img_tag.attrs:
                    return img_tag['src']
        print(f"Failed to retrieve poster from {imdb_url}, status code: {r.status_code}")
    except Exception as e:
        print(f"Exception for {imdb_url}: {e}")
    return None

def scrape_imdb_links(movie_df):
    movie_df['poster_link'] = movie_df['Imdb Link'].apply(fetch_poster_url)
    movie_df.dropna(subset=['poster_link'], inplace=True)
    movie_df.to_csv(FINAL_DATA_PATH, index=False)
    return movie_df

# Step 3: Download Posters
def download_poster(url, save_path):
    try:
        urlretrieve(url, save_path)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def download_posters(movie_df):
    failed_downloads = []
    for idx, row in tqdm(movie_df.iterrows(), total=movie_df.shape[0]):
        imdb_id = row['imdbId']
        poster_url = row['poster_link']
        save_path = os.path.join(SAVE_LOCATION, f"{imdb_id}.jpg")
        if not download_poster(poster_url, save_path):
            failed_downloads.append(idx)
    movie_df.drop(failed_downloads, inplace=True)
    movie_df.to_csv(FINAL_DATA_PATH, index=False)
    return movie_df

# Step 4: Prepare Dataset
def prepare_data(data_path, save_location):
    data = pd.read_csv(data_path)
    images = []
    labels = []
    for idx, row in data.iterrows():
        try:
            img_path = os.path.join(save_location, f"{row['imdbId']}.jpg")
            image = Image.open(img_path).convert('RGB').resize(IMAGE_SIZE)
            images.append(np.array(image))
            labels.append(row['Genre'])
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
    return np.array(images), np.array(labels)

def main():
    print("Loading dataset...")
    movie = load_dataset(CSV_PATH)
    print(f"CSV loaded successfully. Number of rows: {len(movie)}")

    print("Scraping IMDb links for posters...")
    movie_with_posters = scrape_imdb_links(movie)

    print("Downloading posters...")
    movie_final = download_posters(movie_with_posters)

    print("Preparing dataset...")
    images, labels = prepare_data(FINAL_DATA_PATH, SAVE_LOCATION)

    if len(images) == 0:
        print("No valid images to process. Exiting.")
        return

    print(f"Processed {len(images)} images and {len(labels)} labels.")
    np.save(os.path.join(PROCESSED_DATA_DIR, "images.npy"), images)
    np.save(os.path.join(PROCESSED_DATA_DIR, "labels.npy"), labels)

if __name__ == "__main__":
    main()
