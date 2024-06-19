import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from urllib.request import urlretrieve
from PIL import Image

# Constants
DATA_PATH = "/content/drive/MyDrive/MovieGenre/archive"
CSV_PATH = os.path.join(DATA_PATH, "MovieGenre.csv")
TRAIN_POSTERS_PATH = os.path.join(DATA_PATH, "TrainPosters")
TEST_POSTERS_PATH = os.path.join(DATA_PATH, "SampleMoviePosters")
SAVE_LOCATION = 'imdb_posters/'

# Ensure directories exist
os.makedirs(TRAIN_POSTERS_PATH, exist_ok=True)
os.makedirs(TEST_POSTERS_PATH, exist_ok=True)
os.makedirs(SAVE_LOCATION, exist_ok=True)

def download_image(url, save_path):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        image.save(save_path)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def prepare_data():
    # Load CSV
    try:
        df = pd.read_csv(CSV_PATH, encoding='latin1')
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return np.array([]), np.array([]), np.array([]), np.array([])

    print(f"CSV loaded successfully. Number of rows: {len(df)}")

    # Prepare IMDb links
    df['imdb_link'] = ["https://www.imdb.com/title/" + str(x) for x in df['imdb_id']]
    imdb_urls = df['imdb_link'].tolist()
    imdb_ids = df['imdb_id'].tolist()

    records = []
    counter = 0

    for x in tqdm(imdb_urls):
        imdb_id = imdb_ids[counter]
        r = requests.get(x)
        soup = BeautifulSoup(r.text, 'html.parser')
        results = soup.find_all('div', attrs={'class': 'poster'})
        if results:
            first_result = results[0]
            postername = first_result.find('img')['alt']
            imgurl = first_result.find('img')['src']
            records.append((x, postername, imgurl))
        else:
            df = df[df.imdb_id != imdb_id]
        counter += 1

    poster_df = pd.DataFrame(records, columns=['imdb_link', 'postername', 'poster_link'])
    df = pd.merge(df, poster_df, on='imdb_link')
    df.to_csv('movie_metadataWithPoster.csv', sep='\t')

    df_poster = df[['imdb_id', 'poster_link']]
    not_found = []
    for index, row in tqdm(df_poster.iterrows()):
        url = row['poster_link']
        if "https://m.media-amazon.com/" in str(url):
            imdb_id = row['imdb_id']
            jpgname = os.path.join(SAVE_LOCATION, f"{imdb_id}.jpg")
            urlretrieve(url, jpgname)
        else:
            not_found.append(index)

    for filename in os.listdir(SAVE_LOCATION):
        if filename.endswith('.jpg'):
            try:
                img = Image.open(os.path.join(SAVE_LOCATION, filename))
                img.verify()
            except (IOError, SyntaxError) as e:
                print('Bad file:', filename)
                not_found.append(filename.split('.')[0])

    df.drop(df.index[not_found], inplace=True)
    df.drop(columns=[col for col in df.columns if "Unnamed" in col], inplace=True)
    df.to_csv('movie_metadataWithPoster.csv', sep='\t')

    train_images, train_labels, test_images, test_labels = [], [], [], []

    for _, row in df.iterrows():
        poster_url = row['poster_link']
        genre = row['genres']
        imdb_id = row['imdb_id']
        poster_filename = f"{imdb_id}.jpg"
        train_poster_path = os.path.join(TRAIN_POSTERS_PATH, poster_filename)

        if download_image(poster_url, train_poster_path):
            try:
                image = Image.open(train_poster_path).convert('RGB').resize((128, 128))
                train_images.append(np.array(image))
                train_labels.append(genre)
            except Exception as e:
                print(f"Error processing downloaded image {train_poster_path}: {e}")

        test_poster_path = os.path.join(TEST_POSTERS_PATH, poster_filename)
        if os.path.exists(test_poster_path):
            try:
                image = Image.open(test_poster_path).convert('RGB').resize((128, 128))
                test_images.append(np.array(image))
                test_labels.append(genre)
            except Exception as e:
                print(f"Error processing test image {test_poster_path}: {e}")

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    print(f"Successfully downloaded and processed {len(train_images)} training images.")
    print(f"Successfully found and processed {len(test_images)} testing images.")

    return train_images, train_labels, test_images, test_labels

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = prepare_data()
    if len(train_images) > 0 and len(train_labels) > 0 and len(test_images) > 0 and len(test_labels) > 0:
        os.makedirs("data/processed", exist_ok=True)
        np.save("data/processed/train_images.npy", train_images)
        np.save("data/processed/train_labels.npy", train_labels)
        np.save("data/processed/test_images.npy", test_images)
        np.save("data/processed/test_labels.npy", test_labels)
    else:
        print("Insufficient data prepared. Exiting.")
