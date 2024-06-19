import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from urllib.request import urlretrieve
from PIL import Image

# Constants
DATA_PATH = "/content/drive/MyDrive/MovieGenre/archive"
CSV_PATH = os.path.join(DATA_PATH, "MovieGenre.csv")
TRAIN_POSTERS_PATH = os.path.join(DATA_PATH, "TrainPosters")
TEST_POSTERS_PATH = os.path.join(DATA_PATH, "SampleMoviePosters")

# Ensure directories exist
os.makedirs(TRAIN_POSTERS_PATH, exist_ok=True)
os.makedirs(TEST_POSTERS_PATH, exist_ok=True)

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

def prepare_data():
    # Load CSV
    try:
        df = pd.read_csv(CSV_PATH, encoding='latin1')
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return np.array([]), np.array([]), np.array([]), np.array([])

    print(f"CSV loaded successfully. Number of rows: {len(df)}")

    train_images, train_labels, test_images, test_labels = [], [], [], []
    download_success, download_failure = 0, 0
    process_success, process_failure = 0, 0

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        imdb_id = row['imdbId']
        poster_url = row['Poster']
        genre = row['Genre']
        poster_filename = f"{imdb_id}.jpg"
        train_poster_path = os.path.join(TRAIN_POSTERS_PATH, poster_filename)

        # Download and process training images
        if download_image(poster_url, train_poster_path):
            download_success += 1
            try:
                image = Image.open(train_poster_path).convert('RGB').resize((128, 128))
                train_images.append(np.array(image))
                train_labels.append(genre)
                process_success += 1
            except Exception as e:
                print(f"Error processing downloaded image {train_poster_path}: {e}")
                process_failure += 1
        else:
            download_failure += 1

        # Process test images if available
        test_poster_path = os.path.join(TEST_POSTERS_PATH, poster_filename)
        if os.path.exists(test_poster_path):
            try:
                image = Image.open(test_poster_path).convert('RGB').resize((128, 128))
                test_images.append(np.array(image))
                test_labels.append(genre)
                process_success += 1
            except Exception as e:
                print(f"Error processing test image {test_poster_path}: {e}")
                process_failure += 1

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    print(f"Successfully downloaded {download_success} images, failed to download {download_failure} images.")
    print(f"Successfully processed {process_success} images, failed to process {process_failure} images.")

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
