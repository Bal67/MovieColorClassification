import os
import pandas as pd
from PIL import Image
import numpy as np
import requests
from io import BytesIO

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
    data_path = "/content/drive/MyDrive/MovieGenre/archive"
    csv_path = os.path.join(data_path, "MovieGenre.csv")
    train_posters_path = os.path.join(data_path, "TrainPosters")
    test_posters_path = os.path.join(data_path, "SampleMoviePosters")
    
    if not os.path.exists(train_posters_path):
        os.makedirs(train_posters_path)
    
    print(f"CSV Path: {csv_path}")
    print(f"Training Posters Path: {train_posters_path}")
    print(f"Testing Posters Path: {test_posters_path}")

    if not os.path.exists(test_posters_path):
        os.makedirs(test_posters_path)

    # Check if CSV file exists
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}")
        return np.array([]), np.array([])

    # Load CSV
    try:
        df = pd.read_csv(csv_path, encoding='latin1')
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return np.array([]), np.array([])

    print(f"CSV loaded successfully. Number of rows: {len(df)}")

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    num_train_downloaded = 0
    num_train_failed = 0
    num_test_found = 0
    num_test_failed = 0

    for index, row in df.iterrows():
        poster_url = row['Poster']
        genre = row['Genre']
        if isinstance(poster_url, str) and poster_url:
            poster_filename = os.path.basename(poster_url)
            train_poster_path = os.path.join(train_posters_path, poster_filename)

            if download_image(poster_url, train_poster_path):
                try:
                    image = Image.open(train_poster_path).convert('RGB').resize((128, 128))
                    train_images.append(np.array(image))
                    train_labels.append(genre)
                    num_train_downloaded += 1
                except Exception as e:
                    print(f"Error processing downloaded image {train_poster_path}: {e}")
                    num_train_failed += 1
            else:
                num_train_failed += 1
            
            test_poster_path = os.path.join(test_posters_path, poster_filename)
            if os.path.exists(test_poster_path):
                try:
                    image = Image.open(test_poster_path).convert('RGB').resize((128, 128))
                    test_images.append(np.array(image))
                    test_labels.append(genre)
                    num_test_found += 1
                except Exception as e:
                    print(f"Error processing test image {test_poster_path}: {e}")
                    num_test_failed += 1
            else:
                num_test_failed += 1
        else:
            print(f"Invalid poster entry at row {index}")
            num_train_failed += 1
            num_test_failed += 1

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    print(f"Successfully downloaded and processed {num_train_downloaded} training images.")
    print(f"Failed to download or process {num_train_failed} training images.")
    print(f"Successfully found and processed {num_test_found} testing images.")
    print(f"Failed to find or process {num_test_failed} testing images.")

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
