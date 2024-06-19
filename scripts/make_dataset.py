import os
import pandas as pd
from PIL import Image
import numpy as np
import gdown

def download_image(file_id, save_path):
    url = f"https://drive.google.com/drive/folders/1WfK0iYuQ_6v_VWca_dLkcayYYgErx2DJ?usp=sharing"
    try:
        gdown.download(url, save_path, quiet=False)
        img = Image.open(save_path).convert('RGB')
        img.save(save_path)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def prepare_data(missing_value_strategy="default", default_genre="Unknown"):
    data_path = "/content/drive/MyDrive/MovieGenre/archive"
    csv_path = os.path.join(data_path, "MovieGenre.csv")
    posters_path = os.path.join(data_path, "SampleMoviePosters")

    print(f"CSV Path: {csv_path}")
    print(f"Posters Path: {posters_path}")

    if not os.path.exists(posters_path):
        os.makedirs(posters_path)

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

    images = []
    labels = []
    num_missing = 0
    valid_entries = 0
    invalid_entries = 0

    for index, row in df.iterrows():
        poster = row['Poster']
        if isinstance(poster, str) and poster:
            file_id = poster.split('/')[-2]  # Extract file ID from URL
            poster_filename = os.path.basename(poster)
            poster_path = os.path.join(posters_path, poster_filename)

            if not os.path.exists(poster_path):
                if not download_image(file_id, poster_path):
                    invalid_entries += 1
                    continue

            if os.path.exists(poster_path):
                try:
                    image = Image.open(poster_path).convert('RGB').resize((128, 128))
                    images.append(np.array(image))
                    labels.append(row['Genre'])
                    valid_entries += 1
                except Exception as e:
                    print(f"Error processing image {poster_path}: {e}")
                    num_missing += 1
            else:
                print(f"Poster not found at {poster_path}")
                if missing_value_strategy == "remove":
                    num_missing += 1
                elif missing_value_strategy == "default":
                    images.append(np.zeros((128, 128, 3)))
                    labels.append(default_genre)
                    valid_entries += 1
        else:
            print(f"Invalid poster entry at row {index}")
            if missing_value_strategy == "remove":
                num_missing += 1
            elif missing_value_strategy == "default":
                images.append(np.zeros((128, 128, 3)))
                labels.append(default_genre)
                valid_entries += 1

    images = np.array(images)
    labels = np.array(labels)

    print(f"Found {valid_entries} valid entries.")
    print(f"Skipped {invalid_entries} invalid entries due to 404 errors.")
    if num_missing > 0:
        print(f"Handled {num_missing} entries due to missing images.")

    return images, labels

if __name__ == "__main__":
    images, labels = prepare_data(missing_value_strategy="default", default_genre="Uncategorized")
    if len(images) > 0 and len(labels) > 0:
        os.makedirs("data/processed", exist_ok=True)
        np.save("data/processed/images.npy", images)
        np.save("data/processed/labels.npy", labels)
    else:
        print("No valid data prepared. Exiting.")
