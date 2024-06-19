import os
import pandas as pd
from PIL import Image
import numpy as np

def prepare_data(missing_value_strategy="default", default_genre="Unknown"):
    data_path = "/content/drive/MyDrive/MovieGenre/archive"
    csv_path = os.path.join(data_path, "MovieGenre.csv")
    posters_path = os.path.join(data_path, "SampleMoviePosters")

    df = pd.read_csv(csv_path, encoding='latin1')

    images = []
    labels = []
    num_missing = 0
    valid_entries = 0

    for index, row in df.iterrows():
        poster = row['Poster']
        if isinstance(poster, str) and poster:
            poster_path = os.path.join(posters_path, poster)
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
                if missing_value_strategy == "remove":
                    num_missing += 1
                elif missing_value_strategy == "default":
                    images.append(np.zeros((128, 128, 3)))
                    labels.append(default_genre)
                    valid_entries += 1
        else:
            if missing_value_strategy == "remove":
                num_missing += 1
            elif missing_value_strategy == "default":
                images.append(np.zeros((128, 128, 3)))
                labels.append(default_genre)
                valid_entries += 1

    images = np.array(images)
    labels = np.array(labels)

    print(f"Found {valid_entries} valid entries.")
    if num_missing > 0:
        print(f"Handled {num_missing} entries due to missing images.")

    return images, labels

if __name__ == "__main__":
    images, labels = prepare_data(missing_value_strategy="default", default_genre="Uncategorized")
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/images.npy", images)
    np.save("data/processed/labels.npy", labels)
