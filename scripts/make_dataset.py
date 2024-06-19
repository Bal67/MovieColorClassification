import os
import pandas as pd
from PIL import Image
import numpy as np

def prepare_data(missing_value_strategy="default", default_genre="Unknown"):
    """
    Prepares the dataset by loading images and labels from CSV data.

    Args:
        missing_value_strategy (str, optional): Strategy to handle missing image data.
            - "remove": Removes entries with invalid poster paths or missing images.
            - "default": Assigns default values for missing images and genre labels.
        default_genre (str, optional): Genre label to assign for missing images (applicable only when using "default" strategy). Defaults to "Unknown".

    Returns:
        tuple: A tuple containing two NumPy arrays: images and labels.
    """
    data_path = "/content/drive/MyDrive/MovieGenre/archive"
    csv_path = os.path.join(data_path, "MovieGenre.csv")
    posters_path = os.path.join(data_path, "SampleMoviePosters")

    # Load CSV
    df = pd.read_csv(csv_path, encoding='latin1')

    images = []
    labels = []
    num_missing = 0

    for index, row in df.iterrows():
        poster = row['Poster']
        if isinstance(poster, str) and poster:  # Ensure poster path is a non-empty string
            poster_path = os.path.join(posters_path, poster)
            if os.path.exists(poster_path):
                try:
                    image = Image.open(poster_path).convert('RGB').resize((128, 128))
                    images.append(np.array(image))
                    labels.append(row['Genre'])
                except Exception as e:  # Handle potential errors during image processing
                    print(f"Error processing image {poster_path}: {e}")
                    num_missing += 1  # Increment missing image count
            else:  # Poster file doesn't exist
                if missing_value_strategy == "remove":
                    num_missing += 1
                elif missing_value_strategy == "default":
                    images.append(np.zeros((128, 128, 3)))  # Default image (all zeros)
                    labels.append(default_genre)
                else:
                    raise ValueError(f"Invalid missing_value_strategy: {missing_value_strategy}")
        else:
            if missing_value_strategy == "remove":
                num_missing += 1
            elif missing_value_strategy == "default":
                images.append(np.zeros((128, 128, 3)))  # Default image (all zeros)
                labels.append(default_genre)
            else:
                raise ValueError(f"Invalid missing_value_strategy: {missing_value_strategy}")

    images = np.array(images)
    labels = np.array(labels)

    if num_missing > 0:
        print(f"Handled {num_missing} entries due to missing images.")

    return images, labels

# Example usage with "default" strategy (to ensure we have data)
images, labels = prepare_data(missing_value_strategy="default", default_genre="Uncategorized")

# Save the processed data (assuming "data" directory exists)
np.save("data/processed/images.npy", images)
np.save("data/processed/labels.npy", labels)
