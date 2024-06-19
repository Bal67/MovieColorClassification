import os
import json
import random
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# Constants
IMAGE_FOLDER = "/content/drive/My Drive/MovieGenre/archive/SampleMoviePosters"
PROCESSED_DIR = "/content/drive/My Drive/MovieGenre/data/processed"
TRAIN_SIZE = 250
TEST_SIZE = 25
NUM_COLORS = 5
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "primary_colors.json")
IMAGE_SIZE = (100, 100)  # Resize images to 100x100 for faster processing
MAX_WORKERS = 4  # Number of parallel workers

# Ensure the processed directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_images(image_folder):
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    return image_files

def filter_unprocessable_images(image_files):
    valid_images = []
    for image_file in image_files:
        image_path = os.path.join(IMAGE_FOLDER, image_file)
        try:
            with Image.open(image_path) as img:
                img.verify()
            valid_images.append(image_file)
        except (IOError, SyntaxError):
            continue
    return valid_images

def get_primary_colors(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(IMAGE_SIZE)
    pixels = list(image.getdata())
    kmeans = KMeans(n_clusters=NUM_COLORS, n_init=10)
    kmeans.fit(pixels)
    counter = Counter(kmeans.labels_)
    primary_colors = [kmeans.cluster_centers_[i].tolist() for i in counter.keys()]
    return primary_colors

def analyze_image(image_file, image_folder):
    image_path = os.path.join(image_folder, image_file)
    try:
        primary_colors = get_primary_colors(image_path)
        return {"image": image_file, "primary_colors": primary_colors}
    except Exception as e:
        return None

def analyze_images(image_files, image_folder):
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(analyze_image, image_file, image_folder): image_file for image_file in image_files}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
    return results

def main():
    print("Loading images...")
    image_files = load_images(IMAGE_FOLDER)
    
    print("Filtering out unprocessable images...")
    valid_images = filter_unprocessable_images(image_files)
    print(f"Found {len(valid_images)} valid images after filtering.")

    random.shuffle(valid_images)
    train_files = valid_images[:TRAIN_SIZE]
    test_files = valid_images[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]

    print(f"Analyzing {len(train_files)} images for training set...")
    train_results = analyze_images(train_files, IMAGE_FOLDER)
    print(f"Analyzing {len(test_files)} images for test set...")
    test_results = analyze_images(test_files, IMAGE_FOLDER)

    data = {
        "train": train_results,
        "test": test_results
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f)

    print(f"Dataset creation completed. Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
