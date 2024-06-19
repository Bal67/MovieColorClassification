import os
import json
import random
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter

# Constants
IMAGE_FOLDER = "../data/posters"
PROCESSED_DIR = "../data/processed"
TRAIN_SIZE = 500
TEST_SIZE = 50
NUM_COLORS = 5
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "primary_colors.json")

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
    image = image.resize((150, 150))
    pixels = list(image.getdata())
    kmeans = KMeans(n_clusters=NUM_COLORS)
    kmeans.fit(pixels)
    counter = Counter(kmeans.labels_)
    primary_colors = [kmeans.cluster_centers_[i] for i in counter.keys()]
    return primary_colors

def analyze_images(image_files, image_folder):
    results = []
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        try:
            primary_colors = get_primary_colors(image_path)
            results.append({"image": image_file, "primary_colors": primary_colors})
        except Exception as e:
            continue
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

    print(f"Analyzing {len(test_files)} images for testing set...")
    test_results = analyze_images(test_files, IMAGE_FOLDER)

    if len(train_results) < TRAIN_SIZE or len(test_results) < TEST_SIZE:
        print("Insufficient data prepared. Exiting.")
        return

    print("Saving results...")
    results = {
        "train": train_results,
        "test": test_results
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f)

    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
