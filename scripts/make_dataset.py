import os
import json
import random
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import warnings

# Suppress specific sklearn warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# Constants
IMAGE_FOLDER = "/content/drive/My Drive/MovieGenre/archive/SampleMoviePosters"
PROCESSED_DIR = "/content/drive/My Drive/MovieGenre/data/processed"
PRIMARY_COLORS_FILE = os.path.join(PROCESSED_DIR, "primary_colors.json")
CLASS_LABELS_FILE = os.path.join(PROCESSED_DIR, "class_labels.json")
TRAIN_RATIO = 0.8
NUM_COLORS = 5

# Define color labels
COLOR_LABELS = {
    "red": [255, 0, 0],
    "green": [0, 255, 0],
    "blue": [0, 0, 255],
    "yellow": [255, 255, 0]
}

# Create directories if they don't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

def get_primary_colors(image_path, num_colors=NUM_COLORS):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((100, 100))  # Resize to reduce computation
    image_np = np.array(image).reshape((100 * 100, 3))
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(image_np)
    primary_colors = kmeans.cluster_centers_
    return primary_colors.tolist()

def classify_color(primary_colors):
    color_counts = {color: 0 for color in COLOR_LABELS.keys()}
    for color in primary_colors:
        min_distance = float('inf')
        closest_color = None
        for label, rgb in COLOR_LABELS.items():
            distance = np.linalg.norm(np.array(rgb) - np.array(color))
            if distance < min_distance:
                min_distance = distance
                closest_color = label
        color_counts[closest_color] += 1
    return max(color_counts, key=color_counts.get)

def analyze_images(image_files, image_folder):
    results = []
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_folder, image_file)
        try:
            primary_colors = get_primary_colors(image_path)
            label = classify_color(primary_colors)
            results.append({
                "image": image_file,
                "primary_colors": primary_colors,
                "label": label
            })
        except Exception as e:
            print(f"Failed to process image {image_path}: {e}")
    return results

def main():
    # Load and shuffle images
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith('.jpg')]
    random.shuffle(image_files)

    # Split images into training and testing sets
    split_index = int(len(image_files) * TRAIN_RATIO)
    train_files = image_files[:split_index]
    test_files = image_files[split_index:]

    # Analyze images and get primary colors and labels
    print(f"Analyzing {len(train_files)} images for training set...")
    train_results = analyze_images(train_files, IMAGE_FOLDER)
    print(f"Analyzing {len(test_files)} images for test set...")
    test_results = analyze_images(test_files, IMAGE_FOLDER)

    # Save results
    with open(PRIMARY_COLORS_FILE, 'w') as f:
        json.dump({"train": train_results, "test": test_results}, f)
    print(f"Dataset creation completed. Data saved to {PRIMARY_COLORS_FILE}")

if __name__ == "__main__":
    main()
