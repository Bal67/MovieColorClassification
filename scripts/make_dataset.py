import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter
import json
import random

# Constants
IMAGE_FOLDER = "/content/drive/My Drive/MovieGenre/archive/SampleMoviePosters"
TRAIN_RESULTS_FILE = "/content/drive/My Drive/MovieGenre/MovieGenreClassification/data/processed/train_primary_colors.json"
TEST_RESULTS_FILE = "/content/drive/My Drive/MovieGenre/MovieGenreClassification/data/processed/test_primary_colors.json"
NUM_CLUSTERS = 5
TRAIN_SIZE = 500
TEST_SIZE = 50

# Function to extract primary colors from an image
def get_primary_colors(image_path, num_clusters=NUM_CLUSTERS):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((150, 150))  # Resize for faster processing
    image_array = np.array(image)
    image_array = image_array.reshape((-1, 3))
    
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    kmeans.fit(image_array)
    
    colors = kmeans.cluster_centers_
    counts = Counter(kmeans.labels_)
    
    # Sort colors by frequency
    sorted_colors = [colors[i] for i in counts.keys()]
    return sorted_colors

# Load images and analyze primary colors
def analyze_images(image_files, image_folder):
    results = []
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        primary_colors = get_primary_colors(image_path)
        primary_colors = [tuple(map(int, color)) for color in primary_colors]  # Convert to int tuples
        results.append({
            "image": image_file,
            "primary_colors": primary_colors
        })
    
    return results

# Save results to a JSON file
def save_results(results, results_file):
    with open(results_file, 'w') as f:
        json.dump(results, f)
    print(f"Results saved to {results_file}")

# Main function
def main():
    print("Loading and shuffling images...")
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith('.jpg')]
    random.shuffle(image_files)
    
    train_files = image_files[:TRAIN_SIZE]
    test_files = image_files[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]
    
    print(f"Analyzing {len(train_files)} images for training set...")
    train_results = analyze_images(train_files, IMAGE_FOLDER)
    save_results(train_results, TRAIN_RESULTS_FILE)
    
    print(f"Analyzing {len(test_files)} images for testing set...")
    test_results = analyze_images(test_files, IMAGE_FOLDER)
    save_results(test_results, TEST_RESULTS_FILE)

if __name__ == "__main__":
    main()
