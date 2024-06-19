import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter
import json

# Constants
IMAGE_FOLDER = "/content/drive/My Drive/MovieGenre/archive/SampleMoviePosters"
RESULTS_FILE = "/content/drive/My Drive/MovieGenre/MovieGenreClassification/data/processed/primary_colors.json"
NUM_CLUSTERS = 5

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
def analyze_images(image_folder):
    results = []
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    
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
    print("Analyzing images for primary colors...")
    results = analyze_images(IMAGE_FOLDER)
    save_results(results, RESULTS_FILE)

if __name__ == "__main__":
    main()
