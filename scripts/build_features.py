import json
import pandas as pd
import numpy as np

# Constants
DATA_FILE = "/content/drive/My Drive/MovieGenre/data/processed/primary_colors.json"
FEATURES_FILE = "/content/drive/My Drive/MovieGenre/data/processed/features.csv"

# Load data
def load_data(data_file):
    with open(data_file, 'r') as f:
        data = json.load(f)
    return data

# Determine the most dominant primary color
def get_dominant_color(primary_colors):
    color_sums = np.sum(primary_colors, axis=0)
    color_names = ["Red", "Green", "Blue", "Yellow"]
    dominant_color = color_names[np.argmax(color_sums)]
    return dominant_color

# Process data to build features
def build_features(data):
    rows = []
    for item in data['train'] + data['test']:
        primary_colors = item["primary_colors"]
        dominant_color = get_dominant_color(primary_colors)
        row = {"image": item["image"], "label": dominant_color}
        for i, color in enumerate(primary_colors):
            row[f"color_{i}_r"] = color[0]
            row[f"color_{i}_g"] = color[1]
            row[f"color_{i}_b"] = color[2]
        rows.append(row)
    return rows

# Save features to a CSV file
def save_features(features, features_file):
    df = pd.DataFrame(features)
    df.to_csv(features_file, index=False)
    print(f"Features saved to {features_file}")

# Main function
def main():
    data = load_data(DATA_FILE)
    features = build_features(data)
    save_features(features, FEATURES_FILE)

if __name__ == "__main__":
    main()
