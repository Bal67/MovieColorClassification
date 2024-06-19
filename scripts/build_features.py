import json
import pandas as pd

# Constants
DATA_FILE = "/content/drive/My Drive/MovieGenre/data/processed/primary_colors.json"
FEATURES_FILE = "/content/drive/My Drive/MovieGenre/data/processed/features.csv"

# Load data
def load_data(data_file):
    with open(data_file, 'r') as f:
        data = json.load(f)
    return data

# Process data to build features
def build_features(data):
    rows = []
    for item in data['train']:
        row = {"image": item["image"]}
        primary_colors = item["primary_colors"]
        # Flatten the primary colors to match the expected feature shape
        flattened_colors = [color for colors in primary_colors for color in colors]
        row.update({f"color_{i}": color for i, color in enumerate(flattened_colors)})
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
