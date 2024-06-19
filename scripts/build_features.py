import json
import pandas as pd

# Constants
BASIC_DATA_FILE = "../data/processed/basic_model_predictions.json"
CNN_DATA_FILE = "../data/processed/cnn_model_predictions.json"
FEATURES_FILE = "../data/processed/features.csv"

# Load data
def load_data(data_file):
    with open(data_file, 'r') as f:
        data = json.load(f)
    return data

# Process data to build features
def build_features(data):
    rows = []
    for item in data:
        row = {"image": item["image"]}
        for i, color in enumerate(item["primary_colors"]):
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
    basic_data = load_data(BASIC_DATA_FILE)
    cnn_data = load_data(CNN_DATA_FILE)

    basic_features = build_features(basic_data)
    cnn_features = build_features(cnn_data)

    # Combine basic and cnn features into one dataframe
    combined_features = basic_features + cnn_features

    save_features(combined_features, FEATURES_FILE)

if __name__ == "__main__":
    main()
