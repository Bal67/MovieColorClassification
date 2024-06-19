import os
import requests
import pandas as pd
from scripts.make_dataset import prepare_data
from scripts.build_features import extract_features
from models.basic_model import train_basic_model
from models.cnn_model import train_cnn

def download_data():
    url = "https://github.com/neha1703/VisualSearch/raw/master/VisualSearch/moviedata/MovieGenre.csv"  # Replace with a direct download link if available
    data_path = "data/raw"
    
    # Ensure the data/raw directory exists
    os.makedirs(data_path, exist_ok=True)
    
    # Define the path for the CSV file
    csv_path = os.path.join(data_path, "MovieGenre.csv")
    
    # Download the CSV file
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(csv_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=128):
                file.write(chunk)
    else:
        print("Failed to download the file. Status code:", response.status_code)
        return

    # Read the CSV file to ensure it's downloaded correctly
    try:
        df = pd.read_csv(csv_path)
        print("CSV file downloaded and read successfully")
    except Exception as e:
        print("Error reading the CSV file:", e)

if __name__ == "__main__":
    download_data()
    images, labels = prepare_data()
    features = extract_features(images)
    
    train_basic_model(features, labels)
    train_cnn(images, labels)
