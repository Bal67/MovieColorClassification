import os
import requests
from zipfile import ZipFile, BadZipFile
import pandas as pd
from scripts.make_dataset import prepare_data
from scripts.build_features import extract_features
from models.basic_model import train_basic_model
from models.cnn_model import train_cnn

def download_data():
    url = "https://github.com/neha1703/VisualSearch/raw/master/VisualSearch/moviedata/archive.zip"
    data_path = "data/raw"
    
    # Ensure the data/raw directory exists
    os.makedirs(data_path, exist_ok=True)
    
    # Define the path for the zip file
    zip_path = os.path.join(data_path, "archive.zip")
    
    # Download the zip file
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(zip_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=128):
                file.write(chunk)
    else:
        print("Failed to download the file. Status code:", response.status_code)
        return
    
    # Unzip the file
    try:
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
    except BadZipFile:
        print("The downloaded file is not a valid zip file.")
        os.remove(zip_path)
        return
    
    # Check the extracted files
    csv_path = os.path.join(data_path, "MovieGenre.csv")
    if not os.path.exists(csv_path):
        print("CSV file not found after extraction.")
        return
    
    # Read the CSV file to ensure it's downloaded correctly
    try:
        df = pd.read_csv(csv_path)
        print("CSV file downloaded and read successfully")
    except Exception as e:
        print("Error reading the CSV file:", e)

if __name__ == "__main__":
    # Download the dataset
    download_data()

    # Ensure necessary directories for processed data exist
    os.makedirs("data/processed", exist_ok=True)

    # Prepare data and build features
    images, labels = prepare_data()
    features = extract_features(images)
    
    # Train models
    train_basic_model(features, labels)
    train_cnn(images, labels)
