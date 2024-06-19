import os
import requests
from zipfile import ZipFile, BadZipFile
from scripts.make_dataset import prepare_data
from scripts.build_features import extract_features
from models.basic_model import train_basic_model
from models.cnn_model import train_cnn

def download_data():
    url = "https://www.kaggle.com/neha1703/movie-genre-from-its-poster/download"
    data_path = "data/raw"
    
    # Check if data_path is a directory and create it if it doesn't exist
    if os.path.exists(data_path):
        if not os.path.isdir(data_path):
            os.remove(data_path)
            os.makedirs(data_path)
    else:
        os.makedirs(data_path)
    
    zip_path = os.path.join(data_path, "movie_poster_dataset.zip")
    
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(zip_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=128):
                file.write(chunk)
    else:
        print("Failed to download the file. Status code:", response.status_code)
        return
    
    try:
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
    except BadZipFile:
        print("The downloaded file is not a zip file.")
        os.remove(zip_path)

if __name__ == "__main__":
    download_data()
    images, labels = prepare_data()
    features = extract_features(images)
    
    train_basic_model(features, labels)
    train_cnn(images, labels)
