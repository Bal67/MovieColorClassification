import os
import requests
from zipfile import ZipFile
from scripts.make_dataset import prepare_data
from scripts.build_features import extract_features
from models.basic_model import train_basic_model
from models.cnn_model import train_cnn

def download_data():
    url = "https://www.kaggle.com/neha1703/movie-genre-from-its-poster/download"
    data_path = os.path.join("data", "raw")
    os.makedirs(data_path, exist_ok=True)
    
    response = requests.get(url, stream=True)
    with open(os.path.join(data_path, "movie_poster_dataset.zip"), "wb") as file:
        for chunk in response.iter_content(chunk_size=128):
            file.write(chunk)
    
    with ZipFile(os.path.join(data_path, "movie_poster_dataset.zip"), 'r') as zip_ref:
        zip_ref.extractall(data_path)

if __name__ == "__main__":
    download_data()
    images, labels = prepare_data()
    features = extract_features(images)
    
    train_basic_model(features, labels)
    train_cnn(images, labels)

