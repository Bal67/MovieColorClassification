import os
import pandas as pd
from scripts.make_dataset import prepare_data
from scripts.build_features import extract_features
from models.basic_model import train_basic_model
from models.cnn_model import train_cnn

def prepare_local_data():
    data_path = "/content/drive/MyDrive/MovieGenre/archive"
    
    # Define the path for the CSV file and posters directory
    csv_path = os.path.join(data_path, "MovieGenre.csv")
    posters_path = os.path.join(data_path, "SampleMoviePosters")
    
    if not os.path.exists(csv_path) or not os.path.exists(posters_path):
        print("CSV file or SampleMoviePosters directory not found in the archive.")
        return
    
    # Read the CSV file to ensure it's accessible
    try:
        df = pd.read_csv(csv_path, encoding='latin1')  # Using 'latin1' to handle encoding issues
        print("CSV file found and read successfully")
    except Exception as e:
        print("Error reading the CSV file:", e)
        return

if __name__ == "__main__":
    # Prepare the local dataset
    prepare_local_data()

    # Ensure necessary directories for processed data exist
    processed_data_path = "data/processed"
    if os.path.exists(processed_data_path):
        if os.path.isfile(processed_data_path):
            os.remove(processed_data_path)
    else:
        os.makedirs(processed_data_path, exist_ok=True)

    # Prepare data and build features
    images, labels = prepare_data()
    features = extract_features(images)
    
    # Train models
    train_basic_model(features, labels)
    train_cnn(images, labels)
