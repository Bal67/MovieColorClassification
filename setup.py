import os
from scripts.make_dataset import prepare_data
from scripts.build_features import extract_features
from models.basic_model import train_basic_model
from models.cnn_model import train_cnn

if __name__ == "__main__":
    # Ensure necessary directories for processed data exist
    os.makedirs("data/processed", exist_ok=True)

    # Prepare data
    images, labels = prepare_data(missing_value_strategy="default", default_genre="Uncategorized")
    features = extract_features(images)
    
    # Train models
    train_basic_model(features, labels)
    train_cnn(images, labels)
