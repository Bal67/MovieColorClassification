import os
from scripts.make_dataset import prepare_data
from scripts.build_features import extract_features
from models.basic_model import train_basic_model
from models.cnn_model import train_cnn

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    images, labels = prepare_data(missing_value_strategy="default", default_genre="Uncategorized")
    if len(images) == 0 or len(labels) == 0:
        print("No valid data prepared. Exiting.")
        exit()

    features = extract_features(images)
    if len(features) == 0:
        print("No valid features extracted. Exiting.")
        exit()
    
    train_basic_model(features, labels)
    train_cnn(images, labels)
