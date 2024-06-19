import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import numpy as np
from models.basic_model import train_basic_model
from models.cnn_model import train_cnn

# Constants
PROCESSED_DATA_DIR = "/content/drive/My Drive/MovieGenre/data/processed"
FEATURES_FILE = f"{PROCESSED_DATA_DIR}/features.csv"

def setup_directories():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs("/content/drive/My Drive/MovieGenre/models", exist_ok=True)

def prepare_data():
    data = pd.read_csv(FEATURES_FILE)
    X = data.drop(columns=["image", "label"])
    y = data["label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded, num_classes=len(np.unique(y_encoded)))

    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    
    # Save the data splits
    np.save(f"{PROCESSED_DATA_DIR}/X_train.npy", X_train)
    np.save(f"{PROCESSED_DATA_DIR}/X_test.npy", X_test)
    np.save(f"{PROCESSED_DATA_DIR}/y_train.npy", y_train)
    np.save(f"{PROCESSED_DATA_DIR}/y_test.npy", y_test)

def main():
    setup_directories()
    # Assuming that scripts.make_dataset and scripts.build_features are run separately or here
    prepare_data()
    X_train = np.load(f"{PROCESSED_DATA_DIR}/X_train.npy")
    X_test = np.load(f"{PROCESSED_DATA_DIR}/X_test.npy")
    y_train = np.load(f"{PROCESSED_DATA_DIR}/y_train.npy")
    y_test = np.load(f"{PROCESSED_DATA_DIR}/y_test.npy")

    train_basic_model(X_train, y_train)
    train_cnn(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
