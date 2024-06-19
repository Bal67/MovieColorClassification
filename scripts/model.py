import os
import pandas as pd
import joblib
from keras.models import load_model

# Constants
FEATURES_FILE = "../data/processed/features.csv"
BASIC_MODEL_FILE = "../models/basic_model.joblib"
CNN_MODEL_FILE = "../models/cnn_model.h5"

# Load features
def load_features(features_file):
    df = pd.read_csv(features_file)
    return df

# Load models
def load_models():
    basic_model = joblib.load(BASIC_MODEL_FILE)
    cnn_model = load_model(CNN_MODEL_FILE)
    return basic_model, cnn_model

# Main function
def main():
    features = load_features(FEATURES_FILE)
    basic_model, cnn_model = load_models()
    print("Models loaded successfully.")
    # Further code for using models to make predictions can be added here

if __name__ == "__main__":
    main()

