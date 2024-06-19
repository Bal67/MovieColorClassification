import streamlit as st
import pickle
from keras.models import load_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import numpy as np

# Constants
BASIC_MODEL_FILE = "/content/drive/My Drive/MovieGenre/models/basic_model.pkl"
CNN_MODEL_FILE = "/content/drive/My Drive/MovieGenre/models/cnn_model.h5"
FEATURES_FILE = "/content/drive/My Drive/MovieGenre/data/processed/features.csv"

# Load models
def load_basic_model():
    with open(BASIC_MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    return model

def load_cnn_model():
    return load_model(CNN_MODEL_FILE)

# Load and prepare data for testing
def prepare_data():
    data = pd.read_csv(FEATURES_FILE)
    X = data.drop(columns=["image", "label"])
    y = data["label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded, num_classes=len(np.unique(y_encoded)))

    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, y_encoded

# Main function to run the Streamlit app
def main():
    st.title("Model Accuracies")

    # Load models and data
    basic_model = load_basic_model()
    cnn_model = load_cnn_model()
    X_train, X_test, y_train, y_test, y_encoded = prepare_data()

    # Calculate accuracies
    basic_model_accuracy = basic_model.score(X_test, np.argmax(y_test, axis=1))
    cnn_model_accuracy = cnn_model.evaluate(X_test, y_test, verbose=0)[1]

    # Display accuracies
    st.write(f"Basic Model Accuracy: {basic_model_accuracy:.4f}")
    st.write(f"CNN Model Accuracy: {cnn_model_accuracy:.4f}")

if __name__ == "__main__":
    main()
