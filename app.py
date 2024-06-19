import streamlit as st
import pickle
from keras.models import load_model
import numpy as np

# Constants
BASIC_MODEL_FILE = "/content/drive/My Drive/MovieGenre/models/basic_model.pkl"
CNN_MODEL_FILE = "/content/drive/My Drive/MovieGenre/models/cnn_model.h5"
PROCESSED_DATA_DIR = "/content/drive/My Drive/MovieGenre/data/processed"

# Load models
def load_basic_model():
    with open(BASIC_MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    return model

def load_cnn_model():
    return load_model(CNN_MODEL_FILE)

# Load data for testing
def load_test_data():
    X_test = np.load(f"{PROCESSED_DATA_DIR}/X_test.npy")
    y_test = np.load(f"{PROCESSED_DATA_DIR}/y_test.npy")
    return X_test, y_test

# Main function to run the Streamlit app
def main():
    st.title("Model Accuracies")

    # Load models and data
    basic_model = load_basic_model()
    cnn_model = load_cnn_model()
    X_test, y_test = load_test_data()

    # Calculate accuracies
    basic_model_accuracy = basic_model.score(X_test, np.argmax(y_test, axis=1))
    cnn_model_accuracy = cnn_model.evaluate(X_test, y_test, verbose=0)[1]

    # Display accuracies
    st.write(f"Basic Model Accuracy: {basic_model_accuracy:.4f}")
    st.write(f"CNN Model Accuracy: {cnn_model_accuracy:.4f}")

if __name__ == "__main__":
    main()
