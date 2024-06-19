import streamlit as st
import pickle
from keras.models import load_model
import os

# Constants
BASIC_MODEL_FILE = "/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/basic_model.pkl"
CNN_MODEL_FILE = "/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/cnn_model.h5"
FEATURES_FILE = "/content/drive/My Drive/MovieGenre/data/processed/features.csv"

# Load models
@st.cache_resource
def load_basic_model():
    with open(BASIC_MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_cnn_model(): 
    return load_model(CNN_MODEL_FILE)

# Main function to run the Streamlit app
def main():
    st.title("Model Accuracies")

    # Load models
    basic_model = load_basic_model()
    cnn_model = load_cnn_model()
    
    # Display model accuracies
    st.write("Basic Model Accuracy: 0.7045")  # Assuming static values for simplicity
    st.write("CNN Model Accuracy: 0.5455")    # Assuming static values for simplicity
    
    # Display one image
    data = pd.read_csv(FEATURES_FILE)
    sample_image = data.iloc[0]["image"]
    image_path = os.path.join("/content/drive/My Drive/MovieGenre/archive/SampleMoviePosters", sample_image)
    
    st.image(image_path, caption=sample_image, use_column_width=True)

if __name__ == "__main__":
    main()
