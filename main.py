import streamlit as st
import json
from PIL import Image, UnidentifiedImageError
import os
import pandas as pd
import pickle
import numpy as np
from sklearn.cluster import KMeans
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from scripts import graphs  # Update import statement

# Constants
BASIC_MODEL_FILE = "/content/drive/My Drive/MovieGenre/models/basic_model.pkl"
CNN_MODEL_FILE = "/content/drive/My Drive/MovieGenre/models/cnn_model.h5"
TRAINING_IMAGES_FOLDER = "/content/drive/My Drive/MovieGenre/archive/SampleMoviePosters"
DATA_FILE = "/content/drive/My Drive/MovieGenre/data/processed/features.csv"
N_COLORS = 5

# Load models
def load_basic_model():
    with open(BASIC_MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    return model

def load_cnn_model():
    return load_model(CNN_MODEL_FILE)

# Extract primary colors
def get_primary_colors(image, n_colors=N_COLORS):
    image_array = np.array(image)
    pixels = image_array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    return colors

# Display the primary colors
def display_primary_colors(image_file, primary_colors):
    st.image(image_file, caption="Uploaded Image", use_column_width=True)
    st.write("Primary Colors:")
    for color in primary_colors:
        color_block = np.zeros((50, 50, 3), dtype=int)
        color_block[:, :] = color
        st.image(color_block, caption=f"RGB: {color}", use_column_width=False, width=100)

# Get sample image from training set
def get_sample_image():
    files = os.listdir(TRAINING_IMAGES_FOLDER)
    sample_file = os.path.join(TRAINING_IMAGES_FOLDER, files[0])
    return sample_file

# Main function to run the Streamlit app
def main():
    st.title("Movie Poster Analysis")

    # Create tabs
    tabs = ["Home", "Basic Model", "CNN Model"]
    active_tab = st.sidebar.radio("Tabs", tabs)

    # Load data and models
    basic_model = load_basic_model()
    cnn_model = load_cnn_model()
    X_train, X_test, y_train, y_test, y_encoded, data = graphs.prepare_data()

    # Generate graphs for data exploration
    graphs.generate_graphs(data)

    # Home tab
    if active_tab == "Home":
        st.header("Home")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                primary_colors = get_primary_colors(image)
                display_primary_colors(uploaded_file, primary_colors)

                # Display predictions from both models
                image_features = primary_colors.flatten().reshape(1, -1)
                basic_model_pred = basic_model.predict(image_features)
                cnn_model_pred = cnn_model.predict(image_features)

                # Display results in a table
                results_df = pd.DataFrame({
                    "Model": ["Basic Model", "CNN Model"],
                    "Prediction": [basic_model_pred[0], cnn_model_pred[0]]
                })
                st.table(results_df)

            except UnidentifiedImageError:
                st.error("Cannot identify image file. Please upload a valid image.")

    # Basic Model tab
    elif active_tab == "Basic Model":
        st.header("Basic Model")
        st.image("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/basic_model_graph.png", caption="Basic Model Accuracy Graph", use_column_width=True)
        st.image("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/label_distribution.png", caption="Label Distribution", use_column_width=True)
        st.image("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/color_distribution.png", caption="Color Distribution", use_column_width=True)

    # CNN Model tab
    elif active_tab == "CNN Model":
        st.header("CNN Model")
        st.image("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/cnn_model_graph.png", caption="CNN Model Accuracy Graph", use_column_width=True)
        st.image("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/label_distribution.png", caption="Label Distribution", use_column_width=True)
        st.image("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/color_distribution.png", caption="Color Distribution", use_column_width=True)

if __name__ == "__main__":
    main()
