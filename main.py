import json
import streamlit as st
import os
from models.basic_model import train_basic_model
from models.cnn_model import train_cnn

# Constants
BASIC_MODEL_RESULTS_FILE = "/content/drive/My Drive/MovieGenre/data/processed/basic_model_predictions.json"
CNN_MODEL_RESULTS_FILE = "/content/drive/My Drive/MovieGenre/data/processed/cnn_model_predictions.json"
IMAGE_FOLDER = "/content/drive/My Drive/MovieGenre/archive/SampleMoviePosters"

# Load the primary colors data
def load_primary_colors(results_file):
    with open(results_file, 'r') as f:
        return json.load(f)

# Display the primary colors
def display_primary_colors(image_file, primary_colors):
    st.image(image_file, caption=image_file, use_column_width=True)
    st.write("Primary Colors:")
    for color in primary_colors:
        st.write(f"RGB: {color}")
        st.markdown(f"<div style='width: 50px; height: 50px; background-color: rgb{color});'></div>", unsafe_allow_html=True)

# Main function to run the Streamlit app
def main():
    st.title("Movie Poster Primary Colors")

    # Train and evaluate models
    st.write("Training Basic Model...")
    basic_model_re
