import streamlit as st
import json
from PIL import Image
import os
import pandas as pd
import pickle
import numpy as np
from sklearn.cluster import KMeans
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from scripts.graphs import generate_graphs
import matplotlib.pyplot as plt

# Constants
BASIC_MODEL_FILE = "/content/drive/My Drive/MovieGenre/models/basic_model.pkl"
CNN_MODEL_FILE = "/content/drive/My Drive/MovieGenre/models/cnn_model.h5"
TRAINING_IMAGES_FOLDER = "/content/drive/My Drive/MovieGenre/archive/SampleMoviePosters"
DATA_FILE = "/content/drive/My Drive/MovieGenre/data/processed/features.csv"
PRIMARY_COLORS_FILE = "/content/drive/My Drive/MovieGenre/data/processed/primary_colors.json"
GENRE_FILE = "/content/drive/My Drive/MovieGenre/archive/MovieGenre.csv"
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

# Get sample image from test set
def get_sample_image(test_data):
    sample_file = os.path.join(TRAINING_IMAGES_FOLDER, test_data.sample(1)['image'].values[0])
    return sample_file

# Load and prepare data
def prepare_data():
    data = pd.read_csv(DATA_FILE)
    genre_data = pd.read_csv(GENRE_FILE)
    merged_data = pd.merge(data, genre_data, left_on='image', right_on='Image')

    X = merged_data.drop(columns=["image", "label", "Genre", "Image"])
    y = merged_data["label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded, num_classes=len(np.unique(y_encoded)))

    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, y_encoded, merged_data


# Main function to run the Streamlit app
def main():
    st.title("Movie Poster Analysis")

    # Create tabs
    tabs = ["Home", "Basic Model", "CNN Model", "Data Exploration"]
    active_tab = st.sidebar.radio("Tabs", tabs)

    # Load data and models
    X_train, X_test, y_train, y_test, y_encoded, data = prepare_data()
    basic_model = load_basic_model()
    cnn_model = load_cnn_model()
    
    # Generate and save graphs
    generate_graphs(data, X_test, y_test, basic_model, cnn_model)

    # Home tab
    if active_tab == "Home":
        st.header("Home")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            primary_colors = get_primary_colors(image)
            display_primary_colors(uploaded_file, primary_colors)

            # Display predictions from both models
            image_features = primary_colors.flatten().reshape(1, -1)
            basic_model_pred = basic_model.predict_proba(image_features)
            cnn_model_pred = cnn_model.predict(image_features)

            # Format predictions as percentages
            basic_model_pred_percent = [f"{p * 100:.2f}%" for p in basic_model_pred[0]]
            cnn_model_pred_percent = [f"{p * 100:.2f}%" for p in cnn_model_pred[0]]

            # Display results in a table
            results_df = pd.DataFrame({
                "Model": ["Basic Model", "CNN Model"],
                "Red": [basic_model_pred_percent[0], cnn_model_pred_percent[0]],
                "Green": [basic_model_pred_percent[1], cnn_model_pred_percent[1]],
                "Blue": [basic_model_pred_percent[2], cnn_model_pred_percent[2]],
                "Yellow": [basic_model_pred_percent[3], cnn_model_pred_percent[3]],
            })
            st.table(results_df)

    # Basic Model tab
    elif active_tab == "Basic Model":
        st.header("Basic Model")
        sample_image = get_sample_image(data)
        st.image(sample_image, caption="Sample Training Image", use_column_width=True)
        st.write(f"Basic Model Accuracy: {basic_model.score(X_test, np.argmax(y_test, axis=1))}")
        st.image("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/basic_model_graph.png", caption="Basic Model Accuracy")

    # CNN Model tab
    elif active_tab == "CNN Model":
        st.header("CNN Model")
        sample_image = get_sample_image(data)
        st.image(sample_image, caption="Sample Training Image", use_column_width=True)
        st.write(f"CNN Model Accuracy: {cnn_model.evaluate(X_test, y_test, verbose=0)[1]}")
        st.image("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/cnn_model_graph.png", caption="CNN Model Accuracy")
    
    # Data Exploration tab
    elif active_tab == "Data Exploration":
        st.header("Data Exploration")

        st.image("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/sample_images_with_colors.png", caption="Sample Images with Primary Colors")
        st.write("This graph shows a few sample images from the dataset along with their extracted primary colors.")

        st.image("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/color_distribution.png", caption="Distribution of Primary Colors")
        st.write("This graph displays the distribution of primary colors across the entire dataset. Each color channel (Red, Green, Blue) is shown separately.")

        st.image("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/label_distribution.png", caption="Number of Images per Label")
        st.write("This graph shows the number of images available for each genre in the dataset, providing an overview of the dataset's class distribution.")
        
        st.header("Genre Color Distribution")
        st.image("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/genre_color_distribution.png", caption="Genre-wise Distribution of Primary Colors")
        st.write("This graph shows the distribution of primary colors for each genre, giving insights into the color patterns associated with different movie genres.")
        

if __name__ == "__main__":
    main()
