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
import matplotlib.pyplot as plt

# Constants
BASIC_MODEL_FILE = "/content/drive/My Drive/MovieGenre/models/basic_model.pkl"
CNN_MODEL_FILE = "/content/drive/My Drive/MovieGenre/models/cnn_model.h5"
TRAINING_IMAGES_FOLDER = "/content/drive/My Drive/MovieGenre/archive/SampleMoviePosters"
DATA_FILE = "/content/drive/My Drive/MovieGenre/data/processed/features.csv"
PRIMARY_COLORS_FILE = "/content/drive/My Drive/MovieGenre/data/processed/primary_colors.json"
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

# Load and prepare data for testing
def prepare_data():
    data = pd.read_csv(DATA_FILE)
    X = data.drop(columns=["image", "label"])
    y = data["label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded, num_classes=len(np.unique(y_encoded)))

    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, y_encoded, data

# Generate and save graphs
def generate_graphs(data):
    plt.figure(figsize=(15, 10))
    for i, row in data.sample(5).iterrows():
        image_path = os.path.join(TRAINING_IMAGES_FOLDER, row['image'])
        try:
            image = Image.open(image_path)
            primary_colors = get_primary_colors(image)
            plt.subplot(2, 5, i + 1)
            plt.imshow(image)
            plt.axis('off')
            plt.subplot(2, 5, i + 6)
            for color in primary_colors:
                plt.barh([0], [10], color=[color/255.0], edgecolor='none')
            plt.axis('off')
        except UnidentifiedImageError:
            continue
    plt.suptitle("Sample Images with Primary Colors")
    plt.savefig("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/sample_images.png")
    plt.close()

    # Distribution of primary colors
    color_columns = [col for col in data.columns if col.startswith('color_')]
    colors = data[color_columns].values.reshape(-1, 3)

    plt.figure()
    plt.title("Distribution of Primary Colors")
    for i in range(3):
        plt.hist(colors[:, i], bins=256, alpha=0.5, label=['Red', 'Green', 'Blue'][i])
    plt.legend(loc='upper right')
    plt.xlabel("Color value")
    plt.ylabel("Frequency")
    plt.savefig("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/color_distribution.png")
    plt.close()

    # Number of images per label
    label_counts = data['label'].value_counts()

    plt.figure()
    plt.title("Number of Images per Label")
    label_counts.plot(kind='bar')
    plt.xlabel("Label")
    plt.ylabel("Number of Images")
    plt.savefig("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/label_distribution.png")
    plt.close()

    # Basic Model graph
    basic_model = load_basic_model()
    X_train, X_test, y_train, y_test, y_encoded, data = prepare_data()
    basic_model_accuracy = basic_model.score(X_test, np.argmax(y_test, axis=1))
    plt.figure()
    plt.title("Basic Model Accuracy")
    plt.bar(["Accuracy"], [basic_model_accuracy])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.savefig("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/basic_model_graph.png")
    plt.close()

    # CNN Model graph
    cnn_model = load_cnn_model()
    cnn_model_accuracy = cnn_model.evaluate(X_test, y_test, verbose=0)[1]
    plt.figure()
    plt.title("CNN Model Accuracy")
    plt.bar(["Accuracy"], [cnn_model_accuracy])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.savefig("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/cnn_model_graph.png")
    plt.close()

# Main function to run the Streamlit app
def main():
    st.title("Movie Poster Analysis")

    # Create tabs
    tabs = ["Home", "Basic Model", "CNN Model"]
    active_tab = st.sidebar.radio("Tabs", tabs)

    # Load data and models
    basic_model = load_basic_model()
    cnn_model = load_cnn_model()
    X_train, X_test, y_train, y_test, y_encoded, data = prepare_data()

    # Generate graphs for data exploration
    generate_graphs(data)

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
                st.error("Unable to process the uploaded image. Please upload a valid image file.")

    # Basic Model tab
    elif active_tab == "Basic Model":
        st.header("Basic Model")
        try:
            sample_image = get_sample_image()
            st.image(sample_image, caption="Sample Training Image", use_column_width=True)
            st.write(f"Basic Model Accuracy: {basic_model.score(X_test, y_encoded)}")
            st.write("General Information:")
            st.write("The basic model is a logistic regression model that predicts the genre based on the primary colors extracted from the movie poster.")
            st.image("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/basic_model_graph.png")
        except Exception as e:
            st.error(f"Error displaying Basic Model tab: {e}")

    # CNN Model tab
    elif active_tab == "CNN Model":
        st.header("CNN Model")
        try:
            sample_image = get_sample_image()
            st.image(sample_image, caption="Sample Training Image", use_column_width=True)
            st.write(f"CNN Model Accuracy: {cnn_model.evaluate(X_test, y_test)[1]}")
            st.write("General Information:")
            st.write("The CNN model is a convolutional neural network that predicts the genre based on the primary colors extracted from the movie poster.")
            st.image("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/cnn_model_graph.png")
        except Exception as e:
            st.error(f"Error displaying CNN Model tab: {e}")

    # Data Exploration tab
    st.header("Data Exploration")
    try:
        st.image("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/color_distribution.png")
        st.image("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/label_distribution.png")
        st.image("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/sample_images.png")
    except Exception as e:
        st.error(f"Error displaying data exploration graphs: {e}")

if __name__ == "__main__":
    main()
