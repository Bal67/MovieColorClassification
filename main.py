import streamlit as st
import json
from PIL import Image
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Constants
BASIC_MODEL_RESULTS_FILE = "/content/drive/My Drive/MovieGenre/data/processed/basic_model_predictions.json"
CNN_MODEL_RESULTS_FILE = "/content/drive/My Drive/MovieGenre/data/processed/cnn_model_predictions.json"
IMAGE_FOLDER = "/content/drive/My Drive/MovieGenre/archive/SampleMoviePosters"
FEATURES_FILE = "/content/drive/My Drive/MovieGenre/data/processed/features.csv"
BASIC_MODEL_PATH = "/content/drive/My Drive/MovieGenre/models/basic_model.pkl"
CNN_MODEL_PATH = "/content/drive/My Drive/MovieGenre/models/cnn_model.h5"
NUM_CLASSES = 10  # Adjust according to your number of classes

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
        st.markdown(f"<div style='width: 50px; height: 50px; background-color: rgb{color};'></div>", unsafe_allow_html=True)

# Train a basic logistic regression model
def train_basic_model():
    # Load features
    data = pd.read_csv(FEATURES_FILE)
    
    # Prepare features and labels
    X = data.drop(columns=["image"])
    y = data["image"]  # Assuming the "image" column contains the labels, adjust as needed

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Basic Model Accuracy: {accuracy:.2f}")

    # Save the model
    pd.to_pickle(model, BASIC_MODEL_PATH)
    st.write(f"Basic model saved to {BASIC_MODEL_PATH}")

    # Save the predictions
    results = [{"image": img, "primary_colors": pc} for img, pc in zip(X_test.index, y_pred)]
    with open(BASIC_MODEL_RESULTS_FILE, 'w') as f:
        json.dump(results, f)
    st.write(f"Basic model predictions saved to {BASIC_MODEL_RESULTS_FILE}")

# Train a CNN model
def train_cnn():
    # Load features
    data = pd.read_csv(FEATURES_FILE)
    
    # Prepare features and labels
    X = data.drop(columns=["image"])
    y = data["image"]  # Assuming the "image" column contains the labels, adjust as needed

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape and normalize the data
    X_train = X_train.values.reshape(-1, 100, 100, 3) / 255.0
    X_test = X_test.values.reshape(-1, 100, 100, 3) / 255.0
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)

    # Build the CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Save the model
    model.save(CNN_MODEL_PATH)
    st.write(f"CNN model saved to {CNN_MODEL_PATH}")

    # Save the predictions
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)
    results = [{"image": img, "primary_colors": pc.tolist()} for img, pc in zip(X_test.index, y_pred_classes)]
    with open(CNN_MODEL_RESULTS_FILE, 'w') as f:
        json.dump(results, f)
    st.write(f"CNN model predictions saved to {CNN_MODEL_RESULTS_FILE}")

# Main function to run the Streamlit app
def main():
    st.title("Movie Poster Primary Colors")

    model_choice = st.selectbox("Choose Model", ["Basic Model", "CNN Model"])

    if model_choice == "Basic Model":
        results_file = BASIC_MODEL_RESULTS_FILE
        if st.button("Train Basic Model"):
            train_basic_model()
    else:
        results_file = CNN_MODEL_RESULTS_FILE
        if st.button("Train CNN Model"):
            train_cnn()

    # Load the dataset
    primary_colors_data = load_primary_colors(results_file)

    # Display the primary colors for each image
    for data in primary_colors_data:
        image_file = os.path.join(IMAGE_FOLDER, data['image'])
        if os.path.exists(image_file):
            display_primary_colors(image_file, data['primary_colors'])

if __name__ == "__main__":
    main()
