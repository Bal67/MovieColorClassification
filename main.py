import streamlit as st
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from skimage.color import rgb2gray
from skimage.feature import hog
import joblib
from tensorflow.keras.models import load_model

st.title('Movie Genre Classification')

st.write('This is a simple Streamlit app to classify movie genres based on posters.')

# Load the RandomForest model
try:
    rf_model = joblib.load("models/basic_model.joblib")
    rf_model_loaded = True
except Exception as e:
    st.error(f"Error loading RandomForest model: {e}")
    rf_model_loaded = False

# Load the CNN model
try:
    cnn_model = load_model("models/cnn_model.h5")
    cnn_model_loaded = True
except Exception as e:
    st.error(f"Error loading CNN model: {e}")
    cnn_model_loaded = False

# Function to preprocess the image
def preprocess_image(image):
    image = image.convert('RGB').resize((128, 128))
    image_array = np.array(image)
    return image_array

# Function to extract HOG features
def extract_hog_features(image_array):
    img_gray = rgb2gray(image_array)
    features = hog(img_gray, pixels_per_cell=(16, 16))
    return features

# Add a file uploader to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    image_array = preprocess_image(image)

    # Predict with RandomForest model
    if rf_model_loaded:
        features = extract_hog_features(image_array)
        features = features.reshape(1, -1)
        rf_prediction = rf_model.predict(features)
        st.write(f'RandomForest Prediction: {rf_prediction[0]}')

    # Predict with CNN model
    if cnn_model_loaded:
        image_array = image_array.reshape(1, 128, 128, 3)
        cnn_prediction = cnn_model.predict(image_array)
        predicted_genre = np.argmax(cnn_prediction, axis=1)
        st.write(f'CNN Prediction: {predicted_genre[0]}')
