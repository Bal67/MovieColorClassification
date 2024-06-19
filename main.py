import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title('Movie Genre Classification')

st.write('This is a simple Streamlit app to classify movie genres based on posters.')

# Load the trained model
model = np.load("models/basic_model.py", allow_pickle=True).item()

# Load the test images
images = np.load("data/processed/images.npy")
labels = np.load("data/processed/labels.npy")

# Add a file uploader to upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB').resize((128, 128))
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Preprocess the image
    image = np.array(image)
    image = image.reshape(1, -1)
    
    # Predict the genre
    prediction = model.predict(image)
    st.write(f'Predicted Genre: {prediction[0]}')

