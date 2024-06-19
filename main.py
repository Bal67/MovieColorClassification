import streamlit as st
from PIL import Image
from models.basic_model import load_basic_model, predict_basic
from models.cnn_model import load_cnn_model, predict_cnn

# Load models
basic_model = load_basic_model()
cnn_model = load_cnn_model()

st.title("Movie Poster Genre Classification")
uploaded_file = st.file_uploader("Choose a movie poster image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    
    pred_basic = predict_basic(basic_model, image)
    pred_cnn = predict_cnn(cnn_model, image)
    
    st.write(f"Basic Model Prediction: {pred_basic}")
    st.write(f"CNN Model Prediction: {pred_cnn}")
