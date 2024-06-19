import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

# Constants
FEATURES_FILE = "../data/processed/features.csv"
MODEL_FILE = "../models/cnn_model.h5"
IMAGE_SIZE = (150, 150, 3)  # Assuming resized images are 150x150

# Load features
def load_features(features_file):
    df = pd.read_csv(features_file)
    return df

# Build CNN model
def build_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))  # Adjust output layer as needed
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train model
def train_model(features):
    X = features.drop(columns=["image"]).values.reshape(-1, *IMAGE_SIZE)
    y = features["image"]  # Dummy target, replace with actual target if available
    
    y = to_categorical(y)  # One-hot encode target if necessary
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_model(IMAGE_SIZE)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
    
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy}")
    
    return model

# Save model
def save_model(model, model_file):
    model.save(model_file)
    print(f"Model saved to {model_file}")

# Main function
def main():
    features = load_features(FEATURES_FILE)
    model = train_model(features)
    save_model(model, MODEL_FILE)

if __name__ == "__main__":
    main()
