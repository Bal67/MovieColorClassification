import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import json

# Constants
FEATURES_FILE = "/content/drive/My Drive/MovieGenre/data/processed/features.csv"
MODEL_SAVE_PATH = "/content/drive/My Drive/MovieGenre/models/cnn_model.h5"
PREDICTIONS_FILE = "/content/drive/My Drive/MovieGenre/data/processed/cnn_model_predictions.json"

def train_cnn(*args, **kwargs):
    # Load data
    data = pd.read_csv(FEATURES_FILE)

    # Prepare data
    X = data.drop(columns=['image', 'label']).values
    y = data['label'].values

    # Check if the number of features matches the expected number for 100x100x3 images
    num_features = X.shape[1]
    expected_features = 100 * 100 * 3
    if num_features != expected_features:
        raise ValueError(f"Expected {expected_features} features (for 100x100x3 images), but got {num_features}")

    # Reshape data for CNN input
    X = X.reshape(-1, 100, 100, 3) / 255.0
    y = np.array(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Save model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # Save predictions
    predictions = [{"image": img, "primary_colors": data.iloc[i, 2:].values.tolist()} for i, img in enumerate(data['image'])] 
    with open(PREDICTIONS_FILE, 'w') as f:
        json.dump(predictions, f)
    print(f"Predictions saved to {PREDICTIONS_FILE}")

    return model

if __name__ == "__main__":
    train_cnn()
