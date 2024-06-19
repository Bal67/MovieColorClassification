import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Constants
FEATURES_FILE = "/content/drive/My Drive/MovieGenre/data/processed/features.csv"
MODEL_FILE = "/content/drive/My Drive/MovieGenre/models/cnn_model.h5"

def train_cnn(*args, **kwargs):
    # Load features
    data = pd.read_csv(FEATURES_FILE)
    X = data.drop(columns=['label', 'image'])  # Exclude image filenames
    y = data['label']

    # Ensure there are at least two classes
    if len(y.unique()) < 2:
        raise ValueError("The data contains only one class. Ensure there are at least two classes.")

    # Encode labels as integers
    label_mapping = {label: idx for idx, label in enumerate(y.unique())}
    y_encoded = y.map(label_mapping)

    # Convert y to categorical
    y_categorical = to_categorical(y_encoded, num_classes=len(y.unique()))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    # Build neural network model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(len(y.unique()), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=32)

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"CNN Model Accuracy: {accuracy}")

    # Save model
    model.save(MODEL_FILE)

    return accuracy

if __name__ == "__main__":
    train_cnn()
