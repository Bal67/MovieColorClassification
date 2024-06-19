import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
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

    # Convert X to 4D tensor
    num_samples = X.shape[0]
    X_reshaped = X.values.reshape(num_samples, 5, 3, 1) / 255.0  # Adjust the reshape dimensions based on your feature size

    # Convert y to categorical
    y_categorical = to_categorical(y_encoded, num_classes=len(y.unique()))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42)

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    # Build CNN model
    model = Sequential([
        Conv2D(32, (2, 2), activation='relu', input_shape=(5, 3, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(y.unique()), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model with data augmentation
    model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=50, validation_data=(X_test, y_test))

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"CNN Model Accuracy: {accuracy}")

    # Save model
    model.save(MODEL_FILE)

    return accuracy

if __name__ == "__main__":
    train_cnn()
