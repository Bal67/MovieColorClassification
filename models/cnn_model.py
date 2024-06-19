import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import os

def train_cnn(images, labels):
    if len(images) == 0 or len(labels) == 0:
        raise ValueError("No valid data to train on.")
    
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(y_train.shape[1], activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"CNN Model Accuracy: {accuracy}")
    
    os.makedirs("models", exist_ok=True)
    model.save("models/cnn_model.h5")

if __name__ == "__main__":
    images = np.load("data/processed/images.npy")
    labels = np.load("data/processed/labels.npy")
    train_cnn(images, labels)
