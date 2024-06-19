import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import json

# Constants
FEATURES_FILE = "/content/drive/My Drive/MovieGenre/data/processed/features.csv"
CNN_MODEL_PATH = "/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/cnn_model.h5"
CNN_MODEL_RESULTS_FILE = "/content/drive/My Drive/MovieGenre/MovieGenreClassification/data/processed/cnn_model_predictions.json"
NUM_CLASSES = 10  # Adjust according to your number of classes

def train_cnn(*args):
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

def evaluate_cnn(*args):
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

    # Load the model
    model = load_model(CNN_MODEL_PATH)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)
    results = [{"image": img, "primary_colors": pc.tolist()} for img, pc in zip(X_test.index, y_pred_classes)]
    with open(CNN_MODEL_RESULTS_FILE, 'w') as f:
        json.dump(results, f)
