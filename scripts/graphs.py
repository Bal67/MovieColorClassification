import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import load_model


TRAINING_IMAGES_FOLDER = "/content/drive/My Drive/MovieGenre/archive/SampleMoviePosters"
DATA_FILE = "/content/drive/My Drive/MovieGenre/data/processed/features.csv"
BASIC_MODEL_FILE = "/content/drive/My Drive/MovieGenre/models/basic_model.pkl"
CNN_MODEL_FILE = "/content/drive/My Drive/MovieGenre/models/cnn_model.h5"
N_COLORS = 5

# Load models
def load_basic_model():
    with open(BASIC_MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    return model

def load_cnn_model():
    return load_model(CNN_MODEL_FILE)

# Extract primary colors
def get_primary_colors(image, n_colors=5):
    image_array = np.array(image)
    pixels = image_array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    return colors

# Generate and save graphs
def generate_graphs(data, X_test, y_test):
    plt.figure(figsize=(15, 10))
    
    # Sample images with primary colors
    sample_rows = data.sample(5)
    for idx, (i, row) in enumerate(sample_rows.iterrows()):
        image_path = os.path.join(TRAINING_IMAGES_FOLDER, row['image'])
        image = Image.open(image_path)
        primary_colors = get_primary_colors(image)

        plt.subplot(2, 5, idx  + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.subplot(2, 5, idx + 6)
        for color in primary_colors:
            plt.barh([0], [10], color=[color / 255.0], edgecolor='none')
        plt.axis('off')
    plt.suptitle("Sample Images with Primary Colors")
    plt.savefig("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/sample_images_colors.png")
    plt.close()

    basic_model = load_basic_model()
    cnn_model = load_cnn_model()

    # Distribution of primary colors
    color_columns = [col for col in data.columns if col.startswith('color_')]
    colors = data[color_columns].values.reshape(-1, 3)

    plt.figure()
    plt.title("Distribution of Primary Colors")
    for i in range(3):
        plt.hist(colors[:, i], bins=256, alpha=0.5, label=['Red', 'Green', 'Blue'][i])
    plt.legend(loc='upper right')
    plt.xlabel("Color value")
    plt.ylabel("Frequency")
    plt.savefig("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/color_distribution.png")
    plt.close()

    # Number of images per label
    label_counts = data['label'].value_counts()

    plt.figure()
    plt.title("Number of Images per Label")
    label_counts.plot(kind='bar')
    plt.xlabel("Label")
    plt.ylabel("Number of Images")
    plt.savefig("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/label_distribution.png")
    plt.close()  

    # Basic Model graph
    basic_model_accuracy = basic_model.score(X_test, np.argmax(y_test, axis=1))
    plt.figure()
    plt.title("Basic Model Accuracy")
    plt.bar(["Accuracy"], [basic_model_accuracy])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.savefig("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/basic_model_graph.png")
    plt.close()

    # CNN Model graph
    cnn_model_accuracy = cnn_model.evaluate(X_test, y_test, verbose=0)[1]
    plt.figure()
    plt.title("CNN Model Accuracy")
    plt.bar(["Accuracy"], [cnn_model_accuracy])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.savefig("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/cnn_model_graph.png")
    plt.close()

if __name__ == "__main__":
    generate_graphs()
