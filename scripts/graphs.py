import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import pickle
from keras.models import load_model
from PIL import Image
from sklearn.cluster import KMeans

# Constants
BASIC_MODEL_FILE = "/content/drive/My Drive/MovieGenre/models/basic_model.pkl"
CNN_MODEL_FILE = "/content/drive/My Drive/MovieGenre/models/cnn_model.h5"
DATA_FILE = "/content/drive/My Drive/MovieGenre/data/processed/features.csv"
TRAINING_IMAGES_FOLDER = "/content/drive/My Drive/MovieGenre/archive/SampleMoviePosters"
N_COLORS = 5

# Load models
def load_basic_model():
    with open(BASIC_MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    return model

def load_cnn_model():
    return load_model(CNN_MODEL_FILE)

# Extract primary colors
def get_primary_colors(image, n_colors=N_COLORS):
    image_array = np.array(image)
    pixels = image_array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    return colors

# Load and prepare data
def prepare_data():
    data = pd.read_csv(DATA_FILE)
    X = data.drop(columns=["image", "label"])
    y = data["label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded, num_classes=len(np.unique(y_encoded)))

    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, y_encoded, data

# Generate and save graphs
def generate_graphs():
    # Load models and data
    basic_model = load_basic_model()
    cnn_model = load_cnn_model()
    X_train, X_test, y_train, y_test, y_encoded, data = prepare_data()

    # Sample Images with Primary Colors
    plt.figure(figsize=(15, 10))
    for i, row in data.sample(5).iterrows():
        image_path = os.path.join(TRAINING_IMAGES_FOLDER, row['image'])
        image = Image.open(image_path)
        primary_colors = get_primary_colors(image)
        plt.subplot(2, 5, i + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.subplot(2, 5, i + 6)
        for color in primary_colors:
            plt.barh([0], [10], color=[color/255.0], edgecolor='none')
        plt.axis('off')
    plt.suptitle("Sample Images with Primary Colors")
    plt.savefig("/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/sample_images.png")
    plt.close()

    # Distribution of Primary Colors
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
