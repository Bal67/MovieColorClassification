import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import pickle

# Constants
BASIC_MODEL_FILE = "/content/drive/My Drive/MovieGenre/models/basic_model.pkl"
CNN_MODEL_FILE = "/content/drive/My Drive/MovieGenre/models/cnn_model.h5"
DATA_FILE = "/content/drive/My Drive/MovieGenre/data/processed/features.csv"

# Load models
def load_basic_model():
    with open(BASIC_MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    return model

def load_cnn_model():
    return load_model(CNN_MODEL_FILE)

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
    
    # Load models and data
    basic_model = load_basic_model()
    cnn_model = load_cnn_model()
    X_train, X_test, y_train, y_test, y_encoded, data = prepare_data()

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
