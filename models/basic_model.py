import os
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from skimage.color import rgb2gray
from skimage.feature import hog

def train_basic_model(features, labels):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(features, labels)
    
    with open(os.path.join("models", "basic_model.pkl"), "wb") as f:
        pickle.dump(knn, f)

def load_basic_model():
    with open(os.path.join("models", "basic_model.pkl"), "rb") as f:
        return pickle.load(f)

def predict_basic(model, image):
    feature = hog(rgb2gray(np.array(image.resize((128, 128)))), pixels_per_cell=(16, 16)).reshape(1, -1)
    return model.predict(feature)[0]

