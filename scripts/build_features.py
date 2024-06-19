import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog
import os

def extract_features(images):
    features = []
    for img in images:
        img_gray = rgb2gray(img)
        feature = hog(img_gray, pixels_per_cell=(16, 16))
        features.append(feature)
    
    features = np.array(features)

    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/features.npy", features)

    return features

if __name__ == "__main__":
    images = np.load("data/processed/images.npy")
    extract_features(images)
