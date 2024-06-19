import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog

def extract_features(images):
    features = []
    for img in images:
        img_gray = rgb2gray(img)
        feature = hog(img_gray, pixels_per_cell=(16, 16))
        features.append(feature)
    
    features = np.array(features)
    
    return features
