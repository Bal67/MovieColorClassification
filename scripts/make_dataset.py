import os
import pandas as pd
from PIL import Image
import numpy as np

def prepare_data():
    data_dir = os.path.join("data", "raw", "movie_genre")
    metadata_path = os.path.join(data_dir, "MovieGenre.csv")
    metadata = pd.read_csv(metadata_path)
    images, labels = [], []
    
    for _, row in metadata.iterrows():
        img_path = os.path.join(data_dir, "Images", row["Title"] + ".jpg")
        if os.path.exists(img_path):
            image = Image.open(img_path).resize((128, 128)).convert('RGB')
            images.append(np.array(image))
            labels.append(row["Genre"])
    
    images = np.array(images)
    labels = np.array(labels)
    
    np.save(os.path.join("data", "processed", "images.npy"), images)
    np.save(os.path.join("data", "processed", "labels.npy"), labels)
    
    return images, labels

