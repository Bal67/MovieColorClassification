import os
import pandas as pd
from PIL import Image
import numpy as np

def prepare_data():
    data_path = "/content/drive/MyDrive/MovieGenre/archive"
    csv_path = os.path.join(data_path, "MovieGenre.csv")
    posters_path = os.path.join(data_path, "SampleMoviePosters")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    images = []
    labels = []
    
    for index, row in df.iterrows():
        poster_path = os.path.join(posters_path, row['Poster'])
        if os.path.exists(poster_path):
            image = Image.open(poster_path).convert('RGB').resize((128, 128))
            images.append(np.array(image))
            labels.append(row['Genre'])
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels
