import json
import os

# Assuming you have a function to get predictions from your basic model
def get_basic_model_predictions(image_folder):
    predictions = []
    # Your code to generate predictions for each image
    for image_file in os.listdir(image_folder):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(image_folder, image_file)
            # Replace the following line with actual prediction code
            primary_colors = get_primary_colors(image_path)  
            predictions.append({'image': image_file, 'primary_colors': primary_colors})
    
    return predictions

def save_predictions(predictions, output_file):
    with open(output_file, 'w') as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    IMAGE_FOLDER = "/content/drive/My Drive/MovieGenre/archive/SampleMoviePosters"
    OUTPUT_FILE = "/content/drive/My Drive/MovieGenre/MovieGenreClassification/data/processed/basic_model_predictions.json"
    predictions = get_basic_model_predictions(IMAGE_FOLDER)
    save_predictions(predictions, OUTPUT_FILE)
