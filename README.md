# Movie Color Classification

Movie Color Classification

https://colab.research.google.com/drive/1ojBnhAwkWIASMqb0XgJRBeKZC0P14JJy?usp=sharing


This project classifies movies based on the primary colors of movie posters. The models used are a Basic Logistic Regression Model and a Convolutional Neural Network (CNN).


## Table of Contents

- [Setup](#setup)
- [Main](#main)
- [scripts](#scripts)
- [models](#models)
- [data](#data)

## Project Structure

main.py: The main Streamlit app
scripts/: Contains the scripts for generating graphs and processing data
graphs.py: Script for generating graphs
models/: Contains the trained models
basic_model.pkl: The trained basic logistic regression model
cnn_model.h5: The trained CNN model
data/: Contains the dataset
features.csv: Processed features from the dataset
primary_colors.json: JSON file containing primary colors for the images
MovieGenre.csv: Original dataset with movie genres and information
setup.py: Script for setting up the dataset and preprocessing
requirements.txt: List of dependencies
README.md


## Usage

Download the zipfile for the dataset in the data folder. Proceed to the Google Colab page that is linked at the top of this README.md. Once at the page, mount to your own Google Drive in order to and proceed to follow the instructions for each cell of the Google Colab. 

For the StreamLit application: Google Colab has a hard time opening StreamLit applications. In order to do so, you must run the final cell. At the bottom of that cell will be a link that will lead you to a tunnel website. The bottom cell will also provide you with an IP Address that will look as such (XX.XXX.XXX.XX). Insert that address into the tunnel when prompted for a passcode to access the StreamLit application.


# Model Evaluation


## Evaluation Process and Metric Selection

The evaluation process involves splitting the data into training and testing sets, training the models, and then evaluating their performance on the test set. The primary metric used for evaluation is accuracy, which measures the proportion of correctly classified instances.

For the CNN model, additional metrics like loss and validation accuracy are also considered. Accuracy was chosen as it provides a straightforward measure of how well the model is performing overall.


## Data Processing Pipeline

Load Data: Load images and extract primary colors.

Preprocess Data: Scale and encode the data.

Feature Extraction: Use K-Means clustering to extract primary colors from the images.

Train-Test Split: Split the dataset into training and testing sets.


## Models Evaluated

Basic Logistic Regression Model:

Simple model for initial classification tasks.

Provides a baseline for comparison with more complex models.

Convolutional Neural Network (CNN):
More complex model designed to handle image data.

Expected to perform better due to its ability to capture spatial features in images.


## Results and Conclusions
Basic Model Accuracy: Achieved an accuracy of approximately 75.56% on the test set.

CNN Model Accuracy: Achieved a validation accuracy that improved over multiple epochs, although initial performance metrics may vary.

The project demonstrates that both basic logistic regression and CNN models can classify movie genres based on the primary colors of movie posters, with the CNN model showing potential for further improvements with more data and tuning.


# Acknowledgments
Data sourced from MovieGenre.
This project was developed as part of a machine learning course/project.
