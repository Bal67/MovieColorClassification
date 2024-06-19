import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Constants
FEATURES_FILE = "/content/drive/My Drive/MovieGenre/MovieGenreClassification/data/processed/features.csv"
BASIC_MODEL_PATH = "/content/drive/My Drive/MovieGenre/MovieGenreClassification/models/basic_model.pkl"
BASIC_MODEL_RESULTS_FILE = "/content/drive/My Drive/MovieGenre/MovieGenreClassification/data/processed/basic_model_predictions.json"

def train_basic_model():
    # Load features
    data = pd.read_csv(FEATURES_FILE)
    
    # Prepare features and labels
    X = data.drop(columns=["image"])
    y = data["image"]  # Assuming the "image" column contains the labels, adjust as needed

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, BASIC_MODEL_PATH)

def evaluate_basic_model():
    # Load features
    data = pd.read_csv(FEATURES_FILE)
    
    # Prepare features and labels
    X = data.drop(columns=["image"])
    y = data["image"]  # Assuming the "image" column contains the labels, adjust as needed

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load the model
    model = joblib.load(BASIC_MODEL_PATH)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Basic Model Accuracy: {accuracy:.2f}")

    # Save the predictions
    results = [{"image": img, "primary_colors": pc} for img, pc in zip(X_test.index, y_pred)]
    with open(BASIC_MODEL_RESULTS_FILE, 'w') as f:
        json.dump(results, f)

