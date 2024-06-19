import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import json

# Constants
FEATURES_FILE = "/content/drive/My Drive/MovieGenre/data/processed/features.csv"
MODEL_SAVE_PATH = "/content/drive/My Drive/MovieGenre/models/basic_model.pkl"
PREDICTIONS_FILE = "/content/drive/My Drive/MovieGenre/data/processed/basic_model_predictions.json"

def train_basic_model():
    # Load data
    data = pd.read_csv(FEATURES_FILE)

    # Prepare data
    X = data.drop(columns=['image', 'label']).values
    y = data['label'].values

    # Ensure there are at least two classes
    if len(set(y)) < 2:
        raise ValueError("The data contains only one class. Ensure there are at least two classes.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Save model
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # Predict and save results
    predictions = model.predict(X_test)
    results = [{"image": data.iloc[i]['image'], "predicted_color": pred} for i, pred in enumerate(predictions)]
    with open(PREDICTIONS_FILE, 'w') as f:
        json.dump(results, f)
    print(f"Predictions saved to {PREDICTIONS_FILE}")

    # Print classification report
    print(classification_report(y_test, predictions))

    return model

if __name__ == "__main__":
    train_basic_model()
