import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Constants
FEATURES_FILE = "/content/drive/My Drive/MovieGenre/data/processed/features.csv"
MODEL_FILE = "/content/drive/My Drive/MovieGenre/models/basic_model.pkl"

def train_basic_model(*args, **kwargs):
    # Load features
    data = pd.read_csv(FEATURES_FILE)
    X = data.drop(columns=['label'])
    y = data['label']

    # Ensure there are at least two classes
    if len(y.unique()) < 2:
        raise ValueError("The data contains only one class. Ensure there are at least two classes.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Basic Model Accuracy: {accuracy}")

    # Save model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)

    return accuracy

if __name__ == "__main__":
    train_basic_model()
