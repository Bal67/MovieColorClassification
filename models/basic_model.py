import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Constants
FEATURES_FILE = "/content/drive/My Drive/MovieGenre/data/processed/features.csv"

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

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Save the model
    model_path = "/content/drive/My Drive/MovieGenre/models/basic_model.pkl"
    pd.to_pickle(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_basic_model()
