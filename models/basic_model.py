import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def train_basic_model(features, labels):
    if len(features) == 0 or len(labels) == 0:
        print("No valid data to train on.")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/basic_model.joblib")

if __name__ == "__main__":
    features = np.load("data/processed/features.npy")
    labels = np.load("data/processed/labels.npy")
    train_basic_model(features, labels)
