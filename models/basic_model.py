import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Constants
FEATURES_FILE = "../data/processed/features.csv"
MODEL_FILE = "../models/basic_model.joblib"

# Load features
def load_features(features_file):
    df = pd.read_csv(features_file)
    return df

# Train model
def train_model(features):
    X = features.drop(columns=["image"])
    y = features["image"]  # Dummy target, replace with actual target if available
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    return model

# Save model
def save_model(model, model_file):
    import joblib
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")

# Main function
def main():
    features = load_features(FEATURES_FILE)
    model = train_model(features)
    save_model(model, MODEL_FILE)

if __name__ == "__main__":
    main()
