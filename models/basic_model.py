import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Constants
FEATURES_FILE = "/content/drive/My Drive/MovieGenre/data/processed/features.csv"
MODEL_SAVE_PATH = "/content/drive/My Drive/MovieGenre/models/basic_model.pkl"

def train_basic_model():
    # Load data
    data = pd.read_csv(FEATURES_FILE)

    # Verify and prepare data
    # Replace 'positive_class' with actual class names or logic
    data['label'] = data['image'].apply(lambda x: 1 if 'positive_class' in x else 0)  
    if len(data['label'].unique()) < 2:
        raise ValueError("The data contains only one class. Ensure there are at least two classes.")

    X = data.drop(columns=['image', 'label']).values
    y = data['label'].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=1000)  # Increase the number of iterations
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Basic model accuracy: {accuracy}")

    # Save model
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"Basic model saved to {MODEL_SAVE_PATH}")

    return model

if __name__ == "__main__":
    train_basic_model()
