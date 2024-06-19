import os

# Constants
DATA_DIR = "../data"
PROCESSED_DIR = "../data/processed"
MODELS_DIR = "../models"

# Create necessary directories
def create_directories():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Directories created.")

# Main function
def main():
    create_directories()

if __name__ == "__main__":
    main()
