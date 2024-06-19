import os
import pandas as pd

def check_paths():
    data_path = "/content/drive/MyDrive/MovieGenre/archive"
    csv_path = os.path.join(data_path, "MovieGenre.csv")
    posters_path = os.path.join(data_path, "SampleMoviePosters")

    print(f"CSV Path: {csv_path}")
    print(f"Posters Path: {posters_path}")

    # Check if CSV file exists
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}")
    else:
        print(f"CSV file found at {csv_path}")

    # Check if posters directory exists
    if not os.path.exists(posters_path):
        print(f"Posters directory not found at {posters_path}")
    else:
        print(f"Posters directory found at {posters_path}")

    if os.path.exists(csv_path) and os.path.exists(posters_path):
        try:
            df = pd.read_csv(csv_path, encoding='latin1')
            print(f"CSV loaded successfully. Number of rows: {len(df)}")

            # Check if poster paths in CSV exist
            for index, row in df.iterrows():
                poster = row['Poster']
                if isinstance(poster, str) and poster:
                    poster_path = os.path.join(posters_path, poster)
                    if not os.path.exists(poster_path):
                        print(f"Poster not found at {poster_path}")
                    else:
                        print(f"Poster found at {poster_path}")
                        break  # Stop after first valid poster to limit output
        except Exception as e:
            print(f"Error reading CSV file: {e}")

if __name__ == "__main__":
    check_paths()
