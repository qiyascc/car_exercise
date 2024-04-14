import os
import pandas as pd

def find_csv_files(path_to_dir, suffix=".csv"):
    csv_files = []
    for subdir, dirs, files in os.walk(path_to_dir):
        for file in files:
            if file.endswith(suffix):
                csv_files.append(os.path.join(subdir, file))
    return csv_files

def load_data(path_to_dir):
    csv_files = find_csv_files(path_to_dir)
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the directory or its subdirectories.")
    return pd.read_csv(csv_files[0])

def prepare_features(df):
    from sklearn.preprocessing import LabelEncoder

    df['Vehicle Type'] = LabelEncoder().fit_transform(df['Vehicle Type'])
    df['Drive Type'] = LabelEncoder().fit_transform(df['Drive Type'])

    X = df[['Vehicle Type', 'Year', 'Power', 'Speed']]
    y = df['Drive Type']

    return X, y
