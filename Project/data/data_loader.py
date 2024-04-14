import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    return pd.read_csv(filepath)

def prepare_features(df):
    df['Vehicle Type'] = LabelEncoder().fit_transform(df['Vehicle Type'])
    df['Drive Type'] = LabelEncoder().fit_transform(df['Drive Type'])

    X = df[['Vehicle Type', 'Year', 'Power', 'Speed']]
    y = df['Drive Type']

    return X, y
