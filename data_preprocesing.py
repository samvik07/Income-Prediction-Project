import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(file_path):
    return pd.read_csv(file_path)


def encode_labels(df, columns):
    label_encoder = LabelEncoder()
    for column in columns:
        df[column] = label_encoder.fit_transform(df[column])
    return df


def split_features_and_target(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def split_train_and_test_data(X, y, test_size=0.2):
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, y_train, X_test, y_test
