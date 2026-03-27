
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import joblib
import yaml

from model.preprocess import get_preprocessor


THRESHOLD = 0.32


def load_data():
    return pd.read_csv("data/raw/IBM_customer_data.csv")


def split_data(df):
    X = df.drop('Exited', axis=1)
    y = df['Exited']

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def compute_scale_pos_weight(y_train):
    return (y_train == 0).sum() / (y_train == 1).sum()

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def train_model():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    scale_pos_weight = compute_scale_pos_weight(y_train)

    params = load_params()

    model_params = params["model"]

    model = XGBClassifier(
    **model_params,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)

    THRESHOLD = params["threshold"]


    pipeline = Pipeline([
        ('preprocessing', get_preprocessor()),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, "model/pipeline.pkl")
    joblib.dump(THRESHOLD, "model/threshold.pkl")
    print("Model Trained and Saved Successfully")
    return pipeline, X_test, y_test


if __name__ == "__main__":
    train_model()

