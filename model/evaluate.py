import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from model.train import load_data, split_data


def evaluate_model():
    print("Starting evaluation...")

    pipeline = joblib.load("model/pipeline.pkl")
    threshold = joblib.load("model/threshold.pkl")

    print("Model loaded")

    # Load data
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    print("Data ready")

    proba = pipeline.predict_proba(X_test)[:, 1]

    preds = (proba >= threshold).astype(int)

   
    print("\n Classification Report:")
    print(classification_report(y_test, preds))

    print("\n Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\n ROC AUC Score:")
    print(roc_auc_score(y_test, proba))


def main():
    evaluate_model()


if __name__ == "__main__":
    main()


