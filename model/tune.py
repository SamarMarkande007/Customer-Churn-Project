import optuna
import yaml
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score

from model.train import load_data, split_data, compute_scale_pos_weight
from model.preprocess import get_preprocessor


THRESHOLD = 0.32


def objective(trial):

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    scale_pos_weight = compute_scale_pos_weight(y_train)

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "logloss",
        "random_state": 42
    }

    model = XGBClassifier(**params)

    pipeline = Pipeline([
        ('preprocessing', get_preprocessor()),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= THRESHOLD).astype(int)

    return recall_score(y_test, preds)


def main():
    print("Starting Optuna tuning...")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("Best Params:", study.best_params)
    print("Best Recall:", study.best_value)

    params = {
    "model": study.best_params,
    "threshold": 0.32
}

    with open("params.yaml", "w") as f:
        yaml.dump(params, f)


if __name__ == "__main__":
    main()