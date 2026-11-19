# src/model_trainer.py
import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from src.feature_pipeline import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
    fit_and_save_preprocessor,
)


def evaluate_predictions(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
    }


def train_and_track_models():
    preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    # Ensure preprocessor exists
    if not os.path.exists(preprocessor_path):
        fit_and_save_preprocessor()

    preprocessor = joblib.load(preprocessor_path)
    train_df = pd.read_csv(os.path.join("artifacts", "train.csv"))
    test_df = pd.read_csv(os.path.join("artifacts", "test.csv"))

    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X_train = preprocessor.transform(train_df[feature_cols])
    y_train = train_df[TARGET_COL].values

    X_test = preprocessor.transform(test_df[feature_cols])
    y_test = test_df[TARGET_COL].values

    # Calculate class imbalance weight for XGBoost (negatives / positives)
    neg_count = np.sum(y_train == 0)
    pos_count = max(np.sum(y_train == 1), 1)
    scale_weight = neg_count / pos_count

    models = {
        "Logistic_Regression_Baseline": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random_Forest_Classifier": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost_Classifier": XGBClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=scale_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }

    mlflow.set_experiment("Olist_Ecommerce_Churn_Prediction")

    best_model = None
    best_model_name = ""
    best_roc_auc = -1.0

    print("\n--- Starting Model Training & MLflow Tracking ---")
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = evaluate_predictions(y_test, y_pred, y_prob)

            # Log parameters, metrics, and model artifact to MLflow
            mlflow.log_param("model_type", model_name)
            mlflow.log_metrics(metrics)

            if isinstance(model, XGBClassifier):
                mlflow.xgboost.log_model(model, name="model")
            else:
                mlflow.sklearn.log_model(
                    model, name="model", serialization_format="cloudpickle"
                )

            print(
                f"[{model_name}] -> ROC-AUC: {metrics['roc_auc']} | "
                f"F1: {metrics['f1_score']} | Recall: {metrics['recall']}"
            )

            if metrics["roc_auc"] > best_roc_auc:
                best_roc_auc = metrics["roc_auc"]
                best_model = model
                best_model_name = model_name

    # Save winning model artifact for FastAPI / Cloud deployment
    model_output_path = os.path.join("artifacts", "churn_model.pkl")
    joblib.dump(best_model, model_output_path)
    print(f"\nWinner: {best_model_name} (ROC-AUC: {best_roc_auc})")
    print(f"Saved winning model to {model_output_path}")


if __name__ == "__main__":
    train_and_track_models()