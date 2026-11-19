# src/feature_pipeline.py
import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Behavioral, financial, and logistics features (NO target leakage)
NUMERICAL_FEATURES = [
    "frequency_orders",
    "total_monetary_spend",
    "avg_order_value",
    "avg_delivery_delay_days",
    "max_installments",
    "avg_review_score",
    "is_late_delivery",
    "has_written_review",
]

CATEGORICAL_FEATURES = [
    "customer_state",
    "primary_payment_type",
]

TARGET_COL = "is_churned"


def create_preprocessor() -> ColumnTransformer:
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def fit_and_save_preprocessor():
    train_path = os.path.join("artifacts", "train.csv")
    train_df = pd.read_csv(train_path)

    X_train = train_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

    preprocessor = create_preprocessor()
    preprocessor.fit(X_train)

    os.makedirs("artifacts", exist_ok=True)
    output_path = os.path.join("artifacts", "preprocessor.pkl")
    joblib.dump(preprocessor, output_path)
    print(f"Successfully fitted and saved {output_path}")


if __name__ == "__main__":
    fit_and_save_preprocessor()