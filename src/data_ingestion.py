# src/data_ingestion.py
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect
from sklearn.model_selection import train_test_split

# 1. Matches your PostgreSQL credentials from load_to_postgres.py
DB_URL = "postgresql+psycopg2://postgres:madhu123@localhost:5432/olist_ecommerce"


def run_data_ingestion():
    print("Connecting to PostgreSQL and fetching 'customer_rfm_features'...")
    engine = create_engine(DB_URL)

    # Verify the SQL feature table/view exists before querying
    inspector = inspect(engine)
    available_relations = inspector.get_table_names() + inspector.get_view_names()
    if "customer_rfm_features" not in available_relations:
        raise RuntimeError(
            "Table or view 'customer_rfm_features' not found in 'olist_ecommerce'. "
            "Run your SQL RFM feature engineering script first."
        )

    query = "SELECT * FROM customer_rfm_features;"
    df = pd.read_sql(query, engine)
    print(f"Pulled {len(df):,} unique customer records from PostgreSQL.")

    if df.empty:
        raise ValueError("Query returned 0 rows from 'customer_rfm_features'.")

    # --- PANDAS FEATURE ENGINEERING ---
    # 1. Average Order Value (Spend per order, safe against 0 or NaN)
    df["total_monetary_spend"] = df["total_monetary_spend"].fillna(0.0)
    df["frequency_orders"] = df["frequency_orders"].fillna(1).replace(0, 1)
    df["avg_order_value"] = np.round(
        df["total_monetary_spend"] / df["frequency_orders"], 2
    )

    # 2. Binary flag: Did the customer experience a late delivery? (Handle NULL delays safely)
    df["avg_delivery_delay_days"] = df["avg_delivery_delay_days"].fillna(0.0)
    df["is_late_delivery"] = (df["avg_delivery_delay_days"] > 0).astype(int)

    # 3. Binary flag: Did the customer leave a written review? (Handle NULL/empty strings safely)
    review_series = df["latest_review_text"].fillna("").astype(str).str.strip()
    df["has_written_review"] = (
        (review_series != "") & (review_series.str.lower() != "no written review")
    ).astype(int)

    # Save master cleaned table (pre-clipping) for EDA & Power BI
    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    master_path = os.path.join(artifacts_dir, "master_cleaned_customers.csv")
    df.to_csv(master_path, index=False)
    print(f"Saved master dataset to '{master_path}'.")

    # Ensure target column is clean integer before stratified split
    df["is_churned"] = df["is_churned"].astype(int)

    # Stratified Train/Test Split (80% Train, 20% Test) preserving exact Churn %
    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["is_churned"]
    )

    # 4. Cap extreme outliers using ONLY train_df thresholds to prevent data leakage
    spend_cap = train_df["total_monetary_spend"].quantile(0.99)
    delay_cap = train_df["avg_delivery_delay_days"].quantile(0.99)

    for split_df in (train_df, test_df):
        split_df["total_monetary_spend"] = split_df["total_monetary_spend"].clip(
            upper=spend_cap
        )
        split_df["avg_delivery_delay_days"] = split_df[
            "avg_delivery_delay_days"
        ].clip(lower=-30, upper=delay_cap)

    train_path = os.path.join(artifacts_dir, "train.csv")
    test_path = os.path.join(artifacts_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(
        f"Saved train.csv ({len(train_df):,} rows) and test.csv ({len(test_df):,} rows) in '{artifacts_dir}/'"
    )
    print(f"Train Churn Rate: {train_df['is_churned'].mean():.2%} | Test Churn Rate: {test_df['is_churned'].mean():.2%}")


if __name__ == "__main__":
    run_data_ingestion()