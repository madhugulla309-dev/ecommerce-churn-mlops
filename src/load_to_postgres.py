# requirements to install
pip install psycopg2-binary sqlalchemy pandas

# Main code to load the csv files into the pgsql

# src/load_to_postgres.py
import os
import pandas as pd
from sqlalchemy import create_engine

# 1. Connection string: Use the actual postgres user (usually 'postgres'), not the pgAdmin display name
# Format: postgresql+psycopg2://<username>:<password>@<host>:<port>/<dbname>
DB_URL = "postgresql+psycopg2://postgres:madhu123@localhost:5432/olist_ecommerce"
engine = create_engine(DB_URL)

# 2. Raw string (r"...") prevents the Windows '\U' unicode escape syntax error
RAW_PATH = r"C:\Users\Madhu\Desktop\Brazilian E.Commerce Public Dataset\archive"

# 3. Explicit date columns to ensure PostgreSQL creates TIMESTAMP types instead of TEXT
DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": ["shipping_limit_date"],
    "order_reviews": ["review_creation_date", "review_answer_timestamp"],
}

FILES_TO_TABLES = {
    "olist_customers_dataset.csv": "customers",
    "olist_orders_dataset.csv": "orders",
    "olist_order_items_dataset.csv": "order_items",
    "olist_order_payments_dataset.csv": "order_payments",
    "olist_order_reviews_dataset.csv": "order_reviews",
}


def ingest_raw_csvs():
    # Verify the source directory exists before running
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Directory not found: {RAW_PATH}")

    for csv_file, table_name in FILES_TO_TABLES.items():
        file_path = os.path.join(RAW_PATH, csv_file)

        if not os.path.exists(file_path):
            print(f"Warning: File {csv_file} not found at {file_path}. Skipping.")
            continue

        print(f"Loading {csv_file} into PostgreSQL table '{table_name}'...")

        # Parse date columns if specified for the table
        parse_dates = DATE_COLUMNS.get(table_name, False)

        # low_memory=False prevents mixed-type inference warnings
        df = pd.read_csv(file_path, parse_dates=parse_dates, low_memory=False)

        # Write to PostgreSQL in batches
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists="replace",
            index=False,
            chunksize=10000,
            method="multi",  # Speeds up batch insertion
        )
        print(f"Successfully loaded {len(df):,} rows into '{table_name}'.\n")


if __name__ == "__main__":
    ingest_raw_csvs()