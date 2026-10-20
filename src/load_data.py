"""
Load the 9 raw Olist CSV files into a single SQLite database.

Usage (from the project root):
    .venv/bin/python src/load_data.py

Result: data/olist.db with one table per CSV, using short table names.
Running it again rebuilds the tables from scratch (safe to re-run).
"""

import sqlite3
from pathlib import Path

import pandas as pd

# --- 1. Paths -----------------------------------------------------------------
# Build paths relative to this file, so the script works no matter
# which folder you run it from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DB_PATH = PROJECT_ROOT / "data" / "olist.db"

# --- 2. Mapping: short table name -> CSV file name ---------------------------
TABLES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

# Zip code prefixes look like numbers but are really labels ("01037").
# Reading them as text keeps the leading zero.
ZIP_COLUMNS = {
    "customer_zip_code_prefix": str,
    "seller_zip_code_prefix": str,
    "geolocation_zip_code_prefix": str,
}


def load_all():
    """Read every CSV and write it to SQLite as its own table."""
    # Opening a connection creates the .db file if it doesn't exist yet.
    conn = sqlite3.connect(DB_PATH)

    try:
        for table_name, file_name in TABLES.items():
            csv_path = RAW_DIR / file_name

            # encoding="utf-8-sig" strips the invisible BOM character that
            # some files (like the translation one) start with.
            df = pd.read_csv(csv_path, dtype=ZIP_COLUMNS, encoding="utf-8-sig")

            # if_exists="replace" drops the old table first, so re-running
            # never duplicates rows. index=False skips pandas' row numbers.
            df.to_sql(table_name, conn, if_exists="replace", index=False)

            print(f"{table_name:<22} {len(df):>9,} rows  {df.shape[1]:>2} columns")
    finally:
        # Always close the connection, even if something above fails.
        conn.close()

    print(f"\nDone. Database saved to {DB_PATH}")


if __name__ == "__main__":
    load_all()
