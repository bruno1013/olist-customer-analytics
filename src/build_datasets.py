"""
Build the analysis-ready datasets from the SQLite database.

Usage (from the project root, after src/load_data.py):
    .venv/bin/python src/build_datasets.py

Result: data/processed/orders.parquet — one row per delivered order
(query in sql/05_order_dataset.sql).
"""

import sqlite3
from pathlib import Path

import pandas as pd

# --- 1. Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "olist.db"
SQL_DIR = PROJECT_ROOT / "sql"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

# SQLite stores dates as text and booleans as 0/1, so we fix the types here.
DATE_COLUMNS = [
    "purchase_ts", "approved_ts", "carrier_ts",
    "delivered_ts", "estimated_ts", "review_ts",
]
# "boolean" (capital-B pandas type) allows missing values, e.g. is_bad_review
# is missing for orders without a review.
BOOL_COLUMNS = ["same_state", "is_late", "is_bad_review"]


def run_sql_file(conn, file_name):
    """Run a .sql file that contains ONE query and return a DataFrame."""
    query = (SQL_DIR / file_name).read_text()
    return pd.read_sql_query(query, conn)


def build_orders(conn):
    """One row per delivered order with delivery, basket, payment and review info."""
    df = run_sql_file(conn, "05_order_dataset.sql")

    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col])
    for col in BOOL_COLUMNS:
        df[col] = df[col].astype("boolean")

    # Sanity check: the query must return exactly one row per order.
    assert df["order_id"].is_unique, "duplicate orders — check the JOINs"
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        orders = build_orders(conn)
    finally:
        conn.close()

    out_path = OUT_DIR / "orders.parquet"
    orders.to_parquet(out_path, index=False)
    print(f"orders: {len(orders):,} rows, {orders.shape[1]} columns → {out_path}")


if __name__ == "__main__":
    main()
