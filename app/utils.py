"""Shared helpers for the dashboard: paths, cached loading, colours and chart styling."""

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
MODEL_DIR = APP_DIR / "models"
REPO_URL = "https://github.com/bruno1013/olist-customer-analytics"

# src/ holds code shared with the notebooks (text normalisation, complaint
# themes, feature list). The whole repo is deployed, so we can import it.
sys.path.insert(0, str(APP_DIR.parent / "src"))

GREY, RED, BLUE, DARK, GREEN, SAND = "#B0B7BF", "#D1495B", "#2E86AB", "#2B2D42", "#81B29A", "#F2CC8F"
REGION_COLORS = {"North": "#E07A5F", "Northeast": RED, "Center-West": SAND,
                 "Southeast": BLUE, "South": GREEN}

# Readable names for the model features
FEATURE_NAMES = {
    "n_items": "Items in the order",
    "seller_prior_bad_rate": "Seller's past bad-review rate",
    "days_to_deliver": "Days to deliver",
    "category": "Product category",
    "seller_days": "Days until seller ships",
    "freight": "Freight cost",
    "delay_days": "Days late (vs promise)",
    "customer_state": "Customer state",
    "n_sellers": "Sellers in the order",
    "seller_prior_reviews": "Seller's past reviews",
    "days_promised": "Days promised",
    "product_weight_g": "Product weight",
    "carrier_days": "Days with the carrier",
    "revenue": "Order value",
    "freight_ratio": "Freight / order value",
    "installments": "Installments",
    "approval_hours": "Hours to approve payment",
    "product_photos_qty": "Product photos",
    "seller_state": "Seller state",
    "payment_type": "Payment type",
    "purchase_dow": "Day of week",
    "customer_prior_orders": "Customer's past orders",
    "is_late": "Late (vs promise)",
    "delivered_at_survey": "Delivered by survey day",
    "days_late_at_survey": "Days late on survey day",
    "days_since_purchase": "Days since purchase (survey day)",
    "same_state": "Seller in same state",
}


@st.cache_data
def load(name):
    """Read one of the small parquet files in app/data (cached across reruns)."""
    return pd.read_parquet(DATA_DIR / f"{name}.parquet")


@st.cache_resource
def load_model(name):
    """Load a trained model once and share it between users."""
    return joblib.load(MODEL_DIR / f"{name}.joblib")


def style(fig, height=380, title=None):
    """Apply the dashboard's chart style to a Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        title={"text": title, "x": 0, "font": {"size": 16}} if title else None,
        margin={"l": 10, "r": 10, "t": 50 if title else 10, "b": 10},
        font={"size": 13, "color": DARK},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "title": None},
    )
    return fig


def pct(x, decimals=1):
    return f"{x:.{decimals}%}"


def brl(x):
    """Brazilian reais, compact: R$ 13.2M, R$ 137."""
    if abs(x) >= 1e6:
        return f"R$ {x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"R$ {x / 1e3:.0f}k"
    return f"R$ {x:,.0f}"
