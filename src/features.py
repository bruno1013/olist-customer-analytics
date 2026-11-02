"""
Feature engineering for the bad-review model (Phase 4).

Prediction moment: right after the order is DELIVERED. Every feature must be
known at that moment, otherwise the model "sees the future" (data leakage).
"""

import pandas as pd

TARGET = "is_bad_review"

NUMERIC_FEATURES = [
    # delivery
    "delay_days", "days_to_deliver", "days_promised",
    "approval_hours", "seller_days", "carrier_days",
    # basket and payment
    "n_items", "n_sellers", "revenue", "freight", "freight_ratio",
    "product_weight_g", "product_photos_qty", "installments",
    # history (computed only from the past)
    "seller_prior_reviews", "seller_prior_bad_rate", "customer_prior_orders",
    # calendar
    "purchase_dow",
]
BOOL_FEATURES = ["is_late", "same_state"]
CATEGORICAL_FEATURES = ["customer_state", "seller_state", "category", "payment_type"]
FEATURES = NUMERIC_FEATURES + BOOL_FEATURES + CATEGORICAL_FEATURES

# Smoothing for the seller's bad-review rate: a seller with 1 review that was
# bad should not get a 100% rate. We blend it with a prior of PRIOR_RATE,
# worth PRIOR_WEIGHT "virtual" reviews. PRIOR_RATE ≈ the overall bad-review
# rate for reviews written after delivery.
PRIOR_RATE = 0.10
PRIOR_WEIGHT = 5


def modelling_rows(orders):
    """Keep orders where a prediction at delivery time makes sense.

    - must have a review (we need the label)
    - the review must be written on/after the delivery day: if the customer
      already reviewed before the parcel arrived, there is nothing to predict
    - complete months only (Jan 2017 – Aug 2018)
    """
    df = orders.dropna(subset=["review_score"])
    df = df[df["review_ts"].dt.normalize() >= df["delivered_ts"].dt.normalize()]
    df = df[(df["purchase_ts"] >= "2017-01-01") & (df["purchase_ts"] < "2018-09-01")]
    return df.copy()


def add_seller_history(df, all_orders):
    """Seller's track record using ONLY reviews available before this delivery.

    review_ts is a date at midnight, so a review dated day D may have been written
    late on day D. We treat it as available from D + 1 day to be safe.
    """
    reviews = all_orders.dropna(subset=["review_ts"])[["seller_id", "review_ts", TARGET]].copy()
    reviews["available_ts"] = reviews["review_ts"] + pd.Timedelta(days=1)
    reviews["bad"] = reviews[TARGET].astype(int)
    reviews = reviews.sort_values("available_ts")

    by_seller = reviews.groupby("seller_id")
    reviews["seller_prior_reviews"] = by_seller.cumcount() + 1
    reviews["seller_prior_bad"] = by_seller["bad"].cumsum()

    # merge_asof = "for each order, take the most recent seller row whose
    # available_ts is strictly before the delivery time" (a time-aware join)
    left = df[["order_id", "seller_id", "delivered_ts"]].sort_values("delivered_ts")
    hist = reviews[["seller_id", "available_ts", "seller_prior_reviews", "seller_prior_bad"]]
    merged = pd.merge_asof(
        left, hist,
        left_on="delivered_ts", right_on="available_ts",
        by="seller_id", allow_exact_matches=False,
    )
    merged[["seller_prior_reviews", "seller_prior_bad"]] = (
        merged[["seller_prior_reviews", "seller_prior_bad"]].fillna(0)
    )
    merged["seller_prior_bad_rate"] = (
        (merged["seller_prior_bad"] + PRIOR_RATE * PRIOR_WEIGHT)
        / (merged["seller_prior_reviews"] + PRIOR_WEIGHT)
    )
    return df.merge(
        merged[["order_id", "seller_prior_reviews", "seller_prior_bad_rate"]],
        on="order_id", how="left",
    )


def add_customer_history(df, all_orders):
    """How many orders the customer placed before this one."""
    history = all_orders[["order_id", "customer_unique_id", "purchase_ts"]].sort_values("purchase_ts")
    history["customer_prior_orders"] = history.groupby("customer_unique_id").cumcount()
    return df.merge(history[["order_id", "customer_prior_orders"]], on="order_id", how="left")


def build_features(orders):
    """Return the modelling table: features + target + purchase_ts (for the time split)."""
    df = modelling_rows(orders)

    hours = pd.Timedelta(hours=1)
    days = pd.Timedelta(days=1)
    df["approval_hours"] = (df["approved_ts"] - df["purchase_ts"]) / hours
    df["seller_days"] = (df["carrier_ts"] - df["purchase_ts"]) / days
    df["carrier_days"] = (df["delivered_ts"] - df["carrier_ts"]) / days
    df["freight_ratio"] = df["freight"] / df["revenue"]
    df["purchase_dow"] = df["purchase_ts"].dt.dayofweek

    df = add_seller_history(df, orders)
    df = add_customer_history(df, orders)

    for col in BOOL_FEATURES:
        df[col] = df[col].astype(float)       # True/False → 1.0/0.0 (NaN stays NaN)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("unknown").astype("category")
    df[TARGET] = df[TARGET].astype(int)

    return df[["order_id", "purchase_ts"] + FEATURES + [TARGET]].reset_index(drop=True)
