"""
Feature engineering for the bad-review model (Phase 4).

Prediction moment: the day Olist SENDS the satisfaction survey (`review_ts`,
from review_creation_date). Olist sends it the day after delivery or, if the
parcel has not arrived, about 2 days after the promised date. Every feature
must be known on that day, otherwise the model "sees the future" (data leakage).

Why this moment and not "right after delivery"? Most customers whose order is
late answer the survey BEFORE the parcel arrives. A model built at delivery time
would have to drop those orders, i.e. ~70% of all late orders, which are the
main source of bad reviews.
"""

import numpy as np
import pandas as pd

TARGET = "is_bad_review"

NUMERIC_FEATURES = [
    # delivery status on the survey day
    "days_late_at_survey", "days_since_purchase", "days_promised",
    # delivery outcome: only known if the order was delivered by the survey day (else NaN)
    "delay_days", "days_to_deliver", "carrier_days",
    # seller stage
    "approval_hours", "seller_days",
    # basket and payment
    "n_items", "n_sellers", "revenue", "freight", "freight_ratio",
    "product_weight_g", "product_photos_qty", "installments",
    # history (computed only from the past)
    "seller_prior_reviews", "seller_prior_bad_rate", "customer_prior_orders",
    # calendar
    "purchase_dow",
]
BOOL_FEATURES = ["delivered_at_survey", "is_late", "same_state"]
CATEGORICAL_FEATURES = ["customer_state", "seller_state", "category", "payment_type"]
FEATURES = NUMERIC_FEATURES + BOOL_FEATURES + CATEGORICAL_FEATURES

# Smoothing for the seller's bad-review rate: a seller with 1 review that was
# bad should not get a 100% rate. We blend it with a prior of PRIOR_RATE,
# worth PRIOR_WEIGHT "virtual" reviews. PRIOR_RATE ≈ the overall bad-review rate.
PRIOR_RATE = 0.12
PRIOR_WEIGHT = 5


def modelling_rows(orders):
    """Keep orders with a label, in complete months (Jan 2017 – Aug 2018)."""
    df = orders.dropna(subset=["review_score"])
    df = df[(df["purchase_ts"] >= "2017-01-01") & (df["purchase_ts"] < "2018-09-01")]
    return df.copy()


def add_survey_day_features(df):
    """What Olist knows about the delivery on the survey day.

    review_ts is a date at midnight. The survey is triggered by the delivery, so
    an order delivered on the survey day itself counts as delivered.
    """
    day = pd.Timedelta(days=1)
    survey_day = df["review_ts"].dt.normalize()
    delivered_day = df["delivered_ts"].dt.normalize()
    promised_day = df["estimated_ts"].dt.normalize()

    df["delivered_at_survey"] = delivered_day <= survey_day
    known = df["delivered_at_survey"]

    # Days late so far: vs the delivery day if delivered, else vs the survey day.
    reference_day = delivered_day.where(known, survey_day)
    df["days_late_at_survey"] = (reference_day - promised_day) / day
    df["days_since_purchase"] = (survey_day - df["purchase_ts"]) / day

    # is_late as known ON the survey day. Delivered: the final answer. Not delivered:
    # only whether the promised day has already passed. Usually the survey for an
    # undelivered order goes out ~2 days after the promise, but ~280 orders were
    # surveyed while still in transit BEFORE their promised day; their final
    # is_late (mostly "on time") is the future, so using it would be leakage.
    df["is_late"] = df["is_late"].where(known, survey_day > promised_day)

    # Delivery-outcome columns are NOT known for undelivered orders: hide them.
    for col in ["delay_days", "days_to_deliver"]:
        df[col] = df[col].where(known)
    df["carrier_days"] = ((df["delivered_ts"] - df["carrier_ts"]) / day).where(known)

    # Hand-over to the carrier is known only if it happened before the survey day
    df["seller_days"] = ((df["carrier_ts"] - df["purchase_ts"]) / day).where(
        df["carrier_ts"] < survey_day + day
    )

    # Guards: nothing from after the survey day may reach an undelivered order.
    waiting = ~known
    assert df.loc[waiting, ["delay_days", "days_to_deliver", "carrier_days"]].isna().all().all(), \
        "delivery outcome leaked into orders not yet delivered on the survey day"
    assert (df.loc[waiting, "is_late"] == (survey_day > promised_day)[waiting]).all(), \
        "is_late of undelivered orders must be 'promise already passed on the survey day'"
    assert (df.loc[waiting, "days_late_at_survey"] == ((survey_day - promised_day) / day)[waiting]).all(), \
        "days_late_at_survey of undelivered orders must be measured on the survey day"
    return df


def add_seller_history(df, all_orders):
    """Seller's track record using ONLY reviews answered before this order's survey day.

    A review exists only once the customer ANSWERS the survey (answer_ts), not
    when the survey is sent, so answer_ts is the moment it becomes available.
    """
    reviews = all_orders.dropna(subset=["answer_ts"])[["seller_id", "answer_ts", TARGET]].copy()
    reviews["bad"] = reviews[TARGET].astype(int)
    reviews = reviews.sort_values("answer_ts")

    by_seller = reviews.groupby("seller_id")
    reviews["seller_prior_reviews"] = by_seller.cumcount() + 1
    reviews["seller_prior_bad"] = by_seller["bad"].cumsum()

    # merge_asof = "for each order, take the most recent seller row answered
    # strictly before the survey day" (a time-aware join)
    left = df[["order_id", "seller_id", "review_ts"]].sort_values("review_ts")
    hist = reviews[["seller_id", "answer_ts", "seller_prior_reviews", "seller_prior_bad"]]
    merged = pd.merge_asof(
        left, hist,
        left_on="review_ts", right_on="answer_ts",
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
    df = add_survey_day_features(df)

    df["approval_hours"] = (df["approved_ts"] - df["purchase_ts"]) / pd.Timedelta(hours=1)
    df["freight_ratio"] = df["freight"] / df["revenue"].replace(0, np.nan)
    df["purchase_dow"] = df["purchase_ts"].dt.dayofweek

    df = add_seller_history(df, orders)
    df = add_customer_history(df, orders)

    for col in BOOL_FEATURES:
        df[col] = df[col].astype(float)       # True/False → 1.0/0.0 (NaN stays NaN)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("unknown").astype("category")
    df[TARGET] = df[TARGET].astype(int)

    return df[["order_id", "purchase_ts"] + FEATURES + [TARGET]].reset_index(drop=True)
