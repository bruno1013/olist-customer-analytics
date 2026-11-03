"""
Build the small, aggregated files the Streamlit dashboard reads.

Usage (from the project root, after running notebooks 02-04):
    .venv/bin/python src/build_app_data.py

Result: app/data/*.parquet and app/models/*.joblib. They are small enough to
commit, so the deployed app never needs the raw data or the SQLite database.
"""

import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from features import FEATURES, build_features
from text import THEMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"
APP_DATA = PROJECT_ROOT / "app" / "data"
APP_MODELS = PROJECT_ROOT / "app" / "models"

REGION = {
    **dict.fromkeys(["AC", "AP", "AM", "PA", "RO", "RR", "TO"], "North"),
    **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "Northeast"),
    **dict.fromkeys(["DF", "GO", "MT", "MS"], "Center-West"),
    **dict.fromkeys(["ES", "MG", "RJ", "SP"], "Southeast"),
    **dict.fromkeys(["PR", "RS", "SC"], "South"),
}


def save(df, name):
    df.to_parquet(APP_DATA / f"{name}.parquet", index=False)
    print(f"{name:<22} {len(df):>6} rows")


def build_overview(orders, reviewed, customers):
    complete = orders[(orders["purchase_ts"] >= "2017-01-01") & (orders["purchase_ts"] < "2018-09-01")]
    kpis = pd.DataFrame([{
        "orders": len(orders),
        "customers": len(customers),
        "revenue": orders["revenue"].sum(),
        "aov": orders["revenue"].mean(),
        "avg_review": reviewed["review_score"].mean(),
        "late_rate": orders["is_late"].mean(),
        "bad_review_rate": reviewed["is_bad_review"].mean(),
        "repeat_rate": customers["repeat"].mean(),
        "avg_days_to_deliver": orders["days_to_deliver"].mean(),
        "avg_days_promised": orders["days_promised"].mean(),
    }])
    save(kpis, "kpis")

    monthly = complete.groupby("purchase_month").agg(
        orders=("order_id", "count"),
        revenue=("revenue", "sum"),
        late_rate=("is_late", "mean"),
        bad_review_rate=("is_bad_review", "mean"),
        avg_review=("review_score", "mean"),
    ).reset_index().rename(columns={"purchase_month": "month"})
    for col in ["late_rate", "bad_review_rate"]:
        monthly[col] = monthly[col].astype(float)
    save(monthly, "monthly")


def build_delivery(orders, reviewed, centroids):
    bins = [-np.inf, -7, -1, 0, 3, 7, np.inf]
    labels = ["7+ days early", "1–6 days early", "On the promised day",
              "1–3 days late", "4–7 days late", "8+ days late"]
    buckets = (
        reviewed.assign(bucket=pd.cut(reviewed["delay_days"], bins=bins, labels=labels))
        .groupby("bucket", observed=True)
        .agg(orders=("order_id", "count"), avg_review=("review_score", "mean"),
             bad_review_rate=("is_bad_review", "mean"))
        .reset_index()
    )
    buckets["bucket"] = buckets["bucket"].astype(str)
    buckets["bad_review_rate"] = buckets["bad_review_rate"].astype(float)
    save(buckets, "delay_buckets")

    states = reviewed.groupby("customer_state").agg(
        orders=("order_id", "count"),
        late_rate=("is_late", "mean"),
        avg_days_to_deliver=("days_to_deliver", "mean"),
        avg_review=("review_score", "mean"),
        bad_review_rate=("is_bad_review", "mean"),
    ).reset_index().rename(columns={"customer_state": "state"})
    for col in ["late_rate", "bad_review_rate"]:
        states[col] = states[col].astype(float)
    states["region"] = states["state"].map(REGION)
    states = states.merge(centroids[["state", "lat", "lng"]], on="state", how="left")
    save(states, "states")

    st = orders.dropna(subset=["carrier_ts"]).copy()
    st["seller_days"] = (st["carrier_ts"] - st["purchase_ts"]) / pd.Timedelta(days=1)
    st["carrier_days"] = (st["delivered_ts"] - st["carrier_ts"]) / pd.Timedelta(days=1)
    st = st[(st["seller_days"] >= 0) & (st["carrier_days"] >= 0)]
    stages = st.groupby("is_late")[["seller_days", "carrier_days"]].median().reset_index()
    stages["delivery"] = np.where(stages["is_late"], "Late", "On time")
    save(stages[["delivery", "seller_days", "carrier_days"]], "stages")

    basket = reviewed.assign(
        items=pd.cut(reviewed["n_items"], [0, 1, 2, np.inf], labels=["1 item", "2 items", "3+ items"]).astype(str),
        sellers=np.where(reviewed["n_sellers"] >= 2, "2+ sellers", "1 seller"),
    )
    rows = []
    for col, kind in [("items", "Items in the order"), ("sellers", "Sellers in the order")]:
        g = basket.groupby(col).agg(orders=("order_id", "count"), bad_review_rate=("is_bad_review", "mean"))
        for label, r in g.iterrows():
            rows.append({"kind": kind, "group": label, "orders": r["orders"],
                         "bad_review_rate": float(r["bad_review_rate"])})
    save(pd.DataFrame(rows), "basket")


def build_customers(orders, customers):
    order = ["Loyal", "Lapsed loyal", "Recent big spender", "Recent", "At-risk big spender", "Hibernating"]
    seg = customers.groupby("segment").agg(
        customers=("customer_unique_id", "count"),
        revenue=("monetary", "sum"),
        avg_spend=("monetary", "mean"),
        avg_recency_days=("recency_days", "mean"),
        avg_review=("avg_review", "mean"),
    ).reindex(order).reset_index()
    seg["customer_share"] = seg["customers"] / seg["customers"].sum()
    seg["revenue_share"] = seg["revenue"] / seg["revenue"].sum()
    save(seg, "segments")

    sample = customers.sample(5000, random_state=42)[["recency_days", "monetary", "cluster_name"]]
    save(sample, "clusters_sample")

    c = orders[["customer_unique_id", "purchase_ts"]].copy()
    c["month_idx"] = c["purchase_ts"].dt.year * 12 + c["purchase_ts"].dt.month
    c["cohort_idx"] = c.groupby("customer_unique_id")["month_idx"].transform("min")
    c["months_after"] = c["month_idx"] - c["cohort_idx"]
    c["cohort"] = c.groupby("customer_unique_id")["purchase_ts"].transform("min").dt.strftime("%Y-%m")
    active = c.groupby(["cohort", "months_after"])["customer_unique_id"].nunique().unstack()
    retention = active.div(active[0], axis=0).loc["2017-01":"2018-06", 1:12]
    cohort = retention.stack().rename("retention").reset_index()
    save(cohort, "cohort")


def build_model(orders):
    shutil.copy(PROJECT_ROOT / "models" / "bad_review_lgbm.joblib", APP_MODELS)
    shutil.copy(PROJECT_ROOT / "models" / "review_text_model.joblib", APP_MODELS)

    save(pd.read_parquet(PROCESSED / "model_scores.parquet"), "model_scores")
    save(pd.read_parquet(PROCESSED / "shap_importance.parquet"), "shap_importance")
    preds = pd.read_parquet(PROCESSED / "test_predictions.parquet")[["is_bad_review", "risk"]]
    save(preds, "test_predictions")

    # What-if template: a "typical" training order (median numbers, most common
    # categories). The app changes a few fields and asks the model for a risk.
    feats = build_features(orders)
    train = feats[feats["purchase_ts"] < "2018-03-01"][FEATURES]
    row = {}
    for col in FEATURES:
        if isinstance(train[col].dtype, pd.CategoricalDtype):
            row[col] = pd.Categorical([train[col].mode()[0]], categories=train[col].cat.categories)
        else:
            row[col] = [train[col].median()]
    save(pd.DataFrame(row), "whatif_template")


def build_text():
    themes = pd.read_parquet(PROCESSED / "review_themes.parquet")
    bad = themes[themes["bad"] == 1].copy()
    bad["seller_group"] = np.where(bad["n_sellers"] >= 2, "2+ sellers", "1 seller")

    rows = []
    for theme in THEMES:
        rows.append({"theme": theme, "group": "All bad reviews", "share": bad[theme].mean()})
        for grp, g in bad.groupby("seller_group"):
            rows.append({"theme": theme, "group": grp, "share": g[theme].mean()})
    save(pd.DataFrame(rows), "themes")

    weights = pd.read_parquet(PROCESSED / "text_model_weights.parquet").sort_values("weight")
    save(pd.concat([weights.head(20), weights.tail(20)]), "text_weights")

    orders = pd.read_parquet(PROCESSED / "orders.parquet")[["order_id", "review_comment"]]
    bad = bad.merge(orders, on="order_id")
    bad = bad[bad["review_comment"].str.len().between(25, 160)]
    examples = []
    for theme in THEMES:
        pick = bad[bad[theme]].sample(6, random_state=7)
        examples += [{"theme": theme, "comment": t.strip()} for t in pick["review_comment"]]
    save(pd.DataFrame(examples), "theme_examples")


def build_impact(reviewed):
    """Upper-bound estimates: bad reviews avoided if a problem group behaved like the rest."""
    total_bad = reviewed["is_bad_review"].sum()
    rows = []
    for name, mask in [
        ("Late deliveries", reviewed["is_late"].fillna(False).astype(bool)),
        # customers who answered the survey while the parcel was still on its way
        ("Survey answered before delivery", reviewed["answer_ts"] < reviewed["delivered_ts"]),
        ("Multi-seller orders", reviewed["n_sellers"] >= 2),
        ("Multi-item orders (2+ items)", reviewed["n_items"] >= 2),
    ]:
        group, rest = reviewed[mask], reviewed[~mask]
        excess = len(group) * (group["is_bad_review"].mean() - rest["is_bad_review"].mean())
        rows.append({
            "problem": name,
            "orders": len(group),
            "share_of_orders": len(group) / len(reviewed),
            "bad_rate_group": float(group["is_bad_review"].mean()),
            "bad_rate_rest": float(rest["is_bad_review"].mean()),
            "bad_reviews_avoidable": float(excess),
            "share_of_all_bad_reviews": float(excess / total_bad),
        })
    save(pd.DataFrame(rows), "impact")


def main():
    APP_DATA.mkdir(parents=True, exist_ok=True)
    APP_MODELS.mkdir(parents=True, exist_ok=True)

    orders = pd.read_parquet(PROCESSED / "orders.parquet")
    reviewed = orders.dropna(subset=["review_score"])
    customers = pd.read_parquet(PROCESSED / "customer_segments.parquet")

    conn = sqlite3.connect(PROJECT_ROOT / "data" / "olist.db")
    try:
        centroids = pd.read_sql_query((PROJECT_ROOT / "sql" / "06_state_centroids.sql").read_text(), conn)
    finally:
        conn.close()

    build_overview(orders, reviewed, customers)
    build_delivery(orders, reviewed, centroids)
    build_customers(orders, customers)
    build_model(orders)
    build_text()
    build_impact(reviewed)

    total = sum(f.stat().st_size for f in APP_DATA.glob("*")) + sum(f.stat().st_size for f in APP_MODELS.glob("*"))
    print(f"\nTotal app data + models: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
