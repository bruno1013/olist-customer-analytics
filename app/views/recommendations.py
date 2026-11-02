import pandas as pd
import streamlit as st
from utils import REPO_URL, load, pct

impact = load("impact").set_index("problem")
segments = load("segments").set_index("segment")
late = impact.loc["Late deliveries"]
multi_item = impact.loc["Multi-item orders (2+ items)"]
at_risk = segments.loc["At-risk big spender"]

st.title("📋 Recommendations")
st.markdown("What Olist should do, ranked by the share of bad reviews or revenue at stake.")

recs = pd.DataFrame([
    {
        "Priority": 1,
        "Recommendation": "Carrier SLAs and monitoring, starting with the Northeast",
        "Evidence": "Late orders: 62% bad reviews vs 9% on time (6.7x, holds in every state). "
                    "Late orders spend 26 median days with the carrier vs 7.",
        "Size of the prize": f"Up to {pct(late['share_of_all_bad_reviews'], 0)} of all bad reviews "
                             f"(~{late['bad_reviews_avoidable']:,.0f})",
    },
    {
        "Priority": 2,
        "Recommendation": "Proof of delivery (photo / signature) and carrier audits",
        "Evidence": "\"Not received\" is the #1 complaint (30% of bad comments) — 20% even on orders "
                    "delivered on time.",
        "Size of the prize": "The largest complaint theme",
    },
    {
        "Priority": 3,
        "Recommendation": "Fix split shipments: warn customers, track each parcel, mark delivered only "
                          "when everything arrives",
        "Evidence": "2+ sellers: 47% bad reviews even on time; 58% of their complaints mention missing items.",
        "Size of the prize": f"Multi-item orders: up to {pct(multi_item['share_of_all_bad_reviews'], 0)} "
                             f"of bad reviews",
    },
    {
        "Priority": 4,
        "Recommendation": "Proactive customer-service contacts using the risk model",
        "Evidence": "Contacting the riskiest 10% of delivered orders reaches 30% of bad reviews (3x random).",
        "Size of the prize": "Prevent or soften bad reviews before they are written",
    },
    {
        "Priority": 5,
        "Recommendation": "Win-back campaign for at-risk big spenders",
        "Evidence": f"{at_risk['customers']:,.0f} customers, {pct(at_risk['revenue_share'], 0)} of revenue, "
                    f"~{at_risk['avg_recency_days'] / 30:.0f} months without buying.",
        "Size of the prize": f"{pct(at_risk['revenue_share'], 0)} of historical revenue",
    },
    {
        "Priority": 6,
        "Recommendation": "Second-purchase programme (post-purchase emails, vouchers, cross-sell)",
        "Evidence": "Only 3% of customers repeat; 1.95% return within 180 days, and even happy customers "
                    "rarely come back.",
        "Size of the prize": "Retention is the biggest long-term lever",
    },
    {
        "Priority": 7,
        "Recommendation": "Automatic triage of review comments",
        "Evidence": "Text model PR-AUC 0.80; routes \"not received\" and \"missing item\" complaints "
                    "to a priority queue.",
        "Size of the prize": "Faster resolution, less manual reading",
    },
])
st.dataframe(recs, hide_index=True, width="stretch", height=440,
             column_config={"Priority": st.column_config.NumberColumn(width="small")})
st.caption(
    "\"Up to\" figures are upper bounds: bad reviews that would disappear if the problem group had the same "
    "bad-review rate as all other orders. Groups overlap, so the figures should not be added up. "
    "The data are observational, so effects are associations, not proven causes."
)

st.subheader("Method")
st.markdown(
    f"""
The project follows **CRISP-DM**:

| Phase | What was done | Notebook |
|---|---|---|
| Business understanding | Retention problem, reviews as a reputation driver | — |
| Data understanding | SQL on 9 tables (JOINs, CTEs, window functions) | [`sql/`]({REPO_URL}/tree/main/sql) |
| Data preparation | One-row-per-order dataset, leakage-safe features | [`02_eda_logistics`]({REPO_URL}/blob/main/notebooks/02_eda_logistics.ipynb) |
| Modeling | RFM + K-Means, LightGBM risk model, text classifier, NMF | [`03`]({REPO_URL}/blob/main/notebooks/03_customers.ipynb) · [`04`]({REPO_URL}/blob/main/notebooks/04_bad_review_model.ipynb) · [`05`]({REPO_URL}/blob/main/notebooks/05_review_text.ipynb) |
| Evaluation | Statistical tests, time-based split, PR-AUC and lift | 02–05 |
| Deployment | This dashboard | [`app/`]({REPO_URL}/tree/main/app) |
"""
)
