import pandas as pd
import streamlit as st
from utils import REPO_URL, load, pct

impact = load("impact").set_index("problem")
segments = load("segments").set_index("segment")
late = impact.loc["Late deliveries"]
early_answer = impact.loc["Survey answered before delivery"]
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
        "Recommendation": "Don't survey customers who are still waiting: send a status update instead, "
                          "and survey after delivery",
        "Evidence": "70% of late-order customers answer the survey before the parcel arrives; 81% of them "
                    "give 1–2★. Half of all \"not received\" complaints are written before delivery.",
        "Size of the prize": f"Up to {pct(early_answer['share_of_all_bad_reviews'], 0)} of all bad reviews "
                             f"(~{early_answer['bad_reviews_avoidable']:,.0f})",
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
        "Evidence": "On the survey day, contacting the riskiest 10% of orders reaches 38% of bad reviews "
                    "(3.8x random; a 'late orders first' rule reaches 24%).",
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
| Data preparation | One-row-per-order dataset, leakage-safe features | [`01_eda_logistics`]({REPO_URL}/blob/main/notebooks/01_eda_logistics.ipynb) |
| Modeling | RFM + K-Means, LightGBM risk model, text classifier, NMF | [`02`]({REPO_URL}/blob/main/notebooks/02_customers.ipynb) · [`03`]({REPO_URL}/blob/main/notebooks/03_bad_review_model.ipynb) · [`04`]({REPO_URL}/blob/main/notebooks/04_review_text.ipynb) |
| Evaluation | Statistical tests, time-based split, PR-AUC and lift | 01–04 |
| Deployment | This dashboard | [`app/`]({REPO_URL}/tree/main/app) |
"""
)
