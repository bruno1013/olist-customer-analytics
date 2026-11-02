import numpy as np
import plotly.graph_objects as go
import streamlit as st
from utils import DARK, FEATURE_NAMES, GREY, RED, load, load_model, pct, style

from features import FEATURES

scores = load("model_scores")
preds = load("test_predictions")
shap_imp = load("shap_importance")
template = load("whatif_template")
model = load_model("bad_review_lgbm")

st.title("🤖 Bad-review risk model")
st.markdown(
    "Right after an order is delivered, the model scores how likely the customer is to leave a "
    "1–2★ review, so customer service can **contact the riskiest orders first**."
)

with st.expander("How the model was built (and how leakage was avoided)"):
    st.markdown(
        """
- **Target:** review score 1–2. **Prediction moment:** right after delivery.
- **Only information known at delivery** is used — never the review text or date. Reviews written
  *before* the parcel arrived are excluded (nothing left to predict).
- **Seller history** (past bad-review rate) is computed only from reviews available before each delivery.
- **Time-based split:** train Jan 2017–Feb 2018, validate Mar–Apr 2018, test **May–Aug 2018**.
- Models: dummy baseline → logistic regression → decision tree → **LightGBM**, tuned on validation;
  the test set was used once.
"""
    )

st.subheader("1. How good is it? (unseen test months)")
st.dataframe(
    scores[["model", "PR-AUC", "ROC-AUC", "precision@10%", "recall@10%", "lift@10%", "accuracy@0.5"]],
    hide_index=True, width="stretch",
    column_config={
        "model": "Model",
        "PR-AUC": st.column_config.NumberColumn(format="%.3f"),
        "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
        "precision@10%": st.column_config.NumberColumn("Precision top 10%", format="percent"),
        "recall@10%": st.column_config.NumberColumn("Recall top 10%", format="percent"),
        "lift@10%": st.column_config.NumberColumn("Lift top 10%", format="%.1fx"),
        "accuracy@0.5": st.column_config.NumberColumn("Accuracy", format="percent"),
    },
)
st.caption(
    "Accuracy is misleading here: the dummy model scores ~92% by never flagging anything. "
    "PR-AUC of a random model ≈ the bad-review rate (0.085)."
)

st.subheader("2. Business view: how many orders should we contact?")
share = st.slider("Share of delivered orders customer service can contact", 1, 50, 10, format="%d%%") / 100

y = preds["is_bad_review"].to_numpy()
risk = preds["risk"].to_numpy()
order = np.argsort(-risk)
captured = np.cumsum(y[order]) / y.sum()
contacted = np.arange(1, len(y) + 1) / len(y)
k = max(int(len(y) * share), 1)
precision = y[order][:k].mean()

cols = st.columns(4)
cols[0].metric("Orders contacted", f"{k:,}")
cols[1].metric("Bad reviews reached", pct(captured[k - 1], 0))
cols[2].metric("Contacted orders that are at risk", pct(precision, 0))
cols[3].metric("Better than random", f"{precision / y.mean():.1f}x")

step = 50  # plot every 50th point: same curve, lighter page
fig = go.Figure()
fig.add_scatter(x=contacted[::step], y=captured[::step], name="LightGBM", line={"color": RED, "width": 3})
fig.add_scatter(x=[0, 1], y=[0, 1], name="Random contact", line={"color": GREY, "dash": "dash"})
fig.add_scatter(x=[contacted[k - 1]], y=[captured[k - 1]], mode="markers", showlegend=False,
                marker={"color": DARK, "size": 12})
fig.update_xaxes(tickformat=".0%", title="Orders contacted (highest risk first)")
fig.update_yaxes(tickformat=".0%", title="Bad reviews reached")
st.plotly_chart(style(fig, height=380), width="stretch")

left, right = st.columns(2, gap="large")
with left:
    st.subheader("3. What drives the risk?")
    top = shap_imp.head(12).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["mean_abs_shap"], y=[FEATURE_NAMES.get(f, f) for f in top["feature"]],
        orientation="h", marker_color=RED,
    ))
    fig.update_xaxes(title="Mean |SHAP value| (impact on the prediction)")
    st.plotly_chart(style(fig, height=430), width="stretch")
    st.caption("SHAP measures how much each feature moves the model's prediction, on average.")

with right:
    st.subheader("4. What-if simulator")
    st.caption("Start from a typical order and change the key drivers.")
    c1, c2 = st.columns(2)
    days = c1.slider("Days to deliver", 1, 60, 10)
    delay = c2.slider("Days late (negative = early)", -30, 30, -10)
    items = c1.slider("Items in the order", 1, 6, 1)
    sellers = c2.slider("Sellers in the order", 1, 3, 1)
    seller_rate = st.slider("Seller's past bad-review rate", 0.0, 0.5, 0.10, step=0.01, format="%.2f")

    x = template.copy()
    x["days_to_deliver"] = days
    x["delay_days"] = delay
    x["is_late"] = float(delay > 0)
    x["days_promised"] = max(days - delay, 1)
    x["carrier_days"] = max(days - x["seller_days"].iloc[0], 0)
    x["n_items"] = items
    x["n_sellers"] = min(sellers, items)
    x["seller_prior_bad_rate"] = seller_rate
    order_risk = model.predict_proba(x[FEATURES])[0, 1]

    base = risk.mean()
    st.metric("Predicted risk of a 1–2★ review", pct(order_risk, 0),
              delta=f"{order_risk / base:.1f}x the average order", delta_color="inverse")
    st.progress(min(order_risk, 1.0))
    if sellers > items:
        st.caption("Sellers capped at the number of items.")
