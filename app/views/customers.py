import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils import BLUE, GREEN, GREY, RED, SAND, brl, load, pct, style

kpis = load("kpis").iloc[0]
segments = load("segments")
clusters = load("clusters_sample")
cohort = load("cohort")

st.title("👥 Customers")
st.markdown("Who are the valuable customers, and does anybody come back?")

at_risk = segments.set_index("segment").loc["At-risk big spender"]
cols = st.columns(4)
cols[0].metric("Customers", f"{kpis['customers']:,.0f}")
cols[1].metric("Repeat customers", pct(kpis["repeat_rate"]))
cols[2].metric("At-risk big spenders", f"{at_risk['customers']:,.0f}")
cols[3].metric("Their share of revenue", pct(at_risk["revenue_share"], 0))

left, right = st.columns([3, 2], gap="large")
with left:
    fig = go.Figure()
    fig.add_bar(y=segments["segment"], x=segments["customer_share"], name="Share of customers",
                orientation="h", marker_color=GREY)
    fig.add_bar(y=segments["segment"], x=segments["revenue_share"], name="Share of revenue",
                orientation="h", marker_color=BLUE)
    fig.update_layout(barmode="group")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(style(fig, height=420, title="RFM segments: customers vs revenue"),
                    width="stretch")

with right:
    st.markdown("**How the segments are defined**")
    st.markdown(
        """
- **Recency (R)** and **Monetary (M)** are scored 1–5 by quintile.
- **Frequency** cannot be split into quintiles (97% of customers have one order),
  so it becomes a *repeat customer* flag.
- *Loyal* = repeat & recent · *Lapsed loyal* = repeat & not recent ·
  *Big spender* = M ≥ 4 · *Recent* = R ≥ 4 · *At-risk* / *Hibernating* = R ≤ 3.
"""
    )
    st.success(
        f"**Win-back target:** {at_risk['customers']:,.0f} at-risk big spenders spent "
        f"{brl(at_risk['revenue'])} ({pct(at_risk['revenue_share'], 0)} of revenue) but have not bought "
        f"for ~{at_risk['avg_recency_days'] / 30:.0f} months on average."
    )

st.dataframe(
    segments[["segment", "customers", "customer_share", "revenue_share", "avg_spend", "avg_recency_days", "avg_review"]],
    hide_index=True, width="stretch",
    column_config={
        "segment": "Segment",
        "customers": st.column_config.NumberColumn("Customers", format="%d"),
        "customer_share": st.column_config.NumberColumn("% customers", format="percent"),
        "revenue_share": st.column_config.NumberColumn("% revenue", format="percent"),
        "avg_spend": st.column_config.NumberColumn("Avg spend (R$)", format="%.0f"),
        "avg_recency_days": st.column_config.NumberColumn("Days since last order", format="%.0f"),
        "avg_review": st.column_config.NumberColumn("Avg review", format="%.2f ★"),
    },
)

left, right = st.columns(2, gap="large")
with left:
    fig = px.scatter(
        clusters, x="recency_days", y="monetary", color="cluster_name", opacity=0.55, log_y=True,
        color_discrete_sequence=[BLUE, RED, GREEN, SAND],
        labels={"recency_days": "Days since last purchase", "monetary": "Total spent (R$, log)",
                "cluster_name": "Cluster"},
    )
    fig.update_traces(marker={"size": 5})
    st.plotly_chart(style(fig, height=420, title="K-Means finds 4 groups (5k-customer sample)"),
                    width="stretch")
    st.caption("K-Means on recency, log(spend) and the repeat flag. It agrees closely with the simpler RFM rules.")

with right:
    heat = cohort.pivot(index="cohort", columns="months_after", values="retention")
    fig = px.imshow(heat, color_continuous_scale="Blues", zmin=0, zmax=0.01, aspect="auto",
                    labels={"x": "Months after first purchase", "y": "First-purchase cohort", "color": "Returned"})
    fig.update_traces(hovertemplate="Cohort %{y}, month %{x}: %{z:.2%}<extra></extra>")
    fig.update_coloraxes(colorbar={"tickformat": ".1%"})
    st.plotly_chart(style(fig, height=420, title="Cohort retention: < 1% return in any month"),
                    width="stretch")

st.info(
    "**Takeaways.** Revenue is concentrated (top 20% of customers = 54% of revenue) and retention is "
    "extremely low (1.95% return within 180 days). A bad first experience lowers returns only slightly "
    "and not significantly (p = 0.135), so retention needs active levers — win-back campaigns, "
    "second-purchase vouchers, cross-sell — not just better delivery."
)
