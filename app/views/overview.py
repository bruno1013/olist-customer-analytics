import plotly.express as px
import streamlit as st
from utils import BLUE, brl, load, pct, style

kpis = load("kpis").iloc[0]
monthly = load("monthly")

st.title("Olist Customer Analytics")
st.markdown(
    "Why do customers of a Brazilian marketplace leave bad reviews, who are the valuable customers, "
    "and **what should the business fix first?** SQL, statistics, machine learning and NLP on ~100k orders."
)

cols = st.columns(6)
cols[0].metric("Delivered orders", f"{kpis['orders']:,.0f}")
cols[1].metric("Revenue", brl(kpis["revenue"]))
cols[2].metric("Avg order value", brl(kpis["aov"]))
cols[3].metric("Avg review", f"{kpis['avg_review']:.2f} ★")
cols[4].metric("Late deliveries", pct(kpis["late_rate"]))
cols[5].metric("Repeat customers", pct(kpis["repeat_rate"]))

st.divider()

left, right = st.columns([3, 2], gap="large")
with left:
    options = {
        "Orders": ("orders", ",.0f"),
        "Revenue (R$)": ("revenue", ",.0f"),
        "Late deliveries": ("late_rate", ".0%"),
        "Bad reviews (1–2★)": ("bad_review_rate", ".0%"),
    }
    choice = st.radio("Monthly trend", list(options), horizontal=True)
    column, fmt = options[choice]
    fig = px.line(monthly, x="month", y=column, markers=True, color_discrete_sequence=[BLUE])
    fig.update_yaxes(tickformat=fmt, title=None, rangemode="tozero")
    fig.update_xaxes(title=None)
    st.plotly_chart(style(fig, height=360), width="stretch")
    st.caption("Complete months only (Jan 2017 – Aug 2018). Nov 2017 peak = Black Friday.")

with right:
    st.subheader("The story in four findings")
    st.markdown(
        f"""
1. **Late deliveries drive bad reviews.** A late order is **6.7x** more likely to get 1–2 stars
   (62% vs 9%), in every state. Most of it comes from customers who get the survey while still
   waiting: 70% of them answer before the parcel arrives.
2. **The carrier is the bottleneck.** Late orders spend a median of **26 days** with the carrier
   vs 7 for on-time orders. The Northeast is hit hardest.
3. **Half of complaints are about delivery, not the product** — "not received" is the #1 theme (half of it written before delivery),
   and multi-seller orders suffer from **split shipments** (47% bad reviews even when on time).
4. **Almost nobody comes back.** Only **{pct(kpis['repeat_rate'])}** of customers buy twice, yet
   "at-risk big spenders" hold 40% of revenue — a clear win-back target.
"""
    )
    st.page_link("views/recommendations.py", label="See the recommendations", icon="📋")
