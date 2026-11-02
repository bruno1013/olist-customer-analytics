import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils import GREY, RED, SAND, load, pct, style

kpis = load("kpis").iloc[0]
buckets = load("delay_buckets")
states = load("states")
stages = load("stages")
basket = load("basket")
impact = load("impact").set_index("problem")

st.title("🚚 Delivery & satisfaction")
st.markdown("Does a late delivery really cost a good review — and where do the delays come from?")

late = impact.loc["Late deliveries"]
cols = st.columns(4)
cols[0].metric("Avg days to deliver", f"{kpis['avg_days_to_deliver']:.1f}")
cols[1].metric("Avg days promised", f"{kpis['avg_days_promised']:.1f}")
cols[2].metric("Bad reviews when on time", pct(late["bad_rate_rest"]))
cols[3].metric("Bad reviews when late", pct(late["bad_rate_group"]),
               delta=f"{late['bad_rate_group'] / late['bad_rate_rest']:.1f}x", delta_color="inverse")

left, right = st.columns(2, gap="large")
with left:
    colors = [GREY] * 3 + [RED] * 3
    fig = go.Figure(go.Bar(
        x=buckets["bucket"], y=buckets["bad_review_rate"], marker_color=colors,
        text=[pct(v, 0) for v in buckets["bad_review_rate"]], textposition="outside",
        customdata=buckets["orders"],
        hovertemplate="%{x}<br>Bad reviews: %{y:.1%}<br>Orders: %{customdata:,}<extra></extra>",
    ))
    fig.update_yaxes(tickformat=".0%", range=[0, 0.95], title="Share of 1–2★ reviews")
    st.plotly_chart(style(fig, title="Late deliveries turn customers into detractors"),
                    width="stretch")

with right:
    fig = go.Figure()
    fig.add_bar(y=stages["delivery"], x=stages["seller_days"], name="Seller stage",
                orientation="h", marker_color=GREY, text=stages["seller_days"].round(1),
                texttemplate="%{text}d")
    fig.add_bar(y=stages["delivery"], x=stages["carrier_days"], name="Carrier stage",
                orientation="h", marker_color=RED, text=stages["carrier_days"].round(1),
                texttemplate="%{text}d")
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Median days")
    st.plotly_chart(style(fig, title="Late orders lose their time with the carrier"),
                    width="stretch")

st.subheader("Where are the problems? Delivery performance by state")
regions = st.multiselect("Regions", sorted(states["region"].unique()), default=sorted(states["region"].unique()))
shown = states[states["region"].isin(regions)]

map_col, table_col = st.columns([3, 2], gap="large")
with map_col:
    fig = px.scatter_geo(
        shown, lat="lat", lon="lng", size="orders", color="late_rate",
        color_continuous_scale=["#F4D6DA", RED, "#7A1F2B"], size_max=45,
        hover_name="state",
        hover_data={"orders": ":,", "late_rate": ":.1%", "avg_review": ":.2f",
                    "avg_days_to_deliver": ":.1f", "lat": False, "lng": False},
        labels={"late_rate": "Late", "avg_review": "Avg review", "avg_days_to_deliver": "Days to deliver"},
    )
    fig.update_geos(fitbounds="locations", showcountries=True, countrycolor="#CCCCCC",
                    showland=True, landcolor="#F7F7F7", showocean=False, lakecolor="white")
    fig.update_coloraxes(colorbar={"tickformat": ".0%", "title": "Late"})
    st.plotly_chart(style(fig, height=480), width="stretch")
    st.caption("Bubble size = orders; colour = share of late deliveries.")

with table_col:
    st.dataframe(
        shown.sort_values("late_rate", ascending=False)[
            ["state", "region", "orders", "late_rate", "avg_days_to_deliver", "avg_review"]],
        hide_index=True, width="stretch", height=480,
        column_config={
            "state": "State",
            "region": "Region",
            "orders": st.column_config.NumberColumn("Orders", format="%d"),
            "late_rate": st.column_config.ProgressColumn("Late", format="percent", min_value=0, max_value=0.25),
            "avg_days_to_deliver": st.column_config.NumberColumn("Days to deliver", format="%.1f"),
            "avg_review": st.column_config.NumberColumn("Avg review", format="%.2f ★"),
        },
    )

st.subheader("Split orders: more items and sellers, more bad reviews")
fig = px.bar(basket, x="group", y="bad_review_rate", facet_col="kind", text="bad_review_rate",
             color="group", color_discrete_sequence=[GREY, SAND, RED, GREY, RED],
             category_orders={"group": ["1 item", "2 items", "3+ items", "1 seller", "2+ sellers"]})
fig.update_traces(texttemplate="%{text:.0%}", textposition="outside")
fig.update_yaxes(tickformat=".0%", title=None, range=[0, 0.6])
fig.update_xaxes(matches=None, title=None)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_layout(showlegend=False)
st.plotly_chart(style(fig, height=340), width="stretch")

st.info(
    "**Takeaways.** Promises are padded (12.6 days actual vs 23.7 promised) yet 6.8% of orders are late, "
    "and a late order is ~6.7x more likely to get a bad review. Delays concentrate in the carrier stage "
    "and in the Northeast (AL, MA ≈ 4–5x São Paulo's late rate). Orders split across sellers get 47% "
    "bad reviews even when they arrive on time."
)
