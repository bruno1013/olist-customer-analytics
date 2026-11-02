import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils import BLUE, GREY, RED, load, load_model, pct, style

from text import normalize, tag_themes

themes = load("themes")
examples = load("theme_examples")
weights = load("text_weights")
model = load_model("review_text_model")

st.title("💬 What customers say")
st.markdown(
    "38,872 Portuguese review comments, analysed with a text classifier (TF-IDF + logistic regression, "
    "test PR-AUC **0.80**) and complaint themes validated with NMF topic modelling."
)

st.subheader("Try the text model")
SAMPLES = {
    "Write your own": "",
    "Delivered fast (positive)": "Chegou rápido e bem embalado, recomendo!",
    "Not received": "Não recebi o produto até agora, já passou do prazo.",
    "Split shipment": "Comprei dois itens e só veio um, o outro não chegou.",
    "Damaged": "Produto veio quebrado, quero a devolução do dinheiro.",
    "Mixed": "Produto bom, só não dou 5 estrelas porque demorou um pouco.",
}
choice = st.selectbox("Pick an example or write your own (Portuguese)", list(SAMPLES))
text = st.text_area("Review comment", value=SAMPLES[choice] or "O produto chegou antes do prazo, adorei!",
                    height=90)

if text.strip():
    clean = normalize(text)
    risk = model.predict_proba([clean])[0, 1]
    found = tag_themes(pd.Series([clean])).iloc[0]
    found = [theme for theme, hit in found.items() if hit]

    c1, c2 = st.columns([1, 2])
    c1.metric("Probability of a 1–2★ review", pct(risk, 0))
    c1.progress(float(risk))
    c2.markdown("**Complaint themes detected**")
    c2.markdown(" ".join(f"`{t}`" for t in found) if found else "_None_")
    c2.caption(f"Normalised text: {clean}")

st.divider()

left, right = st.columns(2, gap="large")
with left:
    all_bad = themes[themes["group"] == "All bad reviews"].sort_values("share")
    fig = go.Figure(go.Bar(
        x=all_bad["share"], y=all_bad["theme"], orientation="h",
        marker_color=[RED if v == all_bad["share"].max() else GREY for v in all_bad["share"]],
        text=[pct(v, 0) for v in all_bad["share"]], textposition="outside",
    ))
    fig.update_xaxes(tickformat=".0%", range=[0, 0.4])
    st.plotly_chart(style(fig, height=380, title="Complaint themes in bad reviews"),
                    width="stretch")
    st.caption("50.5% of bad-review comments are about delivery; 20.8% about the product. "
               "A comment can mention several themes.")

with right:
    by_group = themes[themes["group"] != "All bad reviews"].pivot(index="theme", columns="group", values="share")
    by_group = by_group.sort_values("2+ sellers")
    fig = go.Figure()
    fig.add_bar(y=by_group.index, x=by_group["1 seller"], name="1 seller", orientation="h", marker_color=GREY)
    fig.add_bar(y=by_group.index, x=by_group["2+ sellers"], name="2+ sellers", orientation="h", marker_color=RED)
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(style(fig, height=380, title="Multi-seller orders: missing items (split shipments)"),
                    width="stretch")

left, right = st.columns(2, gap="large")
with left:
    theme = st.selectbox("Read real bad-review comments by theme", examples["theme"].unique())
    for comment in examples.loc[examples["theme"] == theme, "comment"]:
        st.markdown(f"> {comment}")
    st.caption("Original Portuguese comments from the public dataset.")

with right:
    w = weights.sort_values("weight")
    fig = go.Figure(go.Bar(
        x=w["weight"], y=w["term"], orientation="h",
        marker_color=[BLUE if v < 0 else RED for v in w["weight"]],
    ))
    fig.update_xaxes(title="← good review | bad review →")
    st.plotly_chart(style(fig, height=700, title="Words the model relies on"), width="stretch")
