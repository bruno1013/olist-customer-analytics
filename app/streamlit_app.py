"""
Olist Customer Analytics — Streamlit dashboard (entry point).

Run locally from the project root:
    .venv/bin/streamlit run app/streamlit_app.py
"""

import streamlit as st
from utils import REPO_URL

st.set_page_config(page_title="Olist Customer Analytics", page_icon="📦", layout="wide")

pages = [
    st.Page("views/overview.py", title="Overview", icon="🏠", default=True),
    st.Page("views/delivery.py", title="Delivery & satisfaction", icon="🚚"),
    st.Page("views/customers.py", title="Customers", icon="👥"),
    st.Page("views/risk_model.py", title="Bad-review risk model", icon="🤖"),
    st.Page("views/reviews.py", title="What customers say", icon="💬"),
    st.Page("views/recommendations.py", title="Recommendations", icon="📋"),
]
navigation = st.navigation(pages)

with st.sidebar:
    st.caption(
        "End-to-end data science project on ~100k orders from Olist, "
        "a Brazilian marketplace (2016–2018)."
    )
    st.markdown(f"[Code and notebooks on GitHub]({REPO_URL})")
    st.caption("Data: Olist Brazilian E-commerce Public Dataset (Kaggle, CC BY-NC-SA 4.0).")

navigation.run()
