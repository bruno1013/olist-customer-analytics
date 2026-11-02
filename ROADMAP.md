# Roadmap

Project phases follow CRISP-DM (Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment).

**Core story:** Olist has a retention problem (~3% of customers buy again) and late deliveries hurt satisfaction. We find the most valuable customers, predict bad reviews before they happen, and quantify the business impact.

## Phase 0 — Setup
- [x] Load raw CSVs into SQLite (`src/load_data.py`)
- [x] Project structure, `requirements.txt`, `.gitignore`
- [ ] README skeleton

## Phase 1 — SQL & Data Understanding
- [x] Business questions with JOINs, CTEs and window functions (`sql/01_*.sql` …)
- [x] Key metrics: orders, revenue, AOV, repeat rate, delivery time, review score

## Phase 2 — Logistics & Satisfaction (EDA)
- [x] Delivery delay vs review score (+ chi-square, relative risk, logistic regression controlling for state)
- [x] Delivery performance by state (scatter; interactive map goes in the dashboard)
- [x] Monthly trends (complete months only)
- [x] Seller vs carrier delay decomposition; reviews written before delivery

## Phase 3 — Customers
- [ ] RFM scores per `customer_unique_id`
- [ ] K-Means segments + business names
- [ ] Cohort retention analysis

## Phase 4 — Bad-review prediction (ML)
- [ ] Define target (score ≤ 2) and prediction moment (right after delivery)
- [ ] Feature table without leakage
- [ ] Baseline (logistic regression) → decision tree → LightGBM
- [ ] Evaluation for imbalanced classes (PR-AUC, recall, F1)
- [ ] SHAP explanations

## Phase 5 — Review text NLP (Portuguese)
- [ ] Sentiment with a Portuguese model
- [ ] Complaint topics

## Phase 6 — Dashboard
- [ ] Multipage Streamlit app reading small aggregated parquet files
- [ ] Deploy to Streamlit Community Cloud

## Phase 7 — Storytelling
- [ ] README: executive summary, method, results, recommendations with R$ impact
- [ ] 3-minute demo video, LinkedIn post, CV bullet
