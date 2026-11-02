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
- [x] RFM scores per `customer_unique_id` (Frequency as repeat flag: quintiles fail)
- [x] Rule-based segments + K-Means (k = 4) with business names
- [x] Cohort retention + does a bad first experience reduce returns? (not significant)

## Phase 4 — Bad-review prediction (ML)
- [x] Define target (score ≤ 2) and prediction moment (right after delivery)
- [x] Feature table without leakage (`src/features.py`, time-aware seller history)
- [x] Dummy → logistic regression → decision tree → LightGBM (time-based split)
- [x] Evaluation for imbalanced classes (PR-AUC, top-10% lift, threshold on validation)
- [x] SHAP explanations → new finding: multi-seller orders get 47% bad reviews

## Phase 5 — Review text NLP (Portuguese)
- [x] Text classifier trained on review scores (TF-IDF + logistic regression, PR-AUC 0.80)
- [x] Complaint themes (regex) validated with NMF topics; split-shipment hypothesis confirmed

## Phase 6 — Dashboard
- [ ] Multipage Streamlit app reading small aggregated parquet files
- [ ] Deploy to Streamlit Community Cloud

## Phase 7 — Storytelling
- [ ] README: executive summary, method, results, recommendations with R$ impact
- [ ] 3-minute demo video, LinkedIn post, CV bullet
