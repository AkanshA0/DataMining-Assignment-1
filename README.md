# Data Mining Assignment 1
Medium blog: https://medium.com/@akansha.b1000/from-churn-chaos-to-clarity-how-rfm-behavioral-signals-built-a-predictive-retention-engine-a9ba79672e98 <br>
Chatgpt: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset/data <br>
Kaggle Dataset: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset/data <br>

This repository contains artifacts for a full churn‑prediction project with Business & Data Understanding, RFM analysis, Feature Engineering, Unsupervised Segmentation, Modeling, Threshold Tuning, and Deployment guidance.

---

## Overview

**Business goal:** predict customer churn to drive proactive, cost‑aware retention.  
**Approach:** RFM + friction signals → clean pipelines → interpretable baseline (Logistic) → CatBoost preset for optional lift → tuned thresholds for high‑touch vs low‑touch outreach.

**Top drivers (from EDA & modeling):**
- Support Calls ↑ and Payment Delay ↑ → churn risk ↑  
- Recent Interaction ↑, Usage ↑, Spend ↑, Tenure ↑ → churn risk ↓

**Chosen baseline model:** **Logistic Regression (L2, class_weight="balanced")** — strong, calibrated, interpretable.

## Exploratory Data Analysis (EDA)
For detailed EDA charts check [Churn_EDA_Report](https://github.com/AkanshA0/DataMining-Assignment-1/blob/master/Churn_EDA_Report.pdf), below are some of the charts:
<img width="719" height="377" alt="image" src="https://github.com/user-attachments/assets/9efda0f3-e918-483a-a083-bcf74f711946" />
<img width="786" height="386" alt="image" src="https://github.com/user-attachments/assets/6550bf91-53d1-4a16-9b8c-c60020792578" />
<img width="866" height="438" alt="image" src="https://github.com/user-attachments/assets/2d4c4f32-177d-429d-b562-de3051d62f70" />

**RFM Analysis**: 
RFM analyzes customer value by scoring their Recency, Frequency, and Monetary value of purchases. It is a powerful tool for churn prediction because it directly measures customer engagement. A decline in these scores, particularly in how recently and often a customer buys, serves as a clear warning sign that they are becoming disengaged and likely to leave. This enables businesses to proactively identify at-risk customers and target them with retention campaigns before they are lost for good.
Check [Churn_RFM_Detailed_Report](https://github.com/AkanshA0/DataMining-Assignment-1/blob/master/Churn_RFM_Detailed_Report.pdf) for detailed RFM analysis report.

**Data files expected** (not stored in this repo):  
- `customer_churn_dataset-training-master.csv` (training; with labels)  
- `customer_churn_dataset-testing-master.csv` (testing; no labels)

## Reproduce Key Results

### Run CatBoost with curated presets
```bash
python train_catboost_churn.py --data churn_training_clean.csv --outdir ./catboost_out --preset f1_optimized
# alternatives: --preset high_recall | high_precision
```
Outputs: `catboost_cv_metrics.json`, `threshold_sweep.csv`, `pr_curve.png`, `f1_vs_threshold.png`.

### Use the saved Logistic model
```python
import joblib, pandas as pd
pipe = joblib.load("model_logreg_balanced.pkl")
df = pd.read_csv("customer_churn_dataset-testing-master.csv")  # features only
proba = pipe.predict_proba(df)[:, 1]
# choose threshold (e.g., from logreg_balanced_threshold_sweep.csv)
y_hat = (proba >= 0.51).astype(int)
```
