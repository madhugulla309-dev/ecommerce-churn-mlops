# End-to-End E-Commerce Customer Churn, PyTorch NLP & GenAI Retention Engine

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-18-336791?style=for-the-badge&logo=postgresql&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML_Pipeline-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Microservice-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Power BI](https://img.shields.io/badge/Power_BI-Executive_BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)

An end-to-end **Data Science, Deep Learning, MLOps, and Business Intelligence** pipeline built on the **Brazilian E-Commerce Public Dataset (Olist)** (100K+ multi-table records).

Rather than stopping at static churn predictions, this system bridges tabular machine learning, unstructured text deep learning, generative AI retention automation, and executive Power BI financial reporting.

---

## System Architecture

```text
[Raw Kaggle CSVs & Excel Audit] 
        │
        ▼
[PostgreSQL Data Warehouse] ──► SQL CTEs & Window Functions (RFM, Delivery Delay, Churn Label)
        │
        ▼
[Python EDA & Feature Eng.] ──► Pandas, Matplotlib & Seaborn Statistical Insights
        │
        ├──► [Tabular ML Pipeline] ──► Scikit-Learn + XGBoost + MLflow (Churn Probability)
        │
        └──► [Deep Learning NLP]   ──► PyTorch nn.Embedding Net (Complaint Root-Cause)
                    │
                    ▼
        [GenAI Retention Agent]    ──► Gemini API + Pydantic (Personalized Recovery Offer)
                    │
                    ▼
        [Dockerized FastAPI]       ──► Real-Time REST API (/predict_customer) & Batch SQL Sync
                    │
                    ▼
        [Power BI & Excel BI]      ──► 15 Custom DAX Measures (Revenue at Risk & Campaign ROI)
