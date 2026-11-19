# app.py
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from src.feature_pipeline import NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.nlp_pytorch_classifier import predict_review_complaint, assign_root_cause_label, LABELS
from src.genai_retention_agent import generate_retention_strategy

app = FastAPI(
    title="E-Commerce Churn, PyTorch NLP & GenAI Retention API",
    version="1.0.0",
    description="Full-stack MLOps endpoint combining Tabular ML, PyTorch Deep Learning, and LLM Retention.",
)

# Use os.path.join so file paths work on both Windows and Linux Docker containers
PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")
MODEL_PATH = os.path.join("artifacts", "churn_model.pkl")

preprocessor = joblib.load(PREPROCESSOR_PATH)
churn_model = joblib.load(MODEL_PATH)


class CustomerInput(BaseModel):
    customer_unique_id: str = Field(default="CUST-1001")
    customer_state: str = Field(default="SP")
    frequency_orders: int = Field(default=1, ge=1)
    total_monetary_spend: float = Field(default=240.50, ge=0)
    avg_delivery_delay_days: float = Field(default=6.0)
    max_installments: int = Field(default=3, ge=1)
    primary_payment_type: str = Field(default="credit_card")
    avg_review_score: float = Field(default=2.0, ge=1.0, le=5.0)
    latest_review_text: str = Field(default="O produto chegou com muito atraso e demorou demais!")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "sklearn_preprocessor_loaded": preprocessor is not None,
        "tabular_ml_model_loaded": churn_model is not None,
        "pytorch_weights_present": os.path.exists(os.path.join("artifacts", "review_classifier.pt")),
    }


@app.post("/predict_customer")
def predict_single_customer(customer: CustomerInput):
    try:
        data = customer.model_dump()
        # Compute engineered features on the fly
        data["avg_order_value"] = round(data["total_monetary_spend"] / max(data["frequency_orders"], 1), 2)
        data["is_late_delivery"] = 1 if data["avg_delivery_delay_days"] > 0 else 0
        review_clean = str(data["latest_review_text"]).strip()
        data["has_written_review"] = 0 if (review_clean == "" or review_clean.lower() == "no written review") else 1

        df_in = pd.DataFrame([data])
        X_trans = preprocessor.transform(df_in[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])

        # 1. Tabular ML Churn Probability
        churn_prob = float(churn_model.predict_proba(X_trans)[0][1])
        risk_tier = "High Risk" if churn_prob >= 0.70 else ("Medium Risk" if churn_prob >= 0.40 else "Low Risk")

        # 2. PyTorch Deep Learning Root-Cause Classification
        nlp_result = predict_review_complaint(data["latest_review_text"])

        # 3. GenAI Personalized Retention Offer
        retention_plan = generate_retention_strategy(
            customer_id=data["customer_unique_id"],
            churn_prob=churn_prob,
            complaint_category=nlp_result["complaint_category"],
            review_text=data["latest_review_text"],
            total_spend=data["total_monetary_spend"],
        )

        return {
            "customer_unique_id": data["customer_unique_id"],
            "churn_probability": round(churn_prob, 4),
            "risk_tier": risk_tier,
            "pytorch_complaint_category": nlp_result["complaint_category"],
            "pytorch_confidence": nlp_result["confidence"],
            "recommended_action": retention_plan["action_type"],
            "recommended_discount_pct": retention_plan["discount_pct"],
            "ai_recovery_email": retention_plan["personalized_email"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_score_to_sql")
def batch_score_customers_to_postgres(limit: int = 5000):
    """Scores customers in bulk and writes results to PostgreSQL for Stage 6 Power BI."""
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:madhu123@localhost:5432/olist_ecommerce",
    )
    test_csv_path = os.path.join("artifacts", "test.csv")
    if not os.path.exists(test_csv_path):
        raise HTTPException(status_code=404, detail=f"Missing {test_csv_path}")

    df = pd.read_csv(test_csv_path).head(limit).copy()

    # 1. Bulk Tabular ML Prediction
    X_trans = preprocessor.transform(df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])
    df["churn_probability"] = churn_model.predict_proba(X_trans)[:, 1].round(4)
    df["risk_tier"] = df["churn_probability"].apply(
        lambda p: "High Risk" if p >= 0.70 else ("Medium Risk" if p >= 0.40 else "Low Risk")
    )

    # 2. Root-Cause Assignment & Retention Action Mapping for Power BI
    df["complaint_category"] = df.apply(lambda r: LABELS[assign_root_cause_label(r)], axis=1)
    df["revenue_at_risk"] = (df["total_monetary_spend"] * df["churn_probability"]).round(2)

    # Temporarily disable live external LLM calls during 5,000-row batch scoring to prevent API timeouts
    saved_key = os.environ.pop("GEMINI_API_KEY", None)
    try:
        actions = df.apply(
            lambda r: generate_retention_strategy(
                customer_id=str(r["customer_unique_id"]),
                churn_prob=float(r["churn_probability"]),
                complaint_category=str(r["complaint_category"]),
                review_text=str(r["latest_review_text"]),
                total_spend=float(r["total_monetary_spend"]),
            ),
            axis=1,
        )
    finally:
        if saved_key:
            os.environ["GEMINI_API_KEY"] = saved_key

    df["recommended_action"] = [a["action_type"] for a in actions]
    df["recommended_discount_pct"] = [a["discount_pct"] for a in actions]
    df["ai_recovery_email"] = [a["personalized_email"] for a in actions]

    # Save backup CSV and push to PostgreSQL table for Power BI
    out_path = os.path.join("artifacts", "scored_predictions_for_powerbi.csv")
    df.to_csv(out_path, index=False)

    try:
        engine = create_engine(db_url)
        df.to_sql("customer_churn_predictions", engine, if_exists="replace", index=False)
        sql_status = "Written to PostgreSQL table 'customer_churn_predictions'"
    except Exception as e:
        sql_status = f"Saved to CSV ({out_path}); PostgreSQL skipped: {e}"

    return {
        "rows_scored": len(df),
        "high_risk_customers_found": int((df["risk_tier"] == "High Risk").sum()),
        "total_revenue_at_risk": round(float(df["revenue_at_risk"].sum()), 2),
        "storage_status": sql_status,
    }