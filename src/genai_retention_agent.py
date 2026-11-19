# src/genai_retention_agent.py
import os
from pydantic import BaseModel, Field


class RetentionOffer(BaseModel):
    action_type: str = Field(description="Short retention action badge")
    discount_pct: int = Field(description="Recommended coupon percentage between 0 and 30")
    personalized_email: str = Field(description="2-3 sentence empathetic recovery email")


def generate_retention_strategy(
    customer_id: str,
    churn_prob: float,
    complaint_category: str,
    review_text: str,
    total_spend: float,
) -> dict:
    # 1. Gatekeeper: Skip AI offer if churn risk is low
    if churn_prob < 0.50:
        return {
            "action_type": "No Action Needed (Low Risk)",
            "discount_pct": 0,
            "personalized_email": "Customer is healthy. Include in standard monthly newsletter.",
        }

    api_key = os.getenv("GEMINI_API_KEY")

    # 2. Live LLM Mode (if GEMINI_API_KEY is available)
    if api_key:
        try:
            from google import genai

            client = genai.Client(api_key=api_key)
            prompt = (
                f"Customer ID: {customer_id} | Lifetime Spend: ${total_spend:.2f}\n"
                f"Predicted Churn Probability: {churn_prob*100:.1f}%\n"
                f"PyTorch Detected Root Cause: {complaint_category}\n"
                f"Customer Review: '{review_text}'\n"
                "Generate a targeted e-commerce retention action, discount percentage (10-25%), "
                "and a short empathetic apology/retention email in English."
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": RetentionOffer,
                    "temperature": 0.2,
                },
            )
            return response.parsed.model_dump()
        except Exception as e:
            print(f"LLM API warning ({e}), using deterministic fallback...")

    # 3. Reliable Deterministic Fallback (works offline & without API keys)
    fallback_templates = {
        "Late_Delivery": (
            "Free Express Shipping + 15% Coupon",
            15,
            f"We sincerely apologize for the shipping delay on your recent order. "
            f"As a valued customer (${total_spend:.2f} spend), enjoy 15% off and Priority Express Shipping on your next order with code FAST15.",
        ),
        "Damaged_Or_Defective": (
            "Instant Replacement + 25% VIP Voucher",
            25,
            f"We are sorry your item arrived in less than perfect condition. "
            f"We have unlocked a 25% VIP replacement voucher (code CARE25) and priority quality inspection on your next purchase.",
        ),
        "Poor_Quality_Or_Service": (
            "20% Satisfaction Recovery Credit",
            20,
            f"Thank you for your honest feedback—we want to make things right. "
            f"Please accept a 20% recovery credit (code RECOVER20) valid across our top-rated catalog.",
        ),
        "Satisfied_No_Issue": (
            "10% Win-Back Loyalty Reward",
            10,
            f"We noticed it has been a while since your last visit! "
            f"Here is an exclusive 10% loyalty reward (code WELCOME10) for your next order.",
        ),
    }

    action, disc, email = fallback_templates.get(
        complaint_category, fallback_templates["Satisfied_No_Issue"]
    )
    return {
        "action_type": action,
        "discount_pct": disc,
        "personalized_email": email,
    }


if __name__ == "__main__":
    # Quick sanity test
    from src.nlp_pytorch_classifier import predict_review_complaint

    sample_review = "O produto chegou com muito atraso e demorou demais!"
    nlp_res = predict_review_complaint(sample_review)
    print("PyTorch Output:", nlp_res)

    offer = generate_retention_strategy(
        customer_id="CUST-9981",
        churn_prob=0.84,
        complaint_category=nlp_res["complaint_category"],
        review_text=sample_review,
        total_spend=340.50,
    )
    print("GenAI Retention Output:", offer)