"""
Credit Risk Prediction API
Uses the real LightGBM model + SHAP explainer from the Taiwan dataset.
"""

import os
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd

# ---------- Config ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

ALL_FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

# ---------- Load models once at startup ----------
print("Loading models...")
model = joblib.load(os.path.join(MODEL_DIR, "lgbm_model.joblib"))
explainer = joblib.load(os.path.join(MODEL_DIR, "shap_explainer.joblib"))
X_test = joblib.load(os.path.join(MODEL_DIR, "X_test.joblib"))
defaults = X_test.median().to_dict()
print("Models loaded successfully.")

# ---------- FastAPI app ----------
app = FastAPI(
    title="Credit Risk Predictor API",
    description="Real LightGBM + SHAP inference for Taiwan credit default dataset",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Request / Response schemas ----------
class ClientFeatures(BaseModel):
    LIMIT_BAL: float = Field(140000, description="Credit limit")
    SEX: int = Field(2, description="1=Male, 2=Female")
    EDUCATION: int = Field(2, description="1=Graduate, 2=University, 3=High School, 4=Others")
    MARRIAGE: int = Field(2, description="1=Married, 2=Single, 3=Others")
    AGE: float = Field(35, description="Age in years")
    PAY_0: float = Field(0, description="Most recent repayment status")
    PAY_2: float = Field(0)
    PAY_3: float = Field(0)
    PAY_4: float = Field(0)
    PAY_5: float = Field(0)
    PAY_6: float = Field(0)
    BILL_AMT1: float = Field(45000)
    BILL_AMT2: float = Field(0)
    BILL_AMT3: float = Field(0)
    BILL_AMT4: float = Field(0)
    BILL_AMT5: float = Field(0)
    BILL_AMT6: float = Field(0)
    PAY_AMT1: float = Field(2000)
    PAY_AMT2: float = Field(0)
    PAY_AMT3: float = Field(0)
    PAY_AMT4: float = Field(0)
    PAY_AMT5: float = Field(0)
    PAY_AMT6: float = Field(0)


class ShapContribution(BaseModel):
    feature: str
    value: float
    shap: float
    direction: str   # "Increases Risk" or "Decreases Risk"


class PredictionResponse(BaseModel):
    pd: float
    risk_label: str
    base_value: float
    top_contributions: List[ShapContribution]
    all_contributions: List[ShapContribution]
    message: str = "ok"


# ---------- Helpers ----------
def get_risk_label(pd_score: float) -> str:
    if pd_score >= 0.40:
        return "HIGH RISK"
    elif pd_score >= 0.25:
        return "MEDIUM-HIGH RISK"
    elif pd_score >= 0.15:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"


def predict_and_explain(features: dict) -> dict:
    # Build single-row DataFrame in correct order
    sample = pd.DataFrame([features])[ALL_FEATURES]

    # Probability of default
    pd_score = float(model.predict_proba(sample)[0, 1])

    # SHAP values
    shap_vals = explainer.shap_values(sample)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]          # positive class
    shap_array = np.asarray(shap_vals[0]).flatten()

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1] if len(base_value) > 1 else base_value[0]
    base_value = float(base_value)

    contributions = []
    for feat, val, shap in zip(ALL_FEATURES, sample.iloc[0].values, shap_array):
        contributions.append({
            "feature": feat,
            "value": float(val),
            "shap": float(shap),
            "direction": "Increases Risk" if shap > 0 else "Decreases Risk"
        })

    # Sort by absolute SHAP
    contributions_sorted = sorted(contributions, key=lambda x: abs(x["shap"]), reverse=True)

    return {
        "pd": pd_score,
        "risk_label": get_risk_label(pd_score),
        "base_value": base_value,
        "top_contributions": contributions_sorted[:12],
        "all_contributions": contributions_sorted
    }


# ---------- Endpoints ----------
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Credit Risk Predictor API is running",
        "endpoints": ["/predict", "/defaults", "/health"]
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model": "LightGBM", "features": len(ALL_FEATURES)}


@app.get("/defaults")
def get_defaults():
    """Return median values from the test set (useful for the mobile form)."""
    return {k: float(v) for k, v in defaults.items()}


@app.post("/predict", response_model=PredictionResponse)
def predict(client: ClientFeatures):
    try:
        features = client.dict()
        result = predict_and_explain(features)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
