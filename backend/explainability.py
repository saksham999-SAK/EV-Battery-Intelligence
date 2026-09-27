from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import HTTPException


def explain_soh_prediction(soh_model, features: list[str], input_data: dict) -> dict:
    try:
        import shap
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="SHAP is not installed. Install backend requirements to enable explainability.",
        ) from exc

    X = pd.DataFrame([input_data], columns=features)
    prediction = float(soh_model.predict(X)[0])

    background = pd.DataFrame([np.zeros(len(features))], columns=features)

    try:
        explainer = shap.Explainer(soh_model, background)
        shap_values = explainer(X).values[0]
    except Exception:
        try:
            explainer = shap.KernelExplainer(soh_model.predict, background)
            shap_values = explainer.shap_values(X, silent=True)[0]
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not calculate SHAP values for this SOH model: {exc}",
            ) from exc

    rows = [
        {
            "feature": feature,
            "value": float(input_data[feature]),
            "shap_value": round(float(value), 6),
            "impact": "pushes prediction higher" if value >= 0 else "pushes prediction lower",
        }
        for feature, value in zip(features, shap_values)
    ]
    rows.sort(key=lambda row: abs(row["shap_value"]), reverse=True)

    return {
        "predicted_SOH": round(prediction, 2),
        "baseline": "zero feature baseline because the original training dataset is not included in this repository",
        "interpretation": (
            "Positive SHAP values push this model prediction higher; negative SHAP values "
            "push it lower. SHAP values are model contributions, not causal effects."
        ),
        "features": features,
        "feature_values": {feature: float(input_data[feature]) for feature in features},
        "shap_values": {row["feature"]: row["shap_value"] for row in rows},
        "top_features": rows,
    }
