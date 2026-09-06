"""
Standalone Prediction Script for CTU-UHB Continuous Clinical Model (Model B).

Allows easy zero-dependency inference to predict:
1. Estimated Umbilical Artery pH (pH_hat)
2. Acidemia Risk Score (S = -pH_hat)
3. Standard Acidemia Classification (pH <= 7.15)
4. Severe Acidemia Classification (pH <= 7.05)

Usage Example:
    from models.predict import CTUAcidemiaPredictor
    predictor = CTUAcidemiaPredictor()
    result = predictor.predict_patient(feature_dict)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
HUBER_DIR = os.path.join(MODEL_DIR, "continuous_clinical_huber")

FEATURE_NAMES = [
    "baseline", "stv", "ltv", "acc_count", "early_dec_count", "late_dec_count",
    "var_dec_count", "prolonged_dec_count", "dec_max_depth", "dec_area",
    "dec_burden", "longest_dec", "baseline_slope", "variability_slope",
    "uc_count", "tachysystole", "mean_uc_amp", "fhr_uc_lag", "fhr_uc_coupling"
]

class CTUAcidemiaPredictor:
    def __init__(self, mode="ensemble"):
        """
        mode: 'ensemble' (averages all 5 folds) or 'full' (uses model fit on full 547 patient cohort)
        """
        self.mode = mode
        self.features = FEATURE_NAMES
        
        if mode == "full":
            self.model = joblib.load(os.path.join(HUBER_DIR, "huber_full_cohort.joblib"))
            self.scaler = joblib.load(os.path.join(HUBER_DIR, "scaler_full_cohort.joblib"))
        else:
            self.models = [joblib.load(os.path.join(HUBER_DIR, f"huber_fold_{i}.joblib")) for i in range(5)]
            self.scalers = [joblib.load(os.path.join(HUBER_DIR, f"scaler_fold_{i}.joblib")) for i in range(5)]

    def predict(self, X: np.ndarray) -> dict:
        """
        X: numpy array of shape (N, 19) or (19,)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
            single = True
        else:
            single = False

        if self.mode == "full":
            X_sc = self.scaler.transform(X)
            pred_ph = self.model.predict(X_sc)
        else:
            fold_preds = []
            for m, sc in zip(self.models, self.scalers):
                X_sc = sc.transform(X)
                fold_preds.append(m.predict(X_sc))
            pred_ph = np.mean(fold_preds, axis=0)

        risk_score = -pred_ph
        # Calibrated risk proxy probability (sigmoid mapping centered around pH 7.15)
        risk_prob = 1.0 / (1.0 + np.exp(15.0 * (pred_ph - 7.15)))
        flag_acidemia = (pred_ph <= 7.15).astype(int)
        flag_severe = (pred_ph <= 7.05).astype(int)

        if single:
            return {
                "predicted_ph": float(pred_ph[0]),
                "acidemia_risk_score": float(risk_score[0]),
                "acidemia_probability_proxy": float(risk_prob[0]),
                "acidemia_flag_715": bool(flag_acidemia[0]),
                "severe_acidemia_flag_705": bool(flag_severe[0])
            }
        else:
            return {
                "predicted_ph": pred_ph,
                "acidemia_risk_score": risk_score,
                "acidemia_probability_proxy": risk_prob,
                "acidemia_flag_715": flag_acidemia,
                "severe_acidemia_flag_705": flag_severe
            }

    def predict_patient(self, feature_dict: dict) -> dict:
        """
        Predict for a single patient dictionary of 19 feature names.
        """
        x_vec = np.array([float(feature_dict.get(feat, 0.0)) for feat in self.features], dtype=np.float32)
        return self.predict(x_vec)


if __name__ == "__main__":
    predictor = CTUAcidemiaPredictor(mode="ensemble")
    # Sample test with normal values
    sample_features = {
        "baseline": 135.0, "stv": 6.5, "ltv": 14.0, "acc_count": 3.0,
        "early_dec_count": 0.0, "late_dec_count": 0.0, "var_dec_count": 1.0,
        "prolonged_dec_count": 0.0, "dec_max_depth": 15.0, "dec_area": 120.0,
        "dec_burden": 0.05, "longest_dec": 30.0, "baseline_slope": 0.0,
        "variability_slope": 0.0, "uc_count": 4.0, "tachysystole": 0.0,
        "mean_uc_amp": 45.0, "fhr_uc_lag": 10.0, "fhr_uc_coupling": 0.3
    }
    res = predictor.predict_patient(sample_features)
    print("--- Sample Normal Patient Prediction ---")
    print(json.dumps(res, indent=2))
