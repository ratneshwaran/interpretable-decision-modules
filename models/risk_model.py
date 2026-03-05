"""
ML Risk Model — gradient boosting classifier for procurement risk.

Outputs a scalar risk probability in [0, 1] and top feature contributions
for use in the explanation bundle.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

FEATURES = [
    "amount",
    "vendor_risk",
    "document_confidence",
    "anomaly_score",
    "historical_risk",
]
LABEL_ORDER = ["approve", "review", "reject"]   # low → high risk

MODEL_PATH = Path(__file__).parent / "risk_model.pkl"


class RiskModel:
    """Thin wrapper around a GradientBoostingClassifier."""

    def __init__(self) -> None:
        self.clf = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        self.le = LabelEncoder()
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "RiskModel":
        X = df[FEATURES].copy()
        # Log-transform amount to reduce skew
        X["amount"] = np.log1p(X["amount"])

        y_raw = df["label"].values
        # Encode with explicit ordering so class indices are stable
        self.le.fit(LABEL_ORDER)
        y = self.le.transform(y_raw)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.clf.fit(X_train, y_train)

        y_pred = self.clf.predict(X_val)
        print("=== Validation report ===")
        print(classification_report(y_val, y_pred, target_names=self.le.classes_))

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_risk(self, record: dict[str, Any]) -> dict[str, Any]:
        """
        Returns
        -------
        dict with keys:
          ml_risk       : float in [0, 1] — weighted risk score
          ml_label      : str  — most likely label
          probabilities : dict {label: prob}
          feature_scores: dict {feature: importance * value_deviation}
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")

        x = pd.DataFrame([record])[FEATURES].copy()
        x["amount"] = np.log1p(x["amount"])

        proba = self.clf.predict_proba(x)[0]
        prob_dict = {
            label: float(p) for label, p in zip(self.le.classes_, proba)
        }

        # Weighted risk: approve=0, review=0.5, reject=1
        weights = {"approve": 0.0, "review": 0.5, "reject": 1.0}
        ml_risk = sum(weights[lbl] * p for lbl, p in prob_dict.items())

        ml_label = self.le.inverse_transform([self.clf.predict(x)[0]])[0]

        # Feature importance as proxy for contribution
        fi = self.clf.feature_importances_
        feature_scores = {
            feat: round(float(imp), 4)
            for feat, imp in zip(FEATURES, fi)
        }

        return {
            "ml_risk": round(float(ml_risk), 4),
            "ml_label": ml_label,
            "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
            "feature_scores": feature_scores,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path = MODEL_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "RiskModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise TypeError(f"Expected RiskModel, got {type(model)}")
        return model


def train_and_save() -> RiskModel:
    from data.generate import generate_dataset, save_dataset

    df = generate_dataset()
    save_dataset(df)

    model = RiskModel()
    model.fit(df)
    model.save()
    return model


if __name__ == "__main__":
    train_and_save()
