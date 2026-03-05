"""
Synthetic dataset generator for the interpretable decision pipeline.

Generates procurement-style records with features that correlate realistically
with approval outcomes, including deliberate noise and edge cases.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
DATA_DIR = Path(__file__).parent


def generate_dataset(n_samples: int = 2000, seed: int = SEED) -> pd.DataFrame:
    """
    Generate a synthetic procurement risk dataset.

    Features
    --------
    amount               : transaction value in USD (log-normal)
    vendor_risk          : vendor risk score [0, 1]
    document_confidence  : OCR/document quality score [0, 1]
    anomaly_score        : anomaly detector output [0, 1]
    historical_risk      : rolling historical risk of this vendor [0, 1]

    Label
    -----
    label : {approve, review, reject}
    """
    rng = np.random.default_rng(seed)

    # Raw features
    amount_raw = rng.lognormal(mean=10.5, sigma=1.2, size=n_samples)
    amount = np.clip(amount_raw, 500, 500_000)

    vendor_risk = rng.beta(a=2, b=5, size=n_samples)
    document_confidence = rng.beta(a=5, b=2, size=n_samples)
    anomaly_score = rng.beta(a=1.5, b=6, size=n_samples)
    historical_risk = rng.beta(a=2, b=4, size=n_samples)

    # Normalise amount to [0, 1] for scoring
    amount_norm = (np.log1p(amount) - np.log1p(500)) / (
        np.log1p(500_000) - np.log1p(500)
    )

    # Composite risk signal (higher → riskier)
    risk_signal = (
        0.30 * amount_norm
        + 0.25 * vendor_risk
        + 0.20 * (1 - document_confidence)
        + 0.15 * anomaly_score
        + 0.10 * historical_risk
        + rng.normal(0, 0.05, size=n_samples)  # noise
    )

    # Assign labels based on risk thresholds
    labels = np.where(risk_signal < 0.30, "approve",
              np.where(risk_signal < 0.55, "review", "reject"))

    df = pd.DataFrame({
        "amount": amount.round(2),
        "vendor_risk": vendor_risk.round(4),
        "document_confidence": document_confidence.round(4),
        "anomaly_score": anomaly_score.round(4),
        "historical_risk": historical_risk.round(4),
        "label": labels,
    })

    return df


def save_dataset(df: pd.DataFrame, filename: str = "procurement_risk.csv") -> Path:
    path = DATA_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} records to {path}")
    return path


def load_dataset(filename: str = "procurement_risk.csv") -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run generate_dataset() first."
        )
    return pd.read_csv(path)


if __name__ == "__main__":
    df = generate_dataset(n_samples=2000)
    save_dataset(df)
    print(df["label"].value_counts())
    print(df.describe())
