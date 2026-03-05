"""
Fuzzy membership functions for the interpretable decision system.

Each feature is mapped to linguistic terms (low / medium / high) using
trapezoidal and triangular membership functions, which provide smooth
transitions rather than hard thresholds.
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Primitive membership function shapes
# ---------------------------------------------------------------------------

def trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
    """
    Trapezoidal membership function.

        1      _____
              /     \\
             /       \\
        0  --+--+--+--+--
             a  b  c  d

    Returns 0 outside [a, d], rises linearly from a→b,
    flat at 1 from b→c, falls linearly from c→d.
    """
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def triangle(x: float, a: float, b: float, c: float) -> float:
    """Triangular membership function — peak at b."""
    return trapezoid(x, a, b, b, c)


def shoulder_left(x: float, a: float, b: float) -> float:
    """Left-shoulder: 1 for x ≤ a, falls to 0 at b."""
    return trapezoid(x, -1e9, a, a, b)


def shoulder_right(x: float, a: float, b: float) -> float:
    """Right-shoulder: rises from 0 at a, stays 1 for x ≥ b."""
    return trapezoid(x, a, b, 1e9, 1e9 + 1)


# ---------------------------------------------------------------------------
# Feature-specific membership functions
# ---------------------------------------------------------------------------

class AmountMembership:
    """
    Linguistic mapping for transaction amount (raw USD).
    Thresholds chosen to reflect typical procurement tiers.
    """
    LOW_CAP = 10_000
    MED_CORE_LO = 8_000
    MED_CORE_HI = 75_000
    HIGH_FLOOR = 60_000
    HIGH_CAP = 200_000

    @classmethod
    def low(cls, amount: float) -> float:
        return shoulder_left(amount, cls.LOW_CAP * 0.6, cls.LOW_CAP * 1.4)

    @classmethod
    def medium(cls, amount: float) -> float:
        return trapezoid(amount,
                         cls.MED_CORE_LO * 0.5, cls.MED_CORE_LO,
                         cls.MED_CORE_HI, cls.MED_CORE_HI * 1.3)

    @classmethod
    def high(cls, amount: float) -> float:
        return shoulder_right(amount, cls.HIGH_FLOOR, cls.HIGH_CAP)

    @classmethod
    def evaluate(cls, amount: float) -> dict[str, float]:
        return {
            "low": round(cls.low(amount), 4),
            "medium": round(cls.medium(amount), 4),
            "high": round(cls.high(amount), 4),
        }


class VendorRiskMembership:
    """Linguistic mapping for vendor_risk score [0, 1]."""

    @staticmethod
    def low(v: float) -> float:
        return shoulder_left(v, 0.25, 0.40)

    @staticmethod
    def medium(v: float) -> float:
        return trapezoid(v, 0.20, 0.35, 0.60, 0.75)

    @staticmethod
    def high(v: float) -> float:
        return shoulder_right(v, 0.60, 0.80)

    @classmethod
    def evaluate(cls, v: float) -> dict[str, float]:
        return {
            "low": round(cls.low(v), 4),
            "medium": round(cls.medium(v), 4),
            "high": round(cls.high(v), 4),
        }


class DocumentConfidenceMembership:
    """
    Linguistic mapping for document_confidence [0, 1].
    Note: high confidence → low risk, so terms are inverted relative to risk.
    """

    @staticmethod
    def low(c: float) -> float:
        return shoulder_left(c, 0.30, 0.50)

    @staticmethod
    def medium(c: float) -> float:
        return trapezoid(c, 0.30, 0.45, 0.70, 0.85)

    @staticmethod
    def high(c: float) -> float:
        return shoulder_right(c, 0.65, 0.85)

    @classmethod
    def evaluate(cls, c: float) -> dict[str, float]:
        return {
            "low": round(cls.low(c), 4),
            "medium": round(cls.medium(c), 4),
            "high": round(cls.high(c), 4),
        }


class AnomalyMembership:
    """Linguistic mapping for anomaly_score [0, 1]."""

    @staticmethod
    def low(a: float) -> float:
        return shoulder_left(a, 0.20, 0.35)

    @staticmethod
    def medium(a: float) -> float:
        return trapezoid(a, 0.20, 0.35, 0.55, 0.70)

    @staticmethod
    def high(a: float) -> float:
        return shoulder_right(a, 0.55, 0.75)

    @classmethod
    def evaluate(cls, a: float) -> dict[str, float]:
        return {
            "low": round(cls.low(a), 4),
            "medium": round(cls.medium(a), 4),
            "high": round(cls.high(a), 4),
        }
