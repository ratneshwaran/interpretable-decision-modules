"""
Fuzzy rule engine for the interpretable decision system.

Rules encode domain expertise as IF-THEN statements using Mamdani-style
inference. Antecedents are combined with min() (AND) or max() (OR).
The consequent is a risk level (low / medium / high) with a firing strength.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from fuzzy.membership import (
    AmountMembership,
    AnomalyMembership,
    DocumentConfidenceMembership,
    VendorRiskMembership,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FiredRule:
    name: str
    strength: float          # degree of rule activation [0, 1]
    consequent: str          # "low" | "medium" | "high" risk
    antecedent_desc: str     # human-readable description


@dataclass
class FuzzyInferenceResult:
    fuzzy_risk: float                   # defuzzified risk in [0, 1]
    fired_rules: list[FiredRule] = field(default_factory=list)
    memberships: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

# Each rule is (name, antecedent_fn, consequent, description)
# antecedent_fn receives the membership dicts and returns activation strength

def _build_rules() -> list[tuple[str, Callable, str, str]]:
    rules = [
        (
            "HIGH_AMOUNT_LOW_CONFIDENCE",
            lambda m: min(m["amount"]["high"], m["doc_conf"]["low"]),
            "high",
            "amount is HIGH and document_confidence is LOW",
        ),
        (
            "HIGH_VENDOR_RISK_HIGH_AMOUNT",
            lambda m: min(m["vendor"]["high"], m["amount"]["high"]),
            "high",
            "vendor_risk is HIGH and amount is HIGH",
        ),
        (
            "HIGH_ANOMALY",
            lambda m: m["anomaly"]["high"],
            "high",
            "anomaly_score is HIGH",
        ),
        (
            "HIGH_VENDOR_RISK_MEDIUM_AMOUNT",
            lambda m: min(m["vendor"]["high"], m["amount"]["medium"]),
            "medium",
            "vendor_risk is HIGH and amount is MEDIUM",
        ),
        (
            "MEDIUM_AMOUNT_LOW_CONFIDENCE",
            lambda m: min(m["amount"]["medium"], m["doc_conf"]["low"]),
            "medium",
            "amount is MEDIUM and document_confidence is LOW",
        ),
        (
            "MEDIUM_ANOMALY_MEDIUM_VENDOR",
            lambda m: min(m["anomaly"]["medium"], m["vendor"]["medium"]),
            "medium",
            "anomaly_score is MEDIUM and vendor_risk is MEDIUM",
        ),
        (
            "LOW_AMOUNT_HIGH_CONFIDENCE",
            lambda m: min(m["amount"]["low"], m["doc_conf"]["high"]),
            "low",
            "amount is LOW and document_confidence is HIGH",
        ),
        (
            "LOW_VENDOR_RISK_LOW_AMOUNT",
            lambda m: min(m["vendor"]["low"], m["amount"]["low"]),
            "low",
            "vendor_risk is LOW and amount is LOW",
        ),
        (
            "LOW_VENDOR_HIGH_CONFIDENCE",
            lambda m: min(m["vendor"]["low"], m["doc_conf"]["high"]),
            "low",
            "vendor_risk is LOW and document_confidence is HIGH",
        ),
        (
            "LOW_ANOMALY_LOW_VENDOR",
            lambda m: min(m["anomaly"]["low"], m["vendor"]["low"]),
            "low",
            "anomaly_score is LOW and vendor_risk is LOW",
        ),
    ]
    return rules


RULES = _build_rules()

# Crisp risk values for consequent terms (centroid approximation)
CONSEQUENT_VALUES = {"low": 0.15, "medium": 0.50, "high": 0.85}


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class FuzzyRuleEngine:
    """Mamdani fuzzy inference with centroid defuzzification."""

    def __init__(self, min_activation: float = 0.01) -> None:
        # Rules with strength below this threshold are omitted from output
        self.min_activation = min_activation

    def evaluate(self, record: dict) -> FuzzyInferenceResult:
        """
        Parameters
        ----------
        record : dict
            Must contain: amount, vendor_risk, document_confidence, anomaly_score

        Returns
        -------
        FuzzyInferenceResult
        """
        # Fuzzify inputs
        memberships = {
            "amount":  AmountMembership.evaluate(record["amount"]),
            "vendor":  VendorRiskMembership.evaluate(record["vendor_risk"]),
            "doc_conf": DocumentConfidenceMembership.evaluate(record["document_confidence"]),
            "anomaly": AnomalyMembership.evaluate(record["anomaly_score"]),
        }

        # Fire rules
        fired: list[FiredRule] = []
        for name, antecedent_fn, consequent, desc in RULES:
            strength = float(antecedent_fn(memberships))
            if strength >= self.min_activation:
                fired.append(FiredRule(
                    name=name,
                    strength=round(strength, 4),
                    consequent=consequent,
                    antecedent_desc=desc,
                ))

        # Defuzzify using weighted average of consequent crisp values
        if fired:
            total_strength = sum(r.strength for r in fired)
            fuzzy_risk = sum(
                r.strength * CONSEQUENT_VALUES[r.consequent] for r in fired
            ) / total_strength
        else:
            fuzzy_risk = 0.50  # fallback to neutral

        # Sort by descending strength for readability
        fired.sort(key=lambda r: r.strength, reverse=True)

        return FuzzyInferenceResult(
            fuzzy_risk=round(fuzzy_risk, 4),
            fired_rules=fired,
            memberships=memberships,
        )
