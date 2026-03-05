"""
Explanation bundle generator.

Collects outputs from all pipeline stages and assembles a structured
explanation that can be rendered in reports, API responses, or audit logs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Component explanation types
# ---------------------------------------------------------------------------

@dataclass
class FeatureContribution:
    """Single feature's importance in the ML prediction."""
    feature: str
    importance: float
    value: float
    direction: str          # "increases_risk" | "decreases_risk" | "neutral"


@dataclass
class RuleActivation:
    """A fuzzy rule that fired during inference."""
    rule_name: str
    strength: float
    consequent: str         # "low" | "medium" | "high" risk
    description: str


@dataclass
class FusionContribution:
    """Per-source contribution to the fused risk score."""
    source: str
    risk_score: float
    contribution_weight: float


@dataclass
class ExplanationBundle:
    """
    Complete explanation for a single decision.

    Intended to be serialisable (all primitives) so it can be stored
    in audit logs and rendered via Jinja2 templates.
    """
    decision: str
    fused_risk: float
    ml_risk: float
    fuzzy_risk: float

    # Ordered list of top ML feature contributions
    feature_contributions: list[FeatureContribution] = field(default_factory=list)

    # Fuzzy rules that fired (sorted by strength)
    rule_activations: list[RuleActivation] = field(default_factory=list)

    # Per-source fusion weights
    fusion_contributions: list[FusionContribution] = field(default_factory=list)

    # High-level plain-text summary
    summary: str = ""

    # Conflict and confidence metadata
    conflict_index: float = 0.0
    fusion_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def top_features(self, n: int = 3) -> list[FeatureContribution]:
        return sorted(
            self.feature_contributions, key=lambda f: f.importance, reverse=True
        )[:n]

    def dominant_rules(self, min_strength: float = 0.3) -> list[RuleActivation]:
        return [r for r in self.rule_activations if r.strength >= min_strength]


# ---------------------------------------------------------------------------
# Bundle builder
# ---------------------------------------------------------------------------

class ExplanationBuilder:
    """
    Assembles an ExplanationBundle from the outputs of each pipeline stage.
    """

    # Risk direction thresholds
    HIGH_FEATURE_IMPORTANCE = 0.15
    RISK_NEUTRAL_BAND = (0.35, 0.65)

    def build(
        self,
        *,
        record: dict[str, Any],
        ml_result: dict[str, Any],
        fuzzy_result: Any,          # FuzzyInferenceResult
        fusion_result: Any,         # FusionResult
    ) -> ExplanationBundle:
        """
        Parameters
        ----------
        record       : raw input features
        ml_result    : output from RiskModel.predict_risk()
        fuzzy_result : FuzzyInferenceResult from FuzzyRuleEngine.evaluate()
        fusion_result: FusionResult from DecisionFuser.fuse()
        """

        feature_contribs = self._build_feature_contributions(
            record, ml_result["feature_scores"], ml_result["ml_risk"]
        )

        rule_activations = [
            RuleActivation(
                rule_name=r.name,
                strength=r.strength,
                consequent=r.consequent,
                description=r.antecedent_desc,
            )
            for r in fuzzy_result.fired_rules
        ]

        fusion_contribs = [
            FusionContribution(
                source=name,
                risk_score=self._source_risk(name, ml_result, fuzzy_result),
                contribution_weight=weight,
            )
            for name, weight in fusion_result.source_contributions.items()
        ]

        summary = self._generate_summary(
            decision=fusion_result.decision,
            fused_risk=fusion_result.fused_risk,
            top_features=feature_contribs[:3],
            dominant_rule=rule_activations[0] if rule_activations else None,
            conflict=fusion_result.conflict_index,
        )

        return ExplanationBundle(
            decision=fusion_result.decision,
            fused_risk=fusion_result.fused_risk,
            ml_risk=ml_result["ml_risk"],
            fuzzy_risk=fuzzy_result.fuzzy_risk,
            feature_contributions=feature_contribs,
            rule_activations=rule_activations,
            fusion_contributions=fusion_contribs,
            summary=summary,
            conflict_index=fusion_result.conflict_index,
            fusion_confidence=fusion_result.confidence,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_feature_contributions(
        self,
        record: dict,
        feature_scores: dict[str, float],
        ml_risk: float,
    ) -> list[FeatureContribution]:
        contribs = []
        risk_increasing = {"vendor_risk", "anomaly_score", "historical_risk"}
        risk_decreasing = {"document_confidence"}

        for feat, importance in sorted(
            feature_scores.items(), key=lambda x: x[1], reverse=True
        ):
            val = record.get(feat, 0.0)
            if feat in risk_increasing:
                direction = "increases_risk" if importance > 0.05 else "neutral"
            elif feat in risk_decreasing:
                direction = "decreases_risk" if importance > 0.05 else "neutral"
            else:
                # amount — direction depends on whether normalised value is high
                norm_amount = min(val / 200_000, 1.0)
                if norm_amount > 0.5 and importance > 0.05:
                    direction = "increases_risk"
                elif norm_amount < 0.2 and importance > 0.05:
                    direction = "decreases_risk"
                else:
                    direction = "neutral"

            contribs.append(FeatureContribution(
                feature=feat,
                importance=importance,
                value=val,
                direction=direction,
            ))

        return contribs

    @staticmethod
    def _source_risk(
        source_name: str,
        ml_result: dict,
        fuzzy_result: Any,
    ) -> float:
        mapping = {
            "ml_model":     ml_result["ml_risk"],
            "fuzzy_engine": fuzzy_result.fuzzy_risk,
            "rule_flags":   fuzzy_result.fuzzy_risk,  # approximation
        }
        return mapping.get(source_name, 0.5)

    @staticmethod
    def _generate_summary(
        decision: str,
        fused_risk: float,
        top_features: list[FeatureContribution],
        dominant_rule: RuleActivation | None,
        conflict: float,
    ) -> str:
        risk_pct = int(fused_risk * 100)
        feat_names = ", ".join(f.feature for f in top_features if f.importance > 0.05)

        summary = (
            f"Decision: {decision.upper()} (fused risk {risk_pct}%). "
            f"Key ML drivers: {feat_names or 'none identified'}. "
        )

        if dominant_rule:
            summary += (
                f"Strongest fuzzy rule: '{dominant_rule.description}' "
                f"(activation {dominant_rule.strength:.2f}). "
            )

        if conflict > 0.20:
            summary += (
                f"Sources show moderate disagreement (conflict={conflict:.2f}); "
                "manual review is recommended."
            )
        elif conflict > 0.10:
            summary += f"Minor signal conflict detected (conflict={conflict:.2f})."
        else:
            summary += "Sources are in strong agreement."

        return summary
