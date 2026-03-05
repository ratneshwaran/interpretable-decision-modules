"""
Evidence fusion layer — combines ML, fuzzy, and rule-based signals.

Two strategies are provided:

1. WeightedConsensus  — simple weighted linear combination.
2. EvidenceAccumulator — Dempster-Shafer inspired accumulation that
   tracks agreement/conflict between sources and adjusts confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Input / output types
# ---------------------------------------------------------------------------

@dataclass
class EvidenceSource:
    name: str
    risk_score: float       # [0, 1]
    confidence: float       # [0, 1] — how reliable this source is
    weight: float = 1.0     # relative importance weight


@dataclass
class FusionResult:
    fused_risk: float                           # [0, 1]
    decision: Literal["approve", "review", "reject"]
    strategy: str
    source_contributions: dict[str, float]      # source → contribution score
    conflict_index: float                        # 0 = full agreement
    confidence: float                            # overall fusion confidence


# ---------------------------------------------------------------------------
# Decision thresholds
# ---------------------------------------------------------------------------

APPROVE_THRESHOLD = 0.35
REJECT_THRESHOLD  = 0.65


def _score_to_decision(fused_risk: float) -> str:
    if fused_risk < APPROVE_THRESHOLD:
        return "approve"
    if fused_risk > REJECT_THRESHOLD:
        return "reject"
    return "review"


# ---------------------------------------------------------------------------
# Strategy 1 — Weighted Consensus
# ---------------------------------------------------------------------------

class WeightedConsensus:
    """
    Fused risk = Σ (weight_i * confidence_i * risk_i) / Σ (weight_i * confidence_i)

    Simple and transparent; each source's contribution is directly proportional
    to its weight and confidence.
    """

    def fuse(self, sources: list[EvidenceSource]) -> FusionResult:
        total_w = sum(s.weight * s.confidence for s in sources)
        if total_w == 0:
            fused = 0.5
        else:
            fused = sum(s.weight * s.confidence * s.risk_score for s in sources) / total_w

        fused = round(max(0.0, min(1.0, fused)), 4)

        # Per-source contribution as fraction of total weight
        contributions = {
            s.name: round((s.weight * s.confidence) / total_w, 4) if total_w else 0.0
            for s in sources
        }

        # Conflict: std-dev of risk scores (normalised to [0,1])
        risks = [s.risk_score for s in sources]
        mean_r = sum(risks) / len(risks)
        variance = sum((r - mean_r) ** 2 for r in risks) / len(risks)
        conflict = round(variance ** 0.5, 4)

        # Confidence: weighted average of source confidences, reduced by conflict
        avg_conf = sum(s.confidence * s.weight for s in sources) / sum(s.weight for s in sources)
        overall_conf = round(avg_conf * (1 - 0.5 * conflict), 4)

        return FusionResult(
            fused_risk=fused,
            decision=_score_to_decision(fused),
            strategy="weighted_consensus",
            source_contributions=contributions,
            conflict_index=conflict,
            confidence=overall_conf,
        )


# ---------------------------------------------------------------------------
# Strategy 2 — Evidence Accumulation (Dempster-Shafer inspired)
# ---------------------------------------------------------------------------

class EvidenceAccumulator:
    """
    Iteratively accumulates belief mass from each source, applying a
    conflict-adjusted combination rule.

    Each source provides belief masses for {approve, review, reject}.
    After combining all sources, the final decision is the hypothesis
    with maximum plausibility.
    """

    HYPOTHESES = ("approve", "review", "reject")
    RISK_TO_MASS_SIGMA = 0.25   # smoothing for mass assignment

    def _risk_to_masses(self, risk: float, confidence: float) -> dict[str, float]:
        """Map a scalar risk score to belief masses over {approve, review, reject}."""
        # Soft mass assignment based on distance from canonical risk levels
        canonical = {"approve": 0.15, "review": 0.50, "reject": 0.85}
        sigma = self.RISK_TO_MASS_SIGMA

        raw = {}
        for h, center in canonical.items():
            raw[h] = max(0.0, 1 - abs(risk - center) / (2 * sigma))

        total = sum(raw.values())
        if total == 0:
            masses = {h: 1 / 3 for h in self.HYPOTHESES}
        else:
            masses = {h: raw[h] / total for h in self.HYPOTHESES}

        # Scale by confidence; remainder goes to "ignorance" (handled implicitly)
        return {h: confidence * masses[h] for h in self.HYPOTHESES}

    def _combine(
        self,
        m1: dict[str, float],
        m2: dict[str, float],
    ) -> dict[str, float]:
        """Dempster combination rule (closed world)."""
        combined: dict[str, float] = {h: 0.0 for h in self.HYPOTHESES}
        conflict = 0.0

        for h1, v1 in m1.items():
            for h2, v2 in m2.items():
                if h1 == h2:
                    combined[h1] += v1 * v2
                else:
                    conflict += v1 * v2  # K — conflict mass

        # Normalise (open-world: retain conflict as ignorance reduction)
        normaliser = 1 - conflict
        if normaliser < 1e-9:
            # Total conflict — fall back to average
            return {h: (m1[h] + m2[h]) / 2 for h in self.HYPOTHESES}

        return {h: combined[h] / normaliser for h in self.HYPOTHESES}

    def fuse(self, sources: list[EvidenceSource]) -> FusionResult:
        if not sources:
            raise ValueError("Need at least one evidence source.")

        # Initialise with first source
        current_masses = self._risk_to_masses(
            sources[0].risk_score, sources[0].confidence * sources[0].weight
        )

        for src in sources[1:]:
            new_masses = self._risk_to_masses(
                src.risk_score, src.confidence * src.weight
            )
            current_masses = self._combine(current_masses, new_masses)

        # Plausibility-based decision
        decision = max(current_masses, key=current_masses.__getitem__)

        # Convert back to scalar risk
        risk_map = {"approve": 0.15, "review": 0.50, "reject": 0.85}
        fused_risk = sum(current_masses[h] * risk_map[h] for h in self.HYPOTHESES)
        fused_risk = round(max(0.0, min(1.0, fused_risk)), 4)

        # Conflict: 1 - max mass (degree of ambiguity)
        max_mass = max(current_masses.values())
        conflict = round(1.0 - max_mass, 4)

        contributions = {
            s.name: round(s.weight * s.confidence, 4) for s in sources
        }
        total_c = sum(contributions.values()) or 1.0
        contributions = {k: round(v / total_c, 4) for k, v in contributions.items()}

        overall_conf = round(max_mass * (1 - 0.3 * conflict), 4)

        return FusionResult(
            fused_risk=fused_risk,
            decision=decision,
            strategy="evidence_accumulation",
            source_contributions=contributions,
            conflict_index=conflict,
            confidence=overall_conf,
        )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

class DecisionFuser:
    """
    High-level interface. Accepts raw scores from each pipeline stage
    and returns a FusionResult using both strategies (consensus wins by default).
    """

    def __init__(
        self,
        ml_weight: float = 0.45,
        fuzzy_weight: float = 0.35,
        rule_weight: float = 0.20,
        strategy: str = "weighted",
    ) -> None:
        self.ml_weight = ml_weight
        self.fuzzy_weight = fuzzy_weight
        self.rule_weight = rule_weight
        self.strategy = strategy
        self._wc = WeightedConsensus()
        self._ea = EvidenceAccumulator()

    def fuse(
        self,
        ml_risk: float,
        ml_confidence: float,
        fuzzy_risk: float,
        rule_risk: float,
        rule_confidence: float = 0.90,
    ) -> FusionResult:
        sources = [
            EvidenceSource("ml_model",      ml_risk,    ml_confidence,   self.ml_weight),
            EvidenceSource("fuzzy_engine",  fuzzy_risk, 0.80,            self.fuzzy_weight),
            EvidenceSource("rule_flags",    rule_risk,  rule_confidence, self.rule_weight),
        ]
        if self.strategy == "dempster":
            return self._ea.fuse(sources)
        return self._wc.fuse(sources)
