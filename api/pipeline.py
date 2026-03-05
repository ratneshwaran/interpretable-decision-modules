"""
Decision pipeline orchestrator.

Wires together all reasoning modules into a single callable that accepts
raw input features and returns a complete decision package.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from audit.logger import AuditLogger
from explain.bundle import ExplanationBuilder, ExplanationBundle
from fusion.evidence import DecisionFuser, FusionResult
from fuzzy.rules import FuzzyInferenceResult, FuzzyRuleEngine
from models.risk_model import RiskModel


@dataclass
class PipelineOutput:
    audit_id: str
    timestamp: str
    decision: str
    fused_risk: float
    ml_risk: float
    fuzzy_risk: float
    fusion_confidence: float
    conflict_index: float
    explanation: ExplanationBundle
    fusion_result: FusionResult


class DecisionPipeline:
    """
    Full interpretable decision pipeline.

    Stages
    ------
    1. ML risk model    → risk probability + feature importances
    2. Fuzzy engine     → linguistic rule activations + defuzzified risk
    3. Evidence fusion  → weighted / Dempster combination
    4. Explanation      → human-readable bundle
    5. Audit log        → timestamped JSON trace
    """

    def __init__(
        self,
        model: RiskModel | None = None,
        fusion_strategy: str = "weighted",
        log_decisions: bool = True,
    ) -> None:
        self.model = model or RiskModel.load()
        self.fuzzy_engine = FuzzyRuleEngine()
        self.fuser = DecisionFuser(strategy=fusion_strategy)
        self.explainer = ExplanationBuilder()
        self.auditor = AuditLogger() if log_decisions else None
        self.fusion_strategy = fusion_strategy

    def run(self, record: dict[str, Any]) -> PipelineOutput:
        # Stage 1 — ML inference
        ml_result = self.model.predict_risk(record)

        # Stage 2 — Fuzzy inference
        fuzzy_result: FuzzyInferenceResult = self.fuzzy_engine.evaluate(record)

        # Rule-based hard flag: high anomaly + high vendor risk
        rule_risk = 0.9 if (
            record.get("anomaly_score", 0) > 0.6
            and record.get("vendor_risk", 0) > 0.6
        ) else 0.5 if (
            record.get("anomaly_score", 0) > 0.4
            or record.get("vendor_risk", 0) > 0.5
        ) else 0.15

        # Stage 3 — Evidence fusion
        ml_conf = max(ml_result["probabilities"].values())
        fusion_result: FusionResult = self.fuser.fuse(
            ml_risk=ml_result["ml_risk"],
            ml_confidence=ml_conf,
            fuzzy_risk=fuzzy_result.fuzzy_risk,
            rule_risk=rule_risk,
        )

        # Stage 4 — Explanation bundle
        bundle = self.explainer.build(
            record=record,
            ml_result=ml_result,
            fuzzy_result=fuzzy_result,
            fusion_result=fusion_result,
        )

        timestamp = datetime.now(timezone.utc).isoformat()

        # Stage 5 — Audit log
        audit_id = "no-log"
        if self.auditor:
            audit_id = self.auditor.log(
                inputs=record,
                ml_result=ml_result,
                fuzzy_result=fuzzy_result,
                fusion_result=fusion_result,
                explanation_bundle=bundle,
            )

        return PipelineOutput(
            audit_id=audit_id,
            timestamp=timestamp,
            decision=fusion_result.decision,
            fused_risk=fusion_result.fused_risk,
            ml_risk=ml_result["ml_risk"],
            fuzzy_risk=fuzzy_result.fuzzy_risk,
            fusion_confidence=fusion_result.confidence,
            conflict_index=fusion_result.conflict_index,
            explanation=bundle,
            fusion_result=fusion_result,
        )
