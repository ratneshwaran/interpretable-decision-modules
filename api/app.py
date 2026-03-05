"""
FastAPI application — POST /decision endpoint.

Accepts raw procurement features and returns:
  - decision (approve / review / reject)
  - fused_risk score
  - explanation bundle
  - audit_id for trace retrieval
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from api.pipeline import DecisionPipeline, PipelineOutput
from audit.logger import AuditLogger


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class DecisionRequest(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    vendor_risk: float = Field(..., ge=0.0, le=1.0)
    document_confidence: float = Field(..., ge=0.0, le=1.0)
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    historical_risk: float = Field(0.3, ge=0.0, le=1.0)
    fusion_strategy: str = Field("weighted", pattern="^(weighted|dempster)$")

    @field_validator("amount")
    @classmethod
    def cap_amount(cls, v: float) -> float:
        return min(v, 500_000.0)


class FeatureContributionOut(BaseModel):
    feature: str
    importance: float
    value: float
    direction: str


class RuleActivationOut(BaseModel):
    rule_name: str
    strength: float
    consequent: str
    description: str


class FusionContributionOut(BaseModel):
    source: str
    risk_score: float
    contribution_weight: float


class ExplanationOut(BaseModel):
    summary: str
    feature_contributions: list[FeatureContributionOut]
    rule_activations: list[RuleActivationOut]
    fusion_contributions: list[FusionContributionOut]


class DecisionResponse(BaseModel):
    audit_id: str
    timestamp: str
    decision: str
    fused_risk: float
    ml_risk: float
    fuzzy_risk: float
    fusion_confidence: float
    conflict_index: float
    explanation: ExplanationOut


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

_pipeline_cache: dict[str, DecisionPipeline] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load pipelines for both strategies to avoid cold-start latency
    for strategy in ("weighted", "dempster"):
        _pipeline_cache[strategy] = DecisionPipeline(fusion_strategy=strategy)
    yield
    _pipeline_cache.clear()


app = FastAPI(
    title="Interpretable Decision Modules",
    description="Interpretable procurement risk decisions via ML + fuzzy inference + evidence fusion",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/decision", response_model=DecisionResponse)
async def make_decision(req: DecisionRequest) -> DecisionResponse:
    pipeline = _pipeline_cache.get(req.fusion_strategy)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    record = req.model_dump(exclude={"fusion_strategy"})

    try:
        output: PipelineOutput = pipeline.run(record)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    bundle = output.explanation
    explanation = ExplanationOut(
        summary=bundle.summary,
        feature_contributions=[
            FeatureContributionOut(**vars(fc)) for fc in bundle.feature_contributions
        ],
        rule_activations=[
            RuleActivationOut(
                rule_name=r.rule_name,
                strength=r.strength,
                consequent=r.consequent,
                description=r.description,
            )
            for r in bundle.rule_activations
        ],
        fusion_contributions=[
            FusionContributionOut(**vars(fc)) for fc in bundle.fusion_contributions
        ],
    )

    return DecisionResponse(
        audit_id=output.audit_id,
        timestamp=output.timestamp,
        decision=output.decision,
        fused_risk=output.fused_risk,
        ml_risk=output.ml_risk,
        fuzzy_risk=output.fuzzy_risk,
        fusion_confidence=output.fusion_confidence,
        conflict_index=output.conflict_index,
        explanation=explanation,
    )


@app.get("/audit/{audit_id}")
async def get_audit_trace(audit_id: str) -> dict[str, Any]:
    logger = AuditLogger()
    trace = logger.get_trace(audit_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Audit trace {audit_id!r} not found")
    return trace


@app.get("/audit/stats/summary")
async def audit_stats() -> dict[str, Any]:
    logger = AuditLogger()
    return {
        "decision_counts": logger.decision_stats(),
        "recent": logger.recent_traces(n=5),
    }
