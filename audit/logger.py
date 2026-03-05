"""
Audit logger — stores timestamped decision traces as newline-delimited JSON.

Each trace contains:
  - inputs (raw feature values)
  - intermediate scores (ml_risk, fuzzy_risk)
  - rules fired (names + strengths)
  - fusion result (fused_risk, strategy, conflict)
  - final decision
  - timestamp + unique audit_id
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AUDIT_DIR = Path(__file__).parent
AUDIT_LOG = AUDIT_DIR / "decision_traces.jsonl"


class AuditLogger:
    """
    Appends decision traces to a newline-delimited JSON log file.
    Thread-safe for single-process use (file append is atomic on most OS).
    """

    def __init__(self, log_path: Path = AUDIT_LOG) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        inputs: dict[str, Any],
        ml_result: dict[str, Any],
        fuzzy_result: Any,          # FuzzyInferenceResult
        fusion_result: Any,         # FusionResult
        explanation_bundle: Any,    # ExplanationBundle
    ) -> str:
        """
        Write a decision trace and return the audit_id.
        """
        audit_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        trace = {
            "audit_id": audit_id,
            "timestamp": timestamp,
            "inputs": inputs,
            "intermediate": {
                "ml_risk": ml_result["ml_risk"],
                "ml_label": ml_result["ml_label"],
                "ml_probabilities": ml_result["probabilities"],
                "fuzzy_risk": fuzzy_result.fuzzy_risk,
                "fuzzy_memberships": fuzzy_result.memberships,
                "rules_fired": [
                    {
                        "name": r.name,
                        "strength": r.strength,
                        "consequent": r.consequent,
                        "description": r.antecedent_desc,
                    }
                    for r in fuzzy_result.fired_rules
                ],
            },
            "fusion": {
                "strategy": fusion_result.strategy,
                "fused_risk": fusion_result.fused_risk,
                "conflict_index": fusion_result.conflict_index,
                "confidence": fusion_result.confidence,
                "source_contributions": fusion_result.source_contributions,
            },
            "decision": fusion_result.decision,
            "explanation_summary": explanation_bundle.summary,
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace) + "\n")

        return audit_id

    def get_trace(self, audit_id: str) -> dict[str, Any] | None:
        """Retrieve a specific trace by audit_id (linear scan)."""
        if not self.log_path.exists():
            return None
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record.get("audit_id") == audit_id:
                    return record
        return None

    def recent_traces(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the n most recent traces."""
        if not self.log_path.exists():
            return []
        traces = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                traces.append(json.loads(line))
        return traces[-n:]

    def decision_stats(self) -> dict[str, int]:
        """Quick summary of decision counts in the log."""
        counts: dict[str, int] = {}
        if not self.log_path.exists():
            return counts
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line).get("decision", "unknown")
                counts[d] = counts.get(d, 0) + 1
        return counts
