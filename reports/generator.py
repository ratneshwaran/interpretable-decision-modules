"""
Report generator — renders Jinja2 HTML templates from ExplanationBundle data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

REPORTS_DIR = Path(__file__).parent
TEMPLATE_FILE = "decision_report.html.j2"


class ReportGenerator:
    """Renders decision reports to HTML files."""

    def __init__(self, template_dir: Path = REPORTS_DIR) -> None:
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=True,
        )
        self.template = self.env.get_template(TEMPLATE_FILE)

    def render(
        self,
        *,
        audit_id: str,
        timestamp: str,
        inputs: dict[str, Any],
        bundle: Any,            # ExplanationBundle
        ml_label: str,
        fusion_strategy: str,
    ) -> str:
        """Return rendered HTML as a string."""
        context = {
            "audit_id": audit_id,
            "timestamp": timestamp,
            "inputs": inputs,
            "decision": bundle.decision,
            "fused_risk": bundle.fused_risk,
            "ml_risk": bundle.ml_risk,
            "ml_label": ml_label,
            "fuzzy_risk": bundle.fuzzy_risk,
            "summary": bundle.summary,
            "feature_contributions": bundle.feature_contributions,
            "rule_activations": bundle.rule_activations,
            "fusion_contributions": bundle.fusion_contributions,
            "fusion_strategy": fusion_strategy,
            "conflict_index": bundle.conflict_index,
            "fusion_confidence": bundle.fusion_confidence,
        }
        return self.template.render(**context)

    def save(
        self,
        html: str,
        audit_id: str,
        output_dir: Path = REPORTS_DIR,
    ) -> Path:
        """Write HTML report to disk and return the path."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"report_{audit_id[:8]}.html"
        path.write_text(html, encoding="utf-8")
        return path
