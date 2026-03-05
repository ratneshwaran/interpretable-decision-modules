"""
demo.py — end-to-end demonstration of the interpretable decision pipeline.

Runs three representative cases through the full pipeline and prints
structured explanations, then generates HTML reports for each.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from api.pipeline import DecisionPipeline
from reports.generator import ReportGenerator

# ---------------------------------------------------------------------------
# Example cases
# ---------------------------------------------------------------------------

CASES = [
    {
        "label": "Low-risk routine purchase",
        "record": {
            "amount": 4_200,
            "vendor_risk": 0.08,
            "document_confidence": 0.93,
            "anomaly_score": 0.04,
            "historical_risk": 0.07,
        },
    },
    {
        "label": "Mid-tier borderline case",
        "record": {
            "amount": 48_000,
            "vendor_risk": 0.45,
            "document_confidence": 0.55,
            "anomaly_score": 0.38,
            "historical_risk": 0.40,
        },
    },
    {
        "label": "High-risk large contract",
        "record": {
            "amount": 185_000,
            "vendor_risk": 0.79,
            "document_confidence": 0.22,
            "anomaly_score": 0.71,
            "historical_risk": 0.68,
        },
    },
]

SEPARATOR = "=" * 70


def print_section(title: str) -> None:
    print(f"\n{'-' * 50}")
    print(f"  {title}")
    print(f"{'-' * 50}")


def run_demo() -> None:
    print(SEPARATOR)
    print("  Interpretable Decision Modules — Demo")
    print(SEPARATOR)

    print("\nLoading pipeline (ML model + fuzzy engine + fusion layer)...")
    pipeline = DecisionPipeline()
    reporter = ReportGenerator()
    print("Pipeline ready.\n")

    for i, case in enumerate(CASES, start=1):
        print(f"\n{SEPARATOR}")
        print(f"  CASE {i}: {case['label']}")
        print(SEPARATOR)

        record = case["record"]
        print("\nInputs:")
        for k, v in record.items():
            print(f"  {k:<25} {v}")

        output = pipeline.run(record)
        bundle = output.explanation

        # Decision
        print_section("Decision")
        decision_icons = {"approve": "[APPROVE]", "review": "[REVIEW] ", "reject": "[REJECT] "}
        icon = decision_icons.get(output.decision, "[?]")
        print(f"  {icon}  {output.decision.upper()}")
        print(f"  Fused risk     : {output.fused_risk:.1%}")
        print(f"  ML risk        : {output.ml_risk:.1%}")
        print(f"  Fuzzy risk     : {output.fuzzy_risk:.1%}")
        print(f"  Fusion conf.   : {output.fusion_confidence:.1%}")
        print(f"  Conflict index : {output.conflict_index:.3f}")
        print(f"  Audit ID       : {output.audit_id}")

        # Summary
        print_section("Explanation Summary")
        # Word-wrap at 65 chars
        words = bundle.summary.split()
        line, lines = [], []
        for word in words:
            if sum(len(w) + 1 for w in line) + len(word) > 65:
                lines.append(" ".join(line))
                line = [word]
            else:
                line.append(word)
        if line:
            lines.append(" ".join(line))
        for ln in lines:
            print(f"  {ln}")

        # ML Feature contributions
        print_section("ML Feature Contributions (top 5)")
        for fc in bundle.feature_contributions:
            bar = "#" * int(fc.importance * 30)
            print(f"  {fc.feature:<25} {bar:<10} {fc.importance:.3f}  ({fc.direction})")

        # Fuzzy rules
        print_section("Fuzzy Rules Fired")
        for rule in bundle.rule_activations:
            strength_bar = "*" * int(rule.strength * 10)
            print(f"  [{strength_bar:<10}] {rule.strength:.3f}  {rule.rule_name}")
            print(f"             IF {rule.description} THEN risk is {rule.consequent.upper()}")

        # Fusion breakdown
        print_section("Evidence Fusion Breakdown")
        for fc in bundle.fusion_contributions:
            bar = "=" * int(fc.contribution_weight * 30)
            print(f"  {fc.source:<20} {bar:<12} {fc.contribution_weight:.1%}  (risk={fc.risk_score:.1%})")

        # HTML report
        html = reporter.render(
            audit_id=output.audit_id,
            timestamp=output.timestamp,
            inputs=record,
            bundle=bundle,
            ml_label=output.explanation.decision,
            fusion_strategy=pipeline.fusion_strategy,
        )
        report_path = reporter.save(html, audit_id=output.audit_id)
        print(f"\n  Report saved: {report_path}")

    print(f"\n{SEPARATOR}")
    print("  Demo complete. Check reports/ for HTML output.")
    print(SEPARATOR)


if __name__ == "__main__":
    run_demo()
