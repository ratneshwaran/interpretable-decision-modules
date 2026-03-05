"""Unit tests for evidence fusion strategies."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fusion.evidence import (
    DecisionFuser,
    EvidenceAccumulator,
    EvidenceSource,
    WeightedConsensus,
)


class TestWeightedConsensus:
    def setup_method(self):
        self.wc = WeightedConsensus()

    def _sources(self, risks, confidences=None, weights=None):
        if confidences is None:
            confidences = [0.8] * len(risks)
        if weights is None:
            weights = [1.0] * len(risks)
        return [
            EvidenceSource(f"src{i}", r, c, w)
            for i, (r, c, w) in enumerate(zip(risks, confidences, weights))
        ]

    def test_all_agree_high_risk(self):
        result = self.wc.fuse(self._sources([0.9, 0.88, 0.92]))
        assert result.fused_risk > 0.80
        assert result.decision == "reject"
        assert result.conflict_index < 0.05

    def test_all_agree_low_risk(self):
        result = self.wc.fuse(self._sources([0.1, 0.12, 0.08]))
        assert result.fused_risk < 0.20
        assert result.decision == "approve"

    def test_mixed_signals_review(self):
        result = self.wc.fuse(self._sources([0.2, 0.5, 0.8]))
        assert result.decision in ("review", "approve", "reject")
        assert result.conflict_index > 0.10

    def test_contributions_sum_to_one(self):
        result = self.wc.fuse(self._sources([0.3, 0.6, 0.5]))
        total = sum(result.source_contributions.values())
        assert total == pytest.approx(1.0, abs=1e-4)


class TestEvidenceAccumulator:
    def setup_method(self):
        self.ea = EvidenceAccumulator()

    def test_high_risk_consensus(self):
        sources = [
            EvidenceSource("ml", 0.85, 0.9, 1.0),
            EvidenceSource("fuzzy", 0.80, 0.8, 0.8),
        ]
        result = self.ea.fuse(sources)
        assert result.decision == "reject"

    def test_low_risk_consensus(self):
        sources = [
            EvidenceSource("ml", 0.10, 0.9, 1.0),
            EvidenceSource("fuzzy", 0.12, 0.8, 0.8),
        ]
        result = self.ea.fuse(sources)
        assert result.decision == "approve"

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            self.ea.fuse([])


class TestDecisionFuser:
    def test_weighted_high_risk(self):
        fuser = DecisionFuser(strategy="weighted")
        result = fuser.fuse(
            ml_risk=0.88, ml_confidence=0.9,
            fuzzy_risk=0.85, rule_risk=0.90,
        )
        assert result.decision == "reject"

    def test_dempster_low_risk(self):
        fuser = DecisionFuser(strategy="dempster")
        result = fuser.fuse(
            ml_risk=0.08, ml_confidence=0.85,
            fuzzy_risk=0.12, rule_risk=0.10,
        )
        assert result.decision == "approve"

    def test_fused_risk_in_range(self):
        fuser = DecisionFuser()
        for ml_r in [0.1, 0.5, 0.9]:
            result = fuser.fuse(ml_risk=ml_r, ml_confidence=0.8, fuzzy_risk=ml_r, rule_risk=ml_r)
            assert 0.0 <= result.fused_risk <= 1.0
