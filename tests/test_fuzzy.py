"""Unit tests for fuzzy membership functions and rule engine."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fuzzy.membership import (
    AmountMembership,
    AnomalyMembership,
    DocumentConfidenceMembership,
    VendorRiskMembership,
    shoulder_left,
    shoulder_right,
    trapezoid,
    triangle,
)
from fuzzy.rules import FuzzyRuleEngine


class TestPrimitiveFunctions:
    def test_trapezoid_flat_top(self):
        assert trapezoid(0.5, 0.0, 0.3, 0.7, 1.0) == 1.0

    def test_trapezoid_left_slope(self):
        v = trapezoid(0.15, 0.0, 0.3, 0.7, 1.0)
        assert 0.0 < v < 1.0

    def test_trapezoid_zero_outside(self):
        assert trapezoid(-0.1, 0.0, 0.3, 0.7, 1.0) == 0.0
        assert trapezoid(1.1, 0.0, 0.3, 0.7, 1.0) == 0.0

    def test_triangle_peak(self):
        assert triangle(0.5, 0.0, 0.5, 1.0) == 1.0

    def test_shoulder_left_full(self):
        assert shoulder_left(0.1, 0.5, 0.8) == pytest.approx(1.0, abs=1e-6)

    def test_shoulder_right_full(self):
        assert shoulder_right(1.0, 0.2, 0.5) == 1.0


class TestAmountMembership:
    def test_very_low_amount_is_low(self):
        m = AmountMembership.evaluate(1000)
        assert m["low"] == 1.0
        assert m["high"] == 0.0

    def test_mid_amount_is_medium(self):
        m = AmountMembership.evaluate(40000)
        assert m["medium"] > 0.5

    def test_high_amount_is_high(self):
        m = AmountMembership.evaluate(300000)
        assert m["high"] == 1.0
        assert m["low"] == 0.0

    def test_all_values_non_negative(self):
        for amount in [500, 5000, 30000, 100000, 500000]:
            m = AmountMembership.evaluate(amount)
            assert all(v >= 0.0 for v in m.values())


class TestVendorRiskMembership:
    def test_low_vendor_risk(self):
        m = VendorRiskMembership.evaluate(0.1)
        assert m["low"] > m["high"]

    def test_high_vendor_risk(self):
        m = VendorRiskMembership.evaluate(0.9)
        assert m["high"] > m["low"]


class TestFuzzyRuleEngine:
    def setup_method(self):
        self.engine = FuzzyRuleEngine()

    def test_high_risk_case(self):
        record = {
            "amount": 250000,
            "vendor_risk": 0.85,
            "document_confidence": 0.15,
            "anomaly_score": 0.80,
        }
        result = self.engine.evaluate(record)
        assert result.fuzzy_risk > 0.70
        assert len(result.fired_rules) > 0
        high_rules = [r for r in result.fired_rules if r.consequent == "high"]
        assert len(high_rules) > 0

    def test_low_risk_case(self):
        record = {
            "amount": 3000,
            "vendor_risk": 0.05,
            "document_confidence": 0.95,
            "anomaly_score": 0.03,
        }
        result = self.engine.evaluate(record)
        assert result.fuzzy_risk < 0.30

    def test_risk_in_unit_interval(self):
        for amount in [500, 50000, 400000]:
            for vr in [0.1, 0.5, 0.9]:
                record = {
                    "amount": amount,
                    "vendor_risk": vr,
                    "document_confidence": 0.5,
                    "anomaly_score": 0.3,
                }
                result = self.engine.evaluate(record)
                assert 0.0 <= result.fuzzy_risk <= 1.0
