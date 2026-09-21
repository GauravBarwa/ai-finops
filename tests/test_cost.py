"""Tests for the cost calculation module."""

from __future__ import annotations

import pytest

from app.llm.cost import CostBreakdown, calculate_cost
from app.llm.registry import ModelInfo


def _make_model(
    input_price: float = 0.15,
    output_price: float = 0.60,
) -> ModelInfo:
    """Create a minimal ModelInfo for testing."""
    return ModelInfo(
        model_id="test-model",
        provider="test",
        display_name="Test Model",
        litellm_model_name="test/test-model",
        context_window=128000,
        input_price_per_1m_tokens=input_price,
        output_price_per_1m_tokens=output_price,
    )


class TestCalculateCost:
    """Tests for ``calculate_cost``."""

    def test_basic_cost_calculation(self):
        """Verify input + output cost with known values."""
        model = _make_model(input_price=0.15, output_price=0.60)
        cost = calculate_cost(1000, 500, model)

        # input:  1000 / 1M * 0.15 = 0.00015
        # output: 500  / 1M * 0.60 = 0.0003
        # total:  0.00045
        assert cost.input_cost == pytest.approx(0.00015, abs=1e-8)
        assert cost.output_cost == pytest.approx(0.0003, abs=1e-8)
        assert cost.total_cost == pytest.approx(0.00045, abs=1e-8)

    def test_zero_tokens(self):
        """Zero tokens should produce zero cost."""
        model = _make_model()
        cost = calculate_cost(0, 0, model)
        assert cost.total_cost == 0.0
        assert cost.input_cost == 0.0
        assert cost.output_cost == 0.0

    def test_zero_pricing(self):
        """Models with zero pricing (local/free) should produce zero cost."""
        model = _make_model(input_price=0.0, output_price=0.0)
        cost = calculate_cost(10000, 5000, model)
        assert cost.total_cost == 0.0

    def test_large_token_count(self):
        """Verify correctness with large token counts."""
        model = _make_model(input_price=0.15, output_price=0.60)
        cost = calculate_cost(1_000_000, 500_000, model)

        # input:  1M / 1M * 0.15 = 0.15
        # output: 500K / 1M * 0.60 = 0.30
        # total:  0.45
        assert cost.input_cost == pytest.approx(0.15, abs=1e-6)
        assert cost.output_cost == pytest.approx(0.30, abs=1e-6)
        assert cost.total_cost == pytest.approx(0.45, abs=1e-6)

    def test_is_always_estimated(self):
        """Cost should always be marked as estimated."""
        model = _make_model()
        cost = calculate_cost(100, 100, model)
        assert cost.is_estimated is True
        assert cost.currency == "USD"

    def test_cost_breakdown_dataclass(self):
        """Verify CostBreakdown fields."""
        breakdown = CostBreakdown(
            input_cost=0.001,
            output_cost=0.002,
            total_cost=0.003,
        )
        assert breakdown.currency == "USD"
        assert breakdown.is_estimated is True
        assert breakdown.total_cost == 0.003
