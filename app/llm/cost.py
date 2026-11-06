"""Cost estimation utilities.

All values are **estimated** using Method A shadow commercial USD assumptions
stored in the model registry configuration. Free-tier and local actual billing
may be zero; these experimental prices are not verified current quotes.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.llm.registry import ModelInfo


@dataclass
class CostBreakdown:
    """Breakdown of estimated cost for a single LLM call.

    Attributes
    ----------
    input_cost : float
        Estimated cost of input/prompt tokens.
    output_cost : float
        Estimated cost of output/completion tokens.
    total_cost : float
        ``input_cost + output_cost``.
    currency : str
        Always ``"USD"`` for now.
    is_estimated : bool
        Always ``True`` — we don't have actual enterprise billing data.
    """

    input_cost: float
    output_cost: float
    total_cost: float
    currency: str = "USD"
    is_estimated: bool = True


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model_info: ModelInfo,
) -> CostBreakdown:
    """Estimate the cost of a model call based on token counts and registry pricing.

    Parameters
    ----------
    input_tokens:
        Number of input/prompt tokens consumed.
    output_tokens:
        Number of output/completion tokens generated.
    model_info:
        The ``ModelInfo`` entry from the registry, containing per-1M-token pricing.

    Returns
    -------
    CostBreakdown
        The itemised estimated cost.
    """
    input_cost = (input_tokens / 1_000_000) * model_info.input_price_per_1m_tokens
    output_cost = (output_tokens / 1_000_000) * model_info.output_price_per_1m_tokens
    total_cost = input_cost + output_cost

    return CostBreakdown(
        input_cost=round(input_cost, 8),
        output_cost=round(output_cost, 8),
        total_cost=round(total_cost, 8),
    )
