"""Abstract LLM client interface and response dataclass.

The rest of the application depends on ``LLMClient`` — never on a specific
provider SDK.  Concrete implementations live in sibling modules
(e.g. ``litellm_client.py``).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LLMResponse:
    """Unified response from any LLM provider call.

    Attributes
    ----------
    content : str
        The generated text.
    input_tokens : int
        Number of input/prompt tokens consumed.
    output_tokens : int
        Number of output/completion tokens generated.
    total_tokens : int
        ``input_tokens + output_tokens`` (convenience).
    latency_ms : float
        Wall-clock duration of the provider call in milliseconds.
    model : str
        The model identifier used for this call.
    provider : str
        The provider name (e.g. ``openai``, ``google``, ``anthropic``).
    success : bool
        Whether the call completed without error.
    error : str | None
        Error message if the call failed, ``None`` otherwise.
    raw_response : Any
        The original provider SDK response object (for debugging).
    """

    content: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    provider: str = ""
    success: bool = True
    error: Optional[str] = None
    raw_response: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Keep the aggregate token count correct for adapters that omit it."""
        if self.total_tokens == 0 and (self.input_tokens or self.output_tokens):
            self.total_tokens = self.input_tokens + self.output_tokens


class LLMClient(abc.ABC):
    """Abstract base class for LLM provider clients.

    Subclasses must implement :meth:`generate`.
    """

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float = 60.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send *prompt* to *model* and return a unified ``LLMResponse``.

        Parameters
        ----------
        prompt:
            The user prompt / instruction.
        model:
            The LiteLLM-compatible model string (e.g. ``gemini/gemini-3.6-flash``).
        max_tokens:
            Maximum tokens to generate.
        temperature:
            Sampling temperature.
        timeout:
            Request timeout in seconds.
        """
        ...
