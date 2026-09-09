"""Offline layer-by-layer observability for autoregressive LLM inference."""

from .profiler import LayerProfiler, ProfileConfig
from .trace import InferenceTrace, LayerEvent, StepEvent, TokenEvent

__all__ = [
    "InferenceTrace",
    "LayerEvent",
    "LayerProfiler",
    "ProfileConfig",
    "StepEvent",
    "TokenEvent",
]
__version__ = "0.1.0"

