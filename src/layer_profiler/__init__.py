"""LayerLens: portable layer and engine observability for LLM inference."""

from .metrics import VLLMMetricsSampler, parse_prometheus, summarize_vllm_metrics
from .ollama import capture_ollama, normalize_ollama_metrics
from .privacy import safe_model_identifier
from .profiler import LayerProfiler, ProfileConfig
from .trace import InferenceTrace, LayerEvent, StepEvent, TokenEvent

__all__ = [
    "InferenceTrace",
    "LayerEvent",
    "LayerProfiler",
    "ProfileConfig",
    "StepEvent",
    "TokenEvent",
    "VLLMMetricsSampler",
    "parse_prometheus",
    "summarize_vllm_metrics",
    "safe_model_identifier",
    "capture_ollama",
    "normalize_ollama_metrics",
]
__version__ = "0.3.0"
