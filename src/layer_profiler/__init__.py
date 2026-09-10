"""LayerLens: portable layer and engine observability for LLM inference."""

from .adapters import (
    AdapterRegistry,
    CaptureRequest,
    RuntimeAdapter,
    adapter_registry,
    capture,
)
from .analysis import aggregate_event_grid, choose_group_size, diagnose_trace
from .metrics import VLLMMetricsSampler, parse_prometheus, summarize_vllm_metrics
from .ollama import capture_ollama, normalize_ollama_metrics
from .privacy import safe_model_identifier
from .profiler import LayerProfiler, ProfileConfig
from .trace import InferenceTrace, LayerEvent, RuntimeEvent, StepEvent, TokenEvent

__all__ = [
    "InferenceTrace",
    "LayerEvent",
    "LayerProfiler",
    "ProfileConfig",
    "StepEvent",
    "TokenEvent",
    "RuntimeEvent",
    "VLLMMetricsSampler",
    "parse_prometheus",
    "summarize_vllm_metrics",
    "safe_model_identifier",
    "capture_ollama",
    "normalize_ollama_metrics",
    "AdapterRegistry",
    "CaptureRequest",
    "RuntimeAdapter",
    "adapter_registry",
    "capture",
    "aggregate_event_grid",
    "choose_group_size",
    "diagnose_trace",
]
__version__ = "0.5.0"
