"""LayerLens: portable layer and engine observability for LLM inference."""

from .adapters import (
    AdapterRegistry,
    CaptureRequest,
    RuntimeAdapter,
    adapter_registry,
    capture,
)
from .analysis import aggregate_event_grid, choose_group_size, diagnose_trace
from .cache_plan import (
    PrefixCacheSimulator,
    PrefixRequest,
    compare_prefix_schedules,
    prefix_aware_order,
)
from .metrics import VLLMMetricsSampler, parse_prometheus, summarize_vllm_metrics
from .ollama import capture_ollama, normalize_ollama_metrics
from .privacy import safe_model_identifier
from .profiler import LayerProfiler, ProfileConfig
from .product_atlas import (
    CoordinateProductCache,
    CoordinateProductCacheStats,
    ProductAtlas,
    ProductAtlasAnalysis,
    ProductAtlasBenchmark,
    analyze_weight_family,
    benchmark_product_atlas,
    exact_coordinate_hit_rate,
)
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
    "PrefixCacheSimulator",
    "PrefixRequest",
    "compare_prefix_schedules",
    "prefix_aware_order",
    "ProductAtlas",
    "CoordinateProductCache",
    "CoordinateProductCacheStats",
    "ProductAtlasAnalysis",
    "ProductAtlasBenchmark",
    "analyze_weight_family",
    "benchmark_product_atlas",
    "exact_coordinate_hit_rate",
]
__version__ = "0.7.0"
