"""Opt-in layer timing injection for a vLLM Python process.

Mount this directory into a vLLM container and put it on ``PYTHONPATH``.
Profiling remains dormant until the control file contains a run ID.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path


try:
    import torch
except ImportError:
    torch = None


if torch is not None and os.environ.get("LLM_LAYER_PROFILER", "0") == "1":
    _control_path = Path(
        os.environ.get("LLM_LAYER_PROFILE_CONTROL", "/profiles/layer-profiler.control")
    )
    _output_path = Path(
        os.environ.get("LLM_LAYER_PROFILE_EVENTS", "/profiles/vllm-layer-events.jsonl")
    )
    _class_pattern = re.compile(
        os.environ.get(
            "LLM_LAYER_CLASS_PATTERN",
            r"(?:DecoderLayer|TransformerLayer|TransformerBlock|Block)$",
        )
    )
    _original_call_impl = torch.nn.Module._call_impl
    _state_lock = threading.Lock()
    _write_lock = threading.Lock()
    _module_indices: dict[int, int] = {}
    _next_index = 0
    _step = -1
    _phase = "unscoped"
    _active_run = ""

    def _run_id() -> str:
        try:
            value = _control_path.read_text(encoding="utf-8").strip()
            return "" if value == "off" else value
        except OSError:
            return ""

    def _synchronize() -> None:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.synchronize()
        except RuntimeError:
            pass

    def _shape(value):
        if hasattr(value, "shape"):
            try:
                return [int(size) for size in value.shape]
            except (TypeError, RuntimeError):
                return []
        if isinstance(value, (tuple, list)):
            for child in value:
                shape = _shape(child)
                if shape:
                    return shape
        if isinstance(value, dict):
            for child in value.values():
                shape = _shape(child)
                if shape:
                    return shape
        return []

    def _write(row: dict) -> None:
        try:
            _output_path.parent.mkdir(parents=True, exist_ok=True)
            with _write_lock, _output_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        except OSError:
            pass

    def _module_device(module) -> str:
        try:
            return str(next(module.parameters()).device)
        except StopIteration:
            return "unknown"

    def _profiled_call_impl(module, *args, **kwargs):
        global _active_run, _next_index, _phase, _step

        class_name = type(module).__name__
        if not _class_pattern.search(class_name):
            return _original_call_impl(module, *args, **kwargs)

        run_id = _run_id()
        if not run_id:
            return _original_call_impl(module, *args, **kwargs)

        with _state_lock:
            if run_id != _active_run:
                _active_run = run_id
                _module_indices.clear()
                _next_index = 0
                _step = -1
                _phase = "unscoped"

            identity = id(module)
            if identity not in _module_indices:
                _module_indices[identity] = _next_index
                _next_index += 1
            layer_index = _module_indices[identity]

            input_shape = _shape(args) or _shape(kwargs)
            if layer_index == 0:
                _step += 1
            current_step = _step

        _synchronize()
        started_ns = time.perf_counter_ns()
        output = _original_call_impl(module, *args, **kwargs)
        _synchronize()
        ended_ns = time.perf_counter_ns()
        output_shape = _shape(output)

        with _state_lock:
            if layer_index == 0:
                # vLLM's positional metadata can be shaped [fields, tokens], while
                # a decoder layer's hidden-state output is [tokens, hidden_size].
                token_count = output_shape[-2] if len(output_shape) >= 2 else None
                _phase = "prefill" if current_step == 0 or (token_count and token_count > 1) else "decode"
            current_phase = _phase

        _write(
            {
                "run_id": run_id,
                "step": current_step,
                "phase": current_phase,
                "token_index": current_step,
                "layer_index": layer_index,
                "module": f"layers.{layer_index}",
                "module_type": class_name,
                "started_ns": started_ns,
                "duration_ns": ended_ns - started_ns,
                "input_shapes": [input_shape] if input_shape else [],
                "output_shapes": [output_shape] if output_shape else [],
                "device": _module_device(module),
            }
        )
        return output

    torch.nn.Module._call_impl = _profiled_call_impl
