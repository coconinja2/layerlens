"""PyTorch hook-based layer profiler for autoregressive inference."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .trace import (
    InferenceTrace,
    LayerEvent,
    StepEvent,
    TokenEvent,
    environment_metadata,
)


@dataclass(frozen=True)
class ProfileConfig:
    """Selects measurement granularity and device instrumentation."""

    module_pattern: str = r"(?:^|\.)(?:layers|h|blocks|block)\.\d+$"
    leaf_modules: bool = False
    synchronize_device: bool = True
    record_shapes: bool = True
    record_memory: bool = True


def _walk_tensors(value: Any) -> Iterable[Any]:
    if hasattr(value, "shape") and hasattr(value, "device"):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _walk_tensors(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _walk_tensors(child)


def _shapes(value: Any) -> list[list[int]]:
    shapes: list[list[int]] = []
    for tensor in _walk_tensors(value):
        try:
            shapes.append([int(size) for size in tensor.shape])
        except (TypeError, RuntimeError):
            continue
    return shapes


class LayerProfiler:
    """Capture transformer-block timing for prefill and decode passes.

    The profiler is deliberately offline and synchronizes accelerator work at
    measurement boundaries by default. This adds overhead, but avoids reporting
    asynchronous kernel-launch time as if it were completed device work.
    """

    def __init__(self, model: Any, config: ProfileConfig | None = None):
        self.model = model
        self.config = config or ProfileConfig()
        self.events: list[LayerEvent] = []
        self.steps: list[StepEvent] = []
        self.tokens: list[TokenEvent] = []
        self.errors: list[str] = []

        self._handles: list[Any] = []
        self._layer_starts: dict[int, list[tuple[int, int | None]]] = {}
        self._step_starts: list[tuple[int, int | None]] = []
        self._selected_names: list[str] = []
        self._step = -1
        self._phase = "unscoped"
        self._token_index: int | None = None
        self._origin_ns = 0
        self._sequence = 0
        self._automatic_steps = False

    def __enter__(self) -> "LayerProfiler":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc is not None:
            self.errors.append(f"{exc_type.__name__}: {exc}")
        self.stop()

    def start(self) -> None:
        if self._handles:
            return
        self._origin_ns = time.perf_counter_ns()

        self._handles.append(
            self.model.register_forward_pre_hook(self._root_pre_hook, with_kwargs=True)
        )
        self._handles.append(
            self.model.register_forward_hook(self._root_post_hook, with_kwargs=True)
        )

        for name, module in self._selected_modules():
            module_id = id(module)
            self._selected_names.append(name)
            self._layer_starts[module_id] = []
            self._handles.append(module.register_forward_pre_hook(self._layer_pre_hook(module_id)))
            self._handles.append(module.register_forward_hook(self._layer_post_hook(name, module_id)))

        if not self._selected_names:
            self.errors.append(
                "No modules matched the selection. Pass --module-pattern for this architecture."
            )

    def stop(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def begin_step(self, step: int, token_index: int | None = None) -> None:
        """Set explicit step context for a custom generation loop."""
        self._step = step
        self._phase = "prefill" if step == 0 else "decode"
        self._token_index = step if token_index is None else token_index

    def profile_forward(
        self,
        forward: Callable[..., Any],
        *args: Any,
        step: int,
        token_index: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Profile one explicit prefill or decode forward call."""
        self._automatic_steps = False
        self.begin_step(step, token_index)
        return forward(*args, **kwargs)

    def profile_generate(self, generate: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Profile a Hugging Face-style ``generate`` call automatically."""
        self._automatic_steps = True
        self._step = -1
        try:
            return generate(*args, **kwargs)
        finally:
            self._automatic_steps = False

    def attach_generated_tokens(
        self,
        token_ids: Iterable[int],
        decode: Callable[[int], str] | None = None,
        include_ids: bool = True,
    ) -> None:
        """Attach generated token IDs/text to already-recorded forward steps."""
        by_step = {step.step: step for step in self.steps}
        for token_index, raw_token_id in enumerate(token_ids):
            token_id = int(raw_token_id)
            step = token_index
            step_event = by_step.get(step)
            emitted_ns = (
                step_event.started_ns + step_event.duration_ns if step_event is not None else None
            )
            self.tokens.append(
                TokenEvent(
                    token_index=token_index,
                    step=step,
                    token_id=token_id if include_ids else None,
                    text=decode(token_id) if decode is not None else None,
                    emitted_ns=emitted_ns,
                )
            )

    def mark_token(self, token_id: int | None = None, text: str | None = None) -> None:
        """Record a token immediately in a custom generation loop."""
        self._sync()
        self.tokens.append(
            TokenEvent(
                token_index=len(self.tokens),
                step=self._step,
                token_id=token_id,
                text=text,
                emitted_ns=time.perf_counter_ns() - self._origin_ns,
            )
        )

    def trace(self, **metadata: Any) -> InferenceTrace:
        info = environment_metadata()
        info.update(
            {
                "model_class": type(self.model).__name__,
                "selected_modules": len(self._selected_names),
                "module_pattern": self.config.module_pattern,
                "leaf_modules": self.config.leaf_modules,
                "device_synchronization": self.config.synchronize_device,
                "collector_capabilities": {
                    "layer_timing": "exact",
                    "kv_cache_usage": "unavailable",
                    "scheduler_state": "unavailable",
                    "kernel_timing": "unavailable",
                },
            }
        )
        info.update(metadata)
        return InferenceTrace(
            metadata=info,
            events=list(self.events),
            steps=list(self.steps),
            tokens=list(self.tokens),
            metrics={},
            errors=list(self.errors),
        )

    def _selected_modules(self) -> Iterable[tuple[str, Any]]:
        pattern = re.compile(self.config.module_pattern)
        for name, module in self.model.named_modules():
            if not name:
                continue
            is_leaf = not any(True for _ in module.children())
            if pattern.search(name) or (self.config.leaf_modules and is_leaf):
                yield name, module

    def _root_pre_hook(self, _module: Any, args: Any, kwargs: dict[str, Any]) -> None:
        if self._automatic_steps:
            self.begin_step(self._step + 1)
        elif self._step < 0:
            self.begin_step(0)
        self._sync()
        started_ns = time.perf_counter_ns()
        self._step_starts.append((started_ns, self._input_token_count(args, kwargs)))

    def _root_post_hook(
        self, _module: Any, args: Any, kwargs: dict[str, Any], output: Any
    ) -> None:
        self._sync()
        ended_ns = time.perf_counter_ns()
        if not self._step_starts:
            self.errors.append(f"Missing forward start timestamp for step {self._step}")
            return
        started_ns, input_tokens = self._step_starts.pop()
        self.steps.append(
            StepEvent(
                step=self._step,
                phase=self._phase,
                token_index=self._token_index,
                started_ns=started_ns - self._origin_ns,
                duration_ns=ended_ns - started_ns,
                input_tokens=input_tokens,
            )
        )

    def _layer_pre_hook(self, module_id: int) -> Callable[..., None]:
        def hook(_module: Any, inputs: Any) -> None:
            self._sync()
            memory = self._memory_allocated(inputs) if self.config.record_memory else None
            self._layer_starts[module_id].append((time.perf_counter_ns(), memory))

        return hook

    def _layer_post_hook(self, name: str, module_id: int) -> Callable[..., None]:
        def hook(module: Any, inputs: Any, output: Any) -> None:
            self._sync()
            ended_ns = time.perf_counter_ns()
            stack = self._layer_starts[module_id]
            if not stack:
                self.errors.append(f"Missing layer start timestamp for {name}")
                return
            started_ns, memory_before = stack.pop()
            self.events.append(
                LayerEvent(
                    sequence=self._sequence,
                    step=self._step,
                    phase=self._phase,
                    token_index=self._token_index,
                    module=name,
                    module_type=type(module).__name__,
                    depth=name.count(".") + 1,
                    started_ns=started_ns - self._origin_ns,
                    duration_ns=ended_ns - started_ns,
                    input_shapes=_shapes(inputs) if self.config.record_shapes else [],
                    output_shapes=_shapes(output) if self.config.record_shapes else [],
                    device=self._device(inputs, output),
                    memory_before_bytes=memory_before,
                    memory_after_bytes=(
                        self._memory_allocated(output) if self.config.record_memory else None
                    ),
                )
            )
            self._sequence += 1

        return hook

    def _sync(self) -> None:
        if not self.config.synchronize_device:
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.synchronize()
        except (ImportError, RuntimeError):
            pass

    @staticmethod
    def _device(*values: Any) -> str:
        for value in values:
            for tensor in _walk_tensors(value):
                return str(tensor.device)
        return "unknown"

    @staticmethod
    def _memory_allocated(value: Any) -> int | None:
        device = LayerProfiler._device(value)
        try:
            import torch

            if device.startswith("cuda"):
                return int(torch.cuda.memory_allocated(device))
            if device.startswith("mps") and hasattr(torch.mps, "current_allocated_memory"):
                return int(torch.mps.current_allocated_memory())
        except (ImportError, RuntimeError):
            pass
        return None

    @staticmethod
    def _input_token_count(args: Any, kwargs: dict[str, Any]) -> int | None:
        candidates = [kwargs.get("input_ids"), kwargs.get("inputs_embeds")]
        if args:
            candidates.append(args[0])
        for tensor in candidates:
            if tensor is None or not hasattr(tensor, "shape"):
                continue
            shape = tuple(tensor.shape)
            if len(shape) >= 2:
                return int(shape[-2] if tensor is kwargs.get("inputs_embeds") else shape[-1])
        return None
