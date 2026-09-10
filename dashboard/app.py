"""Streamlit application for exploring offline layer-profiler traces."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACE_DIR = PROJECT_ROOT / "traces"

st.set_page_config(page_title="LayerLens", page_icon="🔬", layout="wide")
st.title("LayerLens")
st.caption("Every layer. Every token. Every millisecond.")


@st.cache_data
def load_path(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def frame(payload: dict, key: str) -> pd.DataFrame:
    return pd.DataFrame(payload.get(key, []))


available = sorted(
    TRACE_DIR.glob("*.json"), key=lambda candidate: candidate.stat().st_mtime, reverse=True
)
uploaded = st.sidebar.file_uploader("Upload trace JSON", type=["json"])

if uploaded is not None:
    payload = json.load(uploaded)
    source = uploaded.name
elif available:
    selected = st.sidebar.selectbox("Saved trace", available, format_func=lambda path: path.name)
    payload = load_path(str(selected))
    source = selected.name
else:
    st.info("No traces yet. Run `poetry run llm-layer-profile ...` or upload a trace JSON file.")
    st.stop()

events = frame(payload, "events")
steps = frame(payload, "steps")
tokens = frame(payload, "tokens")
metadata = payload.get("metadata", {})
server_metrics = payload.get("metrics", {})

model_name = metadata.get("model", metadata.get("model_class", "Unknown model"))
st.subheader(str(model_name))
if metadata.get("prompt"):
    st.caption(f"Prompt: {metadata['prompt']}")

st.sidebar.caption(f"Source: {source}")
if server_metrics.get("available"):
    cache = server_metrics.get("cache", {})
    counters = server_metrics.get("counters", {})
    latency = server_metrics.get("latency_ms", {})
    if server_metrics.get("source") == "ollama_api":
        st.subheader("Ollama runtime telemetry")
        throughput = server_metrics.get("throughput", {})
        runtime = server_metrics.get("runtime", {})
        context_use = cache.get("estimated_context_utilization")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Estimated context use", f"{100 * context_use:.2f}%" if context_use is not None else "N/A")
        m2.metric("Prompt throughput", f"{throughput.get('prompt_tokens_per_second', 0):.2f} tok/s")
        m3.metric("Generation throughput", f"{throughput.get('generation_tokens_per_second', 0):.2f} tok/s")
        m4.metric("Model load", f"{latency.get('load', 0):,.2f} ms")
        m5.metric("Processor memory", f"{runtime.get('processor_memory_mb', 0):,.0f} MB")
        phases = pd.DataFrame(
            {
                "phase": ["Model load", "Prompt evaluation", "Generation", "Runtime overhead"],
                "duration_ms": [
                    latency.get("load", 0),
                    latency.get("prompt_eval", 0),
                    latency.get("generation", 0),
                    latency.get("runtime_overhead", 0),
                ],
            }
        )
        st.plotly_chart(
            px.bar(phases, x="phase", y="duration_ms", color="phase", title="Ollama runtime phases"),
            use_container_width=True,
        )
        st.caption("Ollama does not expose block-level KV occupancy; context utilization is an estimate.")
        with st.expander("Ollama metric details"):
            st.json(server_metrics)
    else:
        st.subheader("vLLM engine & KV cache")
        scheduler = server_metrics.get("scheduler", {})
        process_metrics = server_metrics.get("process", {})
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Peak KV-cache usage", f"{100 * cache.get('peak_usage', 0):.2f}%")
        hit_rate = cache.get("prefix_hit_rate")
        m2.metric("Prefix-cache hit rate", f"{100 * hit_rate:.1f}%" if hit_rate is not None else "N/A")
        m3.metric("Peak running requests", f"{scheduler.get('peak_running', 0):g}")
        ttft = latency.get("time_to_first_token_mean_ms")
        m4.metric("vLLM TTFT", f"{ttft:.2f} ms" if ttft is not None else "N/A")
        m5.metric("Peak server RSS", f"{process_metrics.get('peak_resident_memory_mb', 0):,.0f} MB")

        metric_timeline = pd.DataFrame(server_metrics.get("timeline", []))
        if not metric_timeline.empty:
            engine_chart = go.Figure()
            engine_chart.add_trace(
                go.Scatter(x=metric_timeline["elapsed_ms"], y=metric_timeline["kv_cache_usage"] * 100, name="KV-cache usage", mode="lines", fill="tozeroy")
            )
            engine_chart.add_trace(
                go.Scatter(x=metric_timeline["elapsed_ms"], y=metric_timeline["requests_running"], name="Running requests", mode="lines", yaxis="y2")
            )
            engine_chart.update_layout(
                title="KV-cache pressure and scheduler activity during this request",
                xaxis_title="Request elapsed time (ms)",
                yaxis_title="KV-cache usage (%)",
                yaxis2=dict(title="Requests", overlaying="y", side="right", rangemode="tozero"),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(engine_chart, use_container_width=True)
        with st.expander("Engine metric details"):
            st.json({"cache": cache, "scheduler": scheduler, "counters": counters, "latency_ms": latency})
elif server_metrics:
    st.info("This trace requested engine metrics, but the vLLM metrics endpoint was unavailable.")

if events.empty:
    if server_metrics.get("available"):
        source_kind = server_metrics.get("source")
        if source_kind == "ollama_api":
            st.info("Ollama trace: native API metrics are available; layer boundaries are not exposed by Ollama.")
        else:
            st.info("Metrics-only trace: add the optional injection shim to capture the layer timeline.")
    else:
        st.warning("This trace has no layer events. Check the module selection pattern shown in metadata.")
        st.json(metadata)
    st.stop()

phase_options = events["phase"].dropna().unique().tolist()
phases = st.sidebar.multiselect("Phase", phase_options, default=phase_options)
filtered = events[events["phase"].isin(phases)].copy()

step_options = sorted(int(value) for value in filtered["step"].unique())
selected_steps = st.sidebar.multiselect("Forward steps", step_options, default=step_options)
filtered = filtered[filtered["step"].isin(selected_steps)].copy()

if filtered.empty:
    st.warning("The current filters exclude every layer event.")
    st.stop()

prefill = steps[steps["phase"] == "prefill"] if not steps.empty else pd.DataFrame()
decode = steps[steps["phase"] == "decode"] if not steps.empty else pd.DataFrame()
prefill_ms = prefill["duration_ms"].sum() if not prefill.empty else float("nan")
mean_decode_ms = decode["duration_ms"].mean() if not decode.empty else float("nan")
slowest = filtered.loc[filtered["duration_ms"].idxmax()]

c1, c2, c3, c4 = st.columns(4)
c1.metric("First token · prefill", f"{prefill_ms:,.2f} ms" if pd.notna(prefill_ms) else "N/A")
c2.metric("Later tokens · mean", f"{mean_decode_ms:,.2f} ms" if pd.notna(mean_decode_ms) else "N/A")
c3.metric("Recorded layer calls", f"{len(filtered):,}")
c4.metric("Slowest layer call", f"{slowest['duration_ms']:,.2f} ms")

overview, token_tab, bottleneck_tab, details_tab = st.tabs(
    ["Layer heatmap", "Token timeline", "Bottlenecks", "Trace details"]
)

with overview:
    filtered["layer_order"] = pd.to_numeric(
        filtered["module"].str.extract(r"(\d+)$", expand=False), errors="coerce"
    )
    module_order = (
        filtered[["module", "layer_order"]]
        .drop_duplicates()
        .sort_values(["layer_order", "module"], na_position="last")["module"]
        .tolist()
    )
    heat = filtered.pivot_table(
        index="module", columns="step", values="duration_ms", aggfunc="sum", fill_value=0
    ).reindex(module_order)
    heatmap = px.imshow(
        heat,
        aspect="auto",
        color_continuous_scale="Viridis",
        labels={
            "x": "Forward step (0 = prefill / first token)",
            "y": "Model layer",
            "color": "Duration (ms)",
        },
        title="Layer duration by generation step",
    )
    heatmap.update_layout(height=max(500, min(1200, len(heat.index) * 25)))
    st.plotly_chart(heatmap, use_container_width=True)

    selected_step = st.select_slider(
        "Inspect one forward step", options=step_options, value=step_options[0]
    )
    one_step = filtered[filtered["step"] == selected_step].sort_values("sequence")
    bars = px.bar(
        one_step,
        x="module",
        y="duration_ms",
        color="duration_ms",
        labels={"module": "Layer execution order", "duration_ms": "Duration (ms)"},
        title=f"Step {selected_step}: sequential layer cost",
    )
    bars.update_layout(xaxis_tickangle=-55, showlegend=False)
    st.plotly_chart(bars, use_container_width=True)

with token_tab:
    if steps.empty:
        st.info("This trace predates whole-step timing support.")
    else:
        timeline = steps.copy()
        timeline["label"] = timeline.apply(
            lambda row: "First token (prefill)" if row["step"] == 0 else f"Token {int(row['step']) + 1}",
            axis=1,
        )
        step_chart = px.bar(
            timeline,
            x="step",
            y="duration_ms",
            color="phase",
            hover_data=["input_tokens", "token_index"],
            labels={"step": "Forward step", "duration_ms": "Full forward duration (ms)"},
            title="Time to first token and subsequent decode-token cost",
        )
        st.plotly_chart(step_chart, use_container_width=True)

        layer_totals = filtered.groupby(["step", "module"], as_index=False)["duration_ms"].sum()
        layer_area = px.area(
            layer_totals,
            x="step",
            y="duration_ms",
            color="module",
            labels={"step": "Forward step", "duration_ms": "Layer duration (ms)"},
            title="Layer contribution across generated tokens",
        )
        st.plotly_chart(layer_area, use_container_width=True)

    if not tokens.empty:
        display_tokens = tokens.copy()
        if "emitted_ms" in display_tokens:
            display_tokens["inter_token_ms"] = display_tokens["emitted_ms"].diff()
        st.dataframe(display_tokens, use_container_width=True, hide_index=True)

with bottleneck_tab:
    aggregate = (
        filtered.groupby(["module", "module_type"], as_index=False)
        .agg(
            total_ms=("duration_ms", "sum"),
            mean_ms=("duration_ms", "mean"),
            p95_ms=("duration_ms", lambda values: values.quantile(0.95)),
            max_ms=("duration_ms", "max"),
            calls=("duration_ms", "size"),
        )
        .sort_values("total_ms", ascending=False)
    )
    top = aggregate.head(30).sort_values("total_ms")
    ranking = go.Figure(
        go.Bar(x=top["total_ms"], y=top["module"], orientation="h")
    )
    ranking.update_layout(
        title="Top layers by cumulative time",
        xaxis_title="Total duration (ms)",
        yaxis_title="Layer",
        height=max(500, len(top) * 24),
    )
    st.plotly_chart(ranking, use_container_width=True)
    st.dataframe(aggregate, use_container_width=True, hide_index=True)

with details_tab:
    st.subheader("Run metadata")
    st.json(metadata)
    errors = payload.get("errors", [])
    if errors:
        st.warning("\n".join(errors))
    st.subheader("Raw layer events")
    st.dataframe(filtered.sort_values("sequence"), use_container_width=True, hide_index=True)
