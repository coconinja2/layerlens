"""KV-cache-aware greedy loop with explicit prefill/decode attribution."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from layer_profiler import LayerProfiler


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Explain a KV cache in one sentence."
NEW_TOKENS = 8

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).eval()
encoded = tokenizer(PROMPT, return_tensors="pt")
input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]
past_key_values = None

with torch.inference_mode(), LayerProfiler(model) as profiler:
    for step in range(NEW_TOKENS):
        current_ids = input_ids if step == 0 else input_ids[:, -1:]
        outputs = profiler.profile_forward(
            model,
            input_ids=current_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            step=step,
        )
        next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
        profiler.mark_token(int(next_token.item()), tokenizer.decode(next_token[0]))

trace_path = profiler.trace(model=MODEL_ID, prompt=PROMPT).write("traces/manual.json")
print(f"Trace written to {trace_path}")

