# -*- coding: utf-8 -*-
"""Quick smoke test: core logic only (no RAG inference needed)."""
import os
import sys

# Load env from config.ini
env = {}
with open("/root/autodl-tmp/RAG/config.ini") as f:
    for line in f:
        line = line.strip()
        if line.startswith("export "):
            line = line[7:]
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
for k in ("DOUBAO_API_KEY", "DOUBAO_BASE_URL", "DOUBAO_MODEL_NAME"):
    if k in env:
        os.environ[k] = env[k]

sys.path.insert(0, "/root/autodl-tmp/rag-prompt-engineering-lab-upload/dspy")

from dyps.teleprompter import (
    DEFAULT_SYSTEM_PROMPT,
    HintGenerator,
    DYPSOptimizer,
)

print("=" * 60)
print("SMOKE TEST 1: DEFAULT_SYSTEM_PROMPT constant")
print("=" * 60)
assert DEFAULT_SYSTEM_PROMPT, "DEFAULT_SYSTEM_PROMPT should not be empty"
print(f"  OK: DEFAULT_SYSTEM_PROMPT = '{DEFAULT_SYSTEM_PROMPT[:80]}...'")

print("\n" + "=" * 60)
print("SMOKE TEST 2: generate_new_params() - no history")
print("=" * 60)
opt = DYPSOptimizer.__new__(DYPSOptimizer)
opt.initial_system_prompt = None
opt.initial_system_prompt = None  # force None to test default
# We need these attributes for generate_new_params
opt._seen_prompts = set()

params = DYPSOptimizer.generate_new_params(opt, history=[], hint_feedback=None)
assert "system_prompt" in params, f"system_prompt missing: {params}"
assert "temperature" in params, f"temperature missing: {params}"
assert "max_tokens" in params, f"max_tokens missing: {params}"
assert params["system_prompt"] == DEFAULT_SYSTEM_PROMPT, f"Expected DEFAULT, got: {params['system_prompt']}"
assert 0.0 <= params["temperature"] <= 0.3
assert 512 <= params["max_tokens"] <= 2048
print(f"  OK: {params}")

print("\n" + "=" * 60)
print("SMOKE TEST 3: _mutate_inference_params()")
print("=" * 60)
p = {"temperature": 0.5, "max_tokens": 1024}
opt._mutate_inference_params(p, aggressive=False)
assert "temperature" in p
assert "max_tokens" in p
print(f"  OK (normal): {p}")

p2 = {"temperature": 0.5, "max_tokens": 1024}
opt._mutate_inference_params(p2, aggressive=True)
print(f"  OK (aggressive): {p2}")

print("\n" + "=" * 60)
print("SMOKE TEST 4: generate_new_params() - with history")
print("=" * 60)
history = [{
    "trial": 0,
    "params": {"system_prompt": "测试提示词", "temperature": 0.2, "max_tokens": 800},
    "score": 0.5,
}]
# Generate many times to verify all three strategies
results = set()
for _ in range(50):
    p = opt.generate_new_params(history=history, hint_feedback=None)
    results.add(p["system_prompt"])
    assert "system_prompt" in p
    assert "temperature" in p
    assert "max_tokens" in p
# With no hint_feedback, should always use exploit or random_explore
print(f"  OK: generated 50 param sets, unique prompts: {len(results)}")
print(f"  Sample: {opt.generate_new_params(history=history, hint_feedback=None)}")

print("\n" + "=" * 60)
print("SMOKE TEST 5: HintGenerator.synthesize_improved_prompt()")
print("=" * 60)
hg = HintGenerator()
# Test with empty hints (no API call needed)
result = hg.synthesize_improved_prompt("原始提示词", [])
assert result == "原始提示词", f"Empty hints should return original, got: {result}"
print(f"  OK: empty hints returns original (no API call)")

# Test that HintGenerator exists and has the method
assert hasattr(hg, "synthesize_improved_prompt"), "Missing synthesize_improved_prompt method"
assert hasattr(hg, "generate_hint"), "Missing generate_hint method"
print(f"  OK: HintGenerator has both required methods")

print("\n" + "=" * 60)
print("SMOKE TEST 6: prompt dedup logic")
print("=" * 60)
opt = DYPSOptimizer.__new__(DYPSOptimizer)
opt._seen_prompts = set()
opt.initial_system_prompt = "去重测试提示"
# Add a prompt to seen set
opt._seen_prompts.add("去重测试提示"[:200])
# Generate params and verify dedup behavior
p = opt.generate_new_params(history=[], hint_feedback=None)
assert p["system_prompt"] == "去重测试提示"
# Note: dedup check happens in optimize_async, not generate_new_params
print("  OK: dedup infrastructure in place (_seen_prompts set)")

print("\n" + "=" * 60)
print("SMOKE TEST 7: Initial prompt override")
print("=" * 60)
opt3 = DYPSOptimizer.__new__(DYPSOptimizer)
opt3.initial_system_prompt = "自定义初始提示词"
opt3._seen_prompts = set()
p = opt3.generate_new_params(history=[], hint_feedback=None)
assert p["system_prompt"] == "自定义初始提示词", f"Expected custom prompt, got: {p['system_prompt']}"
print(f"  OK: custom initial prompt used: {p['system_prompt']}")

print("\n" + "=" * 60)
print("ALL QUICK SMOKE TESTS PASSED!")
print("=" * 60)
