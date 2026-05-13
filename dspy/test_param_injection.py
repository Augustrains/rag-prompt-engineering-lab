# -*- coding: utf-8 -*-
"""Minimal test: verify prompt_params flow through to vLLM and produce different answers."""
import os
import sys

sys.path.insert(0, "/root/autodl-tmp/RAG")
sys.path.insert(0, "/root/autodl-tmp/rag-prompt-engineering-lab-upload/dspy")

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

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import dspy
from dyps.models import StudentRAG

# Test queries (different categories to get diverse retrieval)
test_queries = [
    "充电接口被冻住了怎么办？",
    "如何设置离车自动上锁功能？",
]

print("=" * 60)
print("Creating StudentRAG...")
student = StudentRAG(config={"bm25_topk": 5, "milvus_topk": 10, "rerank_topk": 5})
print("StudentRAG created (MongoDB + vLLM connected)")

# ── Test 1: Default params (baseline) ──
print("\n" + "=" * 60)
print("TEST 1: Default params (no prompt_params)")
p1 = {}
pred1 = student(input=test_queries[0], prompt_params=p1)
print(f"  Answer ({len(pred1.answer)} chars): {pred1.answer[:200]}...")
print(f"  Context count: {len(pred1.retrieval_context)}")

# ── Test 2: Custom system_prompt + high temperature ──
print("\n" + "=" * 60)
print("TEST 2: custom system_prompt, temperature=0.8")
p2 = {
    "temperature": 0.8,
    "max_tokens": 512,
    "system_prompt": "你是一个严谨的汽车技术专家。请先逐步推理，再给出精准答案。如果不确定，请如实说明。",
}
pred2 = student(input=test_queries[0], prompt_params=p2)
print(f"  Answer ({len(pred2.answer)} chars): {pred2.answer[:200]}...")
print(f"  Context count: {len(pred2.retrieval_context)}")

# ── Test 3: Different system_prompt, colder + shorter ──
print("\n" + "=" * 60)
print("TEST 3: short system_prompt, temperature=0.0, max_tokens=128")
p3 = {
    "temperature": 0.0,
    "max_tokens": 128,
    "system_prompt": "用一句话简洁回答问题。",
}
pred3 = student(input=test_queries[0], prompt_params=p3)
print(f"  Answer ({len(pred3.answer)} chars): {pred3.answer[:200]}...")
print(f"  Context count: {len(pred3.retrieval_context)}")

# ── Summary ──
print("\n" + "=" * 60)
print("RESULTS COMPARISON (same query)")
print(f"  Test 1 (default):     {len(pred1.answer)} chars")
print(f"  Test 2 (custom+cot):  {len(pred2.answer)} chars")
print(f"  Test 3 (short+cold):  {len(pred3.answer)} chars")

# Check if answers are actually different
answers_differ = (pred1.answer != pred2.answer) or (pred1.answer != pred3.answer)
print(f"\n  Answers differ between tests: {answers_differ}")
print(f"  Test1==Test2: {pred1.answer == pred2.answer}")
print(f"  Test1==Test3: {pred1.answer == pred3.answer}")
print(f"  Test2==Test3: {pred2.answer == pred3.answer}")

if answers_differ:
    print("\n  >>> SUCCESS: Parameter injection is WORKING!")
else:
    print("\n  >>> WARNING: All answers identical, params may not be reaching LLM")

print("\nDone!")
