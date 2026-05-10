# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# 版权所有  @丁师兄大模型
# --------------------------------------------

"""Evaluate filled Golden data with DeepEval LLM-based RAG metrics."""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.models.llms.openai_model import GPTModel
from deepeval.test_case import LLMTestCase
from tqdm import tqdm


DEFAULT_CONFIG_PATH = "config.ini"
DEFAULT_INPUT_PATH = "data/golden/goldens_final_eval_filled.jsonl"
DEFAULT_OUTPUT_PATH = "data/golden/goldens_final_eval_deepeval_scores.jsonl"
DEFAULT_SUMMARY_PATH = "data/golden/goldens_final_eval_deepeval_summary.json"


def load_export_env(config_path: Path) -> Dict[str, str]:
    env = {}
    export_pattern = re.compile(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*?)(?:\s+#.*)?$")
    with config_path.open(encoding="utf-8") as f:
        for line in f:
            match = export_pattern.match(line)
            if not match:
                continue
            key, value = match.groups()
            value = value.strip().strip('"').strip("'")
            env[key] = value
    return env


def build_judge_model(config_path: Path) -> GPTModel:
    env = load_export_env(config_path)
    api_key = env.get("DOUBAO_API_KEY") or os.getenv("DOUBAO_API_KEY")
    base_url = env.get("DOUBAO_BASE_URL") or os.getenv("DOUBAO_BASE_URL")
    model_name = env.get("DOUBAO_MODEL_NAME") or os.getenv("DOUBAO_MODEL_NAME")

    missing = [
        name
        for name, value in {
            "DOUBAO_API_KEY": api_key,
            "DOUBAO_BASE_URL": base_url,
            "DOUBAO_MODEL_NAME": model_name,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing judge model config: {', '.join(missing)}")

    return GPTModel(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        cost_per_input_token=0,
        cost_per_output_token=0,
        generation_kwargs={
            "max_tokens": 2048,
            "response_format": {"type": "json_object"},
        },
    )


def build_metrics(model: GPTModel):
    return [
        AnswerRelevancyMetric(threshold=0.5, include_reason=True, async_mode=False, model=model),
        FaithfulnessMetric(threshold=0.5, include_reason=True, async_mode=False, model=model),
        ContextualRecallMetric(threshold=0.6, include_reason=True, async_mode=False, model=model),
    ]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_no} must be a JSON object.")
            records.append(record)
    return records


def to_test_case(record: Dict[str, Any]) -> LLMTestCase:
    return LLMTestCase(
        input=record["input"],
        actual_output=record["actual_output"],
        expected_output=record["expected_output"],
        retrieval_context=record["retrieval_context"],
    )


def evaluate_record(record: Dict[str, Any], metrics) -> Dict[str, Any]:
    test_case = to_test_case(record)
    metric_results = {}

    for metric in metrics:
        score = metric.measure(test_case)
        metric_results[metric.__class__.__name__] = {
            "score": score,
            "success": metric.is_successful(),
            "reason": metric.reason,
        }

    scores = [result["score"] for result in metric_results.values()]
    return {
        "id": record.get("id"),
        "input": record["input"],
        "metrics": metric_results,
        "overall_score": sum(scores) / len(scores),
        "success": all(result["success"] for result in metric_results.values()),
    }


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary(path: Path, records: List[Dict[str, Any]]) -> None:
    metric_names = list(records[0]["metrics"]) if records else []
    metric_averages = {}
    for metric_name in metric_names:
        scores = [record["metrics"][metric_name]["score"] for record in records]
        metric_averages[metric_name] = sum(scores) / len(scores)

    summary = {
        "count": len(records),
        "success_count": sum(1 for record in records if record["success"]),
        "overall_score": sum(record["overall_score"] for record in records) / len(records),
        "metric_averages": metric_averages,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Golden data with DeepEval LLM metrics.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="config.ini path with DOUBAO exports.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Filled Golden JSONL path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output metric JSONL path.")
    parser.add_argument("--summary", default=DEFAULT_SUMMARY_PATH, help="Output summary JSON path.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N records.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(Path(args.input))
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError("No records selected for evaluation.")

    model = build_judge_model(Path(args.config))
    metrics = build_metrics(model)

    results = []
    for record in tqdm(records, desc="DeepEval"):
        results.append(evaluate_record(record, metrics))

    write_jsonl(Path(args.output), results)
    write_summary(Path(args.summary), results)
    print(f"Evaluation results saved to: {args.output}")
    print(f"Evaluation summary saved to: {args.summary}")


if __name__ == "__main__":
    main()
