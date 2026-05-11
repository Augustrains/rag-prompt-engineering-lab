# -*- coding: utf-8 -*-
"""
Evaluation harness using DeepEval metrics.

Provides:
1. DeepEvalJudge - wraps API model for DeepEval metric evaluation
2. Evaluator - runs evaluation on golden dataset
3. Metrics - composite scoring with multiple dimensions
"""

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import dspy
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.test_case import LLMTestCase
from openai import AsyncOpenAI, OpenAI
from tqdm.asyncio import tqdm_asyncio


# ──────────────────────────── DeepEval Judge ────────────────────────────

class DeepEvalJudge(DeepEvalBaseLLM):
    """
    Wraps a teacher API model as a DeepEval judge model.

    This allows us to use the strong teacher model for evaluating
    student answers using DeepEval's built-in metrics.
    """

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ):
        import os
        self.model_name = model
        self._api_key = api_key or os.environ.get("DOUBAO_API_KEY", "")
        self._base_url = base_url or os.environ.get("DOUBAO_BASE_URL", "")
        self.temperature = temperature

        super().__init__(model)

    def generate(self, prompt, schema=None):
        client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a strict evaluation assistant. Output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content or ""
        if schema:
            parsed = trim_and_load_json(raw)
            return schema.model_validate(parsed), 0.0
        return raw, 0.0

    async def a_generate(self, prompt, schema=None):
        client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        completion = await client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a strict evaluation assistant. Output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content or ""
        if schema:
            parsed = trim_and_load_json(raw)
            return schema.model_validate(parsed), 0.0
        return raw, 0.0

    def load_model(self, async_mode: bool = False):
        if async_mode:
            return AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        return OpenAI(api_key=self._api_key, base_url=self._base_url)

    def get_model_name(self) -> str:
        return self.model_name


# ──────────────────────────── 评测结果 ────────────────────────────

@dataclass
class EvalResult:
    """Single evaluation result."""
    unique_id: str
    category: str
    input: str
    expected_output: str
    actual_output: str
    retrieval_context: List[str]
    answer_relevancy: float
    faithfulness: float
    contextual_recall: float
    overall_score: float
    success: bool
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unique_id": self.unique_id,
            "category": self.category,
            "input": self.input,
            "expected_output": self.expected_output,
            "actual_output": self.actual_output,
            "retrieval_context": self.retrieval_context,
            "answer_relevancy": self.answer_relevancy,
            "faithfulness": self.faithfulness,
            "contextual_recall": self.contextual_recall,
            "overall_score": self.overall_score,
            "success": self.success,
            "reason": self.reason,
        }


# ──────────────────────────── 核心评测器 ────────────────────────────

class Evaluator:
    """
    Evaluates student RAG model answers using DeepEval metrics.

    Usage:
        judge = build_judge()
        evaluator = Evaluator(judge)
        results = await evaluator.evaluate(student_answers)
    """

    MAX_CONCURRENCY = 1
    MAX_RETRIES = 3

    def __init__(
        self,
        judge: DeepEvalJudge,
        thresholds: Optional[Dict[str, float]] = None,
        max_concurrency: int = 1,
    ):
        self.judge = judge
        self.thresholds = thresholds or {
            "answer_relevancy": 0.5,
            "faithfulness": 0.5,
            "contextual_recall": 0.6,
        }
        self.max_concurrency = max_concurrency

    def _build_metrics(self):
        """Build DeepEval metrics with judge model."""
        return [
            AnswerRelevancyMetric(
                threshold=self.thresholds["answer_relevancy"],
                include_reason=True,
                async_mode=True,
                model=self.judge,
            ),
            FaithfulnessMetric(
                threshold=self.thresholds["faithfulness"],
                include_reason=True,
                async_mode=True,
                model=self.judge,
            ),
            ContextualRecallMetric(
                threshold=self.thresholds["contextual_recall"],
                include_reason=True,
                async_mode=True,
                model=self.judge,
            ),
        ]

    async def _eval_one(
        self,
        record: Dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> EvalResult:
        """Evaluate a single record."""
        async with semaphore:
            test_case = LLMTestCase(
                input=record["input"],
                actual_output=record["actual_output"],
                expected_output=record.get("expected_output", ""),
                retrieval_context=record.get("retrieval_context", []),
            )

            metrics = self._build_metrics()
            metric_results: Dict[str, Dict[str, Any]] = {}

            for metric in metrics:
                metric_name = metric.__class__.__name__
                last_error = None

                for attempt in range(1, self.MAX_RETRIES + 1):
                    try:
                        score = await metric.a_measure(test_case)
                        metric_results[metric_name] = {
                            "score": score,
                            "success": metric.is_successful(),
                            "reason": metric.reason,
                            "attempts": attempt,
                        }
                        break
                    except Exception as e:
                        last_error = e
                        if attempt < self.MAX_RETRIES:
                            await asyncio.sleep(2 * attempt)

                if metric_name not in metric_results:
                    raise RuntimeError(
                        f"Metric {metric_name} failed after {self.MAX_RETRIES} "
                        f"attempts: {last_error}"
                    )

            scores = {
                "answer_relevancy": metric_results.get("AnswerRelevancyMetric", {}).get("score", 0.0),
                "faithfulness": metric_results.get("FaithfulnessMetric", {}).get("score", 0.0),
                "contextual_recall": metric_results.get("ContextualRecallMetric", {}).get("score", 0.0),
            }

            weights = [0.3, 0.4, 0.3]  # relevancy, faithfulness, recall
            overall = sum(s * w for s, w in zip(scores.values(), weights))
            success = all(r["success"] for r in metric_results.values())

            reasons = [
                metric_results.get("AnswerRelevancyMetric", {}).get("reason", ""),
                metric_results.get("FaithfulnessMetric", {}).get("reason", ""),
                metric_results.get("ContextualRecallMetric", {}).get("reason", ""),
            ]
            combined_reason = " | ".join(filter(None, reasons))

            return EvalResult(
                unique_id=record.get("unique_id", ""),
                category=record.get("category", ""),
                input=record["input"],
                expected_output=record.get("expected_output", ""),
                actual_output=record["actual_output"],
                retrieval_context=record.get("retrieval_context", []),
                answer_relevancy=scores["answer_relevancy"],
                faithfulness=scores["faithfulness"],
                contextual_recall=scores["contextual_recall"],
                overall_score=overall,
                success=success,
                reason=combined_reason[:500],
            )

    async def evaluate(
        self,
        records: List[Dict[str, Any]],
        desc: str = "DeepEval",
    ) -> List[EvalResult]:
        """
        Evaluate a batch of records asynchronously.

        Args:
            records: List of dicts with keys: input, actual_output, expected_output, retrieval_context
            desc: Progress bar description

        Returns:
            List of EvalResult objects
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks = [self._eval_one(rec, semaphore) for rec in records]
        results = await tqdm_asyncio.gather(*tasks, desc=desc)
        return list(results)

    def evaluate_sync(
        self,
        records: List[Dict[str, Any]],
        desc: str = "DeepEval",
    ) -> List[EvalResult]:
        """Synchronous wrapper for evaluate()."""
        return asyncio.run(self.evaluate(records, desc))

    @staticmethod
    def print_summary(results: List[EvalResult]) -> Dict[str, float]:
        """Print evaluation summary and return summary dict."""
        if not results:
            print("No results to summarize.")
            return {}

        avg = {
            "overall": sum(r.overall_score for r in results) / len(results),
            "answer_relevancy": sum(r.answer_relevancy for r in results) / len(results),
            "faithfulness": sum(r.faithfulness for r in results) / len(results),
            "contextual_recall": sum(r.contextual_recall for r in results) / len(results),
        }
        success_count = sum(1 for r in results if r.success)

        print(f"\n{'=' * 50}")
        print(f"评测完成，共 {len(results)} 条")
        print(f"成功率: {success_count}/{len(results)}")
        print(f"平均综合分: {avg['overall']:.4f}")
        print(f"答案相关性: {avg['answer_relevancy']:.4f}")
        print(f"忠诚度: {avg['faithfulness']:.4f}")
        print(f"上下文召回: {avg['contextual_recall']:.4f}")
        print(f"{'=' * 50}\n")
        return avg


# ──────────────────────────── 便捷构建函数 ────────────────────────────

def load_env_config(config_path: Path) -> Dict[str, str]:
    """Load environment variables from config.ini."""
    env = {}
    pattern = re.compile(r'^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*?)(?:\s+#.*)?$')
    with config_path.open(encoding="utf-8") as f:
        for line in f:
            match = pattern.match(line)
            if match:
                key, val = match.groups()
                env[key] = val.strip().strip('"').strip("'")
    return env


def build_judge(
    config_path: Path = Path("/root/autodl-tmp/RAG/config.ini"),
) -> DeepEvalJudge:
    """Build a DeepEval judge from config.ini."""
    env = load_env_config(config_path)

    model = env.get("DOUBAO_MODEL_NAME", "deepseek-v4-flash")
    api_key = env.get("DOUBAO_API_KEY", "")
    base_url = env.get("DOUBAO_BASE_URL", "")

    if not api_key:
        raise ValueError("DOUBAO_API_KEY not found in config.ini")

    return DeepEvalJudge(model=model, api_key=api_key, base_url=base_url)
