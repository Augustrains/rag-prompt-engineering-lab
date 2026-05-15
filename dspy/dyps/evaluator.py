# -*- coding: utf-8 -*-
"""
DSPy DYPS 评测模块
==================

基于 DeepEval 框架的 RAG 答案质量评测。

核心组件：
  1. DeepEvalJudge — 将 API 模型包装成 DeepEval 兼容的 Judge 接口
  2. EvalResult — 单条评测结果的数据类
  3. Evaluator — 异步批量评测器，运行三维指标并计算综合分

三维评测指标（DeepEval 内置）：

  ┌─────────────────────┬────────┬────────────────────────────────┐
  │ 指标                 │ 权重   │ 评估内容                        │
  ├─────────────────────┼────────┼────────────────────────────────┤
  │ AnswerRelevancy     │ 0.3    │ 答案是否切题回应了用户问题        │
  │ Faithfulness        │ 0.4    │ 答案是否忠实于检索上下文，无幻觉   │
  │ ContextualRecall    │ 0.3    │ 答案是否覆盖了标准答案的关键信息   │
  └─────────────────────┴────────┴────────────────────────────────┘

Faithfulness 权重最高（0.4），因为幻觉是 RAG 最致命的失败模式。

已知限制：
  - AnswerRelevancy 对"模型正确输出无答案"场景存在评分盲点（给 0 分）
  - 这是 DeepEval 指标设计问题，不反映提示词质量
  - ContextualRecall 依赖 expected_output 的质量

异步评测设计：
  - 使用 asyncio.Semaphore 控制并发数（默认 1，避免 API 限流和非法 JSON）
  - 每个 metric 最多重试 3 次，失败后降级为 score=0.0（不中断整个优化）
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


# ═══════════════════════════════════════════════════════════════════════════════
# DeepEval Judge 模型包装器
# ═══════════════════════════════════════════════════════════════════════════════

class DeepEvalJudge(DeepEvalBaseLLM):
    """
    将 API 模型包装为 DeepEval 兼容的 Judge 接口。

    DeepEval 的 metric 需要调用 LLM 来做评判（LLM-as-a-Judge）。
    这个类让 DeepEval 的 AnswerRelevancyMetric / FaithfulnessMetric /
    ContextualRecallMetric 可以通过我们的 API 模型来执行评测。

    核心方法：
      - generate(): 同步调用（DeepEval 某些操作需要）
      - a_generate(): 异步调用（评测主要使用）
      - load_model(): DeepEval 框架要求实现，返回底层客户端
      - get_model_name(): 返回模型标识

    注意：属性初始化必须在 super().__init__() 之前，
    因为 DeepEvalBaseLLM.__init__ 会立即调用 self.load_model()。
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
        """
        同步生成（供 DeepEval metric 调用）。

        DeepEval 传入评测 prompt，我们调用 API 获取评判结果。
        使用 json_object 模式确保返回合法 JSON（减少解析失败）。
        """
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
        """
        异步生成（评测主要使用此方法）。

        与 generate() 逻辑相同，但使用 AsyncOpenAI 客户端，
        支持 asyncio 并发，大幅提升评测速度。
        """
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
        """DeepEval 框架要求的模型加载方法"""
        if async_mode:
            return AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        return OpenAI(api_key=self._api_key, base_url=self._base_url)

    def get_model_name(self) -> str:
        """返回模型名（DeepEval 框架要求）"""
        return self.model_name


# ═══════════════════════════════════════════════════════════════════════════════
# 评测结果数据类
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    """
    单条评测的完整结果。

    字段说明：
      - unique_id: 样本唯一标识
      - category: 问题类别（如"充电相关"）
      - input: 用户问题
      - expected_output: 标准答案
      - actual_output: 模型实际输出
      - retrieval_context: 检索到的上下文文档列表
      - answer_relevancy: 答案相关性分数 (0-1)
      - faithfulness: 忠诚度/无幻觉分数 (0-1)
      - contextual_recall: 上下文召回分数 (0-1)
      - overall_score: 加权综合分 (0-1)
      - success: 所有指标是否都通过阈值
      - reason: 评测理由（各指标 reason 的合并，截断至 500 字符）
    """
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
        """序列化为字典（用于保存 JSONL）"""
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


# ═══════════════════════════════════════════════════════════════════════════════
# 核心评测器
# ═══════════════════════════════════════════════════════════════════════════════

class Evaluator:
    """
    DeepEval 异步评测器。

    对一批 RAG 答案运行三维指标评测，返回 EvalResult 列表。

    并发控制：
      - 使用 asyncio.Semaphore 限制同时进行的 API 调用数
      - 默认 max_concurrency=1（保守，避免 API 返回非法 JSON）
      - 可以调高来提高速度，但可能增加解析失败概率

    容错机制：
      - 每个 metric 最多重试 3 次（MAX_RETRIES）
      - 重试间隔递增：第 1 次 2s，第 2 次 4s
      - 所有重试都失败后降级为 score=0.0（而非抛异常中断评测）
      - 这样即使个别样本失败，整体优化可以继续

    使用示例：
        judge = build_judge()
        evaluator = Evaluator(judge, max_concurrency=1)
        results = await evaluator.evaluate(records, desc="评测中...")
        evaluator.print_summary(results)
    """

    MAX_CONCURRENCY = 1   # 默认最大并发数
    MAX_RETRIES = 3       # 每个 metric 的最大重试次数

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
        """
        构建 DeepEval 三维指标列表。

        三个指标都配置为：
          - async_mode=True：使用 a_measure() 异步评测
          - include_reason=True：记录评分理由（方便调试和生成 hints）
          - model=self.judge：使用教师模型作为评判员
        """
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
        """
        对单条记录进行三维评测（异步）。

        Args:
            record: 包含 input / actual_output / expected_output / retrieval_context 的字典
            semaphore: 并发控制信号量

        Returns:
            EvalResult 对象

        评测流程：
          1. 构建 LLMTestCase（DeepEval 的数据格式）
          2. 对三个 metric 分别调用 a_measure()（异步并行）
          3. 每个 metric 最多重试 MAX_RETRIES 次，指数退避
          4. 失败降级为 score=0.0（而非抛异常）
          5. 计算加权综合分
        """
        async with semaphore:
            # 构建 DeepEval 测试用例
            test_case = LLMTestCase(
                input=record["input"],
                actual_output=record["actual_output"],
                expected_output=record.get("expected_output", ""),
                retrieval_context=record.get("retrieval_context", []),
            )

            metrics = self._build_metrics()
            metric_results: Dict[str, Dict[str, Any]] = {}

            # 对每个 metric 独立评测
            for metric in metrics:
                metric_name = metric.__class__.__name__
                last_error = None

                # 重试循环（指数退避：第1次等2s，第2次等4s）
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

                # 所有重试都失败 → 降级处理，返回 0 分
                if metric_name not in metric_results:
                    print(f"[WARN] Metric {metric_name} failed after {self.MAX_RETRIES} attempts: {last_error}")
                    metric_results[metric_name] = {
                        "score": 0.0,
                        "success": False,
                        "reason": str(last_error),
                        "attempts": self.MAX_RETRIES,
                    }

            # 提取三维分数（从 metric_results 字典中按类名查找）
            scores = {
                "answer_relevancy": metric_results.get("AnswerRelevancyMetric", {}).get("score", 0.0),
                "faithfulness": metric_results.get("FaithfulnessMetric", {}).get("score", 0.0),
                "contextual_recall": metric_results.get("ContextualRecallMetric", {}).get("score", 0.0),
            }

            # 加权综合分：faithfulness 权重最高（幻觉是最严重的问题）
            weights = [0.3, 0.4, 0.3]  # relevancy, faithfulness, recall
            overall = sum(s * w for s, w in zip(scores.values(), weights))
            success = all(r["success"] for r in metric_results.values())

            # 合并各指标的 reason（方便后续分析问题）
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
                reason=combined_reason[:500],  # 截断，防止 JSONL 行过长
            )

    async def evaluate(
        self,
        records: List[Dict[str, Any]],
        desc: str = "DeepEval",
    ) -> List[EvalResult]:
        """
        异步批量评测。

        Args:
            records: 评测记录列表，每条需包含:
                - input: 用户问题
                - actual_output: 模型实际输出
                - expected_output: 标准答案
                - retrieval_context: 检索上下文
                - unique_id / category: 元数据（可选）
            desc: 进度条描述文字

        Returns:
            EvalResult 列表，顺序与输入一致
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
        """
        同步包装器。

        内部调用 asyncio.run()，方便在非异步上下文中使用。
        如果在已有 event loop 中运行，请使用 evaluate()。
        """
        return asyncio.run(self.evaluate(records, desc))

    @staticmethod
    def print_summary(results: List[EvalResult]) -> Dict[str, float]:
        """
        打印评测汇总并返回汇总字典。

        输出内容：
          - 评测总数和成功率
          - 各指标的平均分
          - 综合平均分
        """
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


# ═══════════════════════════════════════════════════════════════════════════════
# 便捷构建函数
# ═══════════════════════════════════════════════════════════════════════════════

def load_env_config(config_path: Path) -> Dict[str, str]:
    """
    从 config.ini 加载环境变量配置。

    config.ini 格式示例：
        export DOUBAO_API_KEY=sk-xxx
        export DOUBAO_BASE_URL=https://api.example.com
        export DOUBAO_MODEL_NAME=deepseek-v4-flash

    解析规则：
      - 匹配 "export KEY=VALUE" 模式
      - VALUE 支持双引号和单引号包裹
      - 忽略注释（# 后的内容）
    """
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
    """
    从 config.ini 构建 DeepEvalJudge。

    自动读取 API 配置，创建可用的 Judge 实例。
    如果缺少 DOUBAO_API_KEY 则抛出 ValueError。
    """
    env = load_env_config(config_path)

    model = env.get("DOUBAO_MODEL_NAME", "deepseek-v4-flash")
    api_key = env.get("DOUBAO_API_KEY", "")
    base_url = env.get("DOUBAO_BASE_URL", "")

    if not api_key:
        raise ValueError("DOUBAO_API_KEY not found in config.ini")

    return DeepEvalJudge(model=model, api_key=api_key, base_url=base_url)
