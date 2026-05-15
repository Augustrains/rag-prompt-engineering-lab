# -*- coding: utf-8 -*-
"""
DSPy DYPS Teleprompter — 教师引导的 RAG 提示词自动优化核心
==========================================================

DYPS = Dynamic Prompt Selection：通过强大的教师模型（deepseek-v4-flash API）
指导弱学生模型（本地 Qwen3-8B + RAG pipeline）迭代优化提示词参数。

整体优化流程：

  1. 冷启动（Cold Start）
     └─ BootstrapDemonstrator: 学生检索+生成 → 教师基于学生检索上下文生成
        参考答案 → DeepEval 评分 → 构建正/负例示例池

  2. 优化循环（optimize_async），每轮 trial：
     ├─ generate_new_params() — 三种策略生成新参数
     │   ├─ 65% Exploit：保留最佳 system_prompt，小幅扰动 temperature/max_tokens
     │   ├─ 20% Random Explore：大幅随机调参，40%概率同步重写 system_prompt
     │   └─ 15% Hint-Driven：教师分析最差 case → 重写 system_prompt
     │
     ├─ StudentRAG(prompt_params={system_prompt, temperature, max_tokens})
     │   └─ 参数注入到 vLLM → 生成答案（使用学生实际检索上下文）
     │
     ├─ Evaluator + DeepEvalJudge — 三维评测（相关性/忠诚度/召回）
     │
     ├─ HintGenerator — 取评分最差的 3 条，教师生成改进建议
     │
     └─ 去重 + 早停：system_prompt 去重避免冗余 API 调用；
        teacher_win_rate 低于阈值时提前终止

  3. 返回 OptimizationResult（最佳参数、最佳分、全程历史）

关键设计决策：
  - 检索上下文使用学生实际检索结果（而非 golden 数据中的 null 值）
  - 教师基于学生检索到的上下文生成参考答案（教师在学生能看到的
    信息范围内评估，更有指导意义）
  - system_prompt 去重避免对相同提示词重复 API 调用
  - 容错：任何步骤失败都不中断整体流程，降级处理继续下一轮
"""

import copy
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dspy
from tqdm import tqdm

from .data import GoldenDataset, GoldenRecord
from .evaluator import DeepEvalJudge, EvalResult, Evaluator


# ═══════════════════════════════════════════════════════════════════════════════
# 默认系统提示词
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_SYSTEM_PROMPT = (
    "你是一个知识库问答助手。请基于检索到的上下文信息，"
    "准确、简洁地回答问题。如果无法从知识库中获得答案，请如实说明。"
)


# ═══════════════════════════════════════════════════════════════════════════════
# 数据类型定义
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DemoRecord:
    """
    一条 Bootstrap 演示记录。

    用于在优化过程中提供参考示例。正例（is_positive=True）
    表示学生答案已经很好，负例表示需要改进的方向。

    字段说明：
      - input: 用户问题
      - retrieval_context: 学生实际检索到的文档列表
      - expected_output: golden 数据中的标准答案
      - student_answer: 学生模型的当前答案
      - teacher_answer: 教师模型生成的参考答案（基于学生检索上下文）
      - score: 综合评分 (0-1)
      - is_positive: 是否为正例（score ≥ positive_threshold）
    """
    input: str
    retrieval_context: List[str]
    expected_output: str
    student_answer: str
    teacher_answer: str
    score: float
    is_positive: bool


@dataclass
class OptimizationTrial:
    """
    单次优化试验的结果。

    每轮 trial 记录当时使用的参数和获得的评分，
    用于追踪优化过程和分析趋势。
    """
    prompt_params: Dict[str, Any]      # 本轮使用的参数 {system_prompt, temperature, max_tokens}
    demos: List[DemoRecord]            # 当前的 demo 示例池
    score: float                       # 本轮在 dev 集上的综合评分
    teacher_win_rate: float            # 教师显著优于学生的样本比例
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationResult:
    """
    完整优化运行的结果。

    包含最佳参数、历史记录和最终指标，
    序列化后保存为 run_summary.json。
    """
    best_prompt_params: Dict[str, Any]      # 最佳 prompt 参数
    best_score: float                       # 最佳 dev 集评分
    trials: List[OptimizationTrial]         # 所有 trial 的详细记录
    history: List[Dict[str, Any]]           # 评分历史（精简版，方便绘图）
    final_metrics: Dict[str, float]         # 最终 demo 池的统计指标


# ═══════════════════════════════════════════════════════════════════════════════
# 教师评分器
# ═══════════════════════════════════════════════════════════════════════════════

class TeacherScorer:
    """
    利用教师模型（通过 DeepEvalJudge）对学生答案进行三维评分。

    评分流程：
      1. 如果有 judge → 创建 Evaluator，异步运行三维指标
      2. 如果没有 judge → 回退到基于关键词重叠的规则评分

    规则评分的局限性：
      - 仅基于 expected 和 student_answer 的字符集重叠
      - 不考虑语义、不考虑检索上下文
      - 仅作为无 judge 时的最低保障
    """

    def __init__(
        self,
        judge: Optional[DeepEvalJudge] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.judge = judge
        self.weights = weights or {
            "answer_relevancy": 0.3,
            "faithfulness": 0.4,
            "contextual_recall": 0.3,
        }

    def score(
        self,
        input: str,
        expected: str,
        student_answer: str,
        retrieval_context: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        对学生答案评分。

        Args:
            input: 用户问题
            expected: 标准答案
            student_answer: 学生答案
            retrieval_context: 检索上下文（用于 Faithfulness 判断）

        Returns:
            {answer_relevancy, faithfulness, contextual_recall, overall}
        """
        # 优先使用 DeepEval judge（更准确的 LLM 评判）
        if self.judge:
            evaluator = Evaluator(self.judge)
            records = [{
                "input": input,
                "expected_output": expected,
                "actual_output": student_answer,
                "retrieval_context": retrieval_context or [],
            }]
            results = evaluator.evaluate_sync(records, desc="Scoring")
            if results:
                r = results[0]
                return {
                    "answer_relevancy": r.answer_relevancy,
                    "faithfulness": r.faithfulness,
                    "contextual_recall": r.contextual_recall,
                    "overall": r.overall_score,
                }

        # 回退：基于字符集重叠的简单规则评分
        return self._rule_based_score(expected, student_answer)

    def _rule_based_score(
        self,
        expected: str,
        student_answer: str,
    ) -> Dict[str, float]:
        """
        规则评分（仅当 judge 不可用时）。

        方法：计算 expected 和 student_answer 的字符集重叠率，
        用 recall 作为 overall 分。简单粗暴但不准确。
        """
        expected_words = set(expected)
        student_words = set(student_answer)

        overlap = len(expected_words & student_words)
        recall = overlap / max(len(expected_words), 1)

        return {
            "answer_relevancy": min(1.0, recall * 1.2),
            "faithfulness": min(1.0, recall),
            "contextual_recall": recall,
            "overall": recall,
        }

    async def async_score(
        self,
        input: str,
        expected: str,
        student_answer: str,
        retrieval_context: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """异步版本的评分（用于优化循环内）"""
        evaluator = Evaluator(self.judge)
        records = [{
            "input": input,
            "expected_output": expected,
            "actual_output": student_answer,
            "retrieval_context": retrieval_context or [],
        }]
        results = await evaluator.evaluate(records, desc="Scoring")
        if results:
            r = results[0]
            return {
                "answer_relevancy": r.answer_relevancy,
                "faithfulness": r.faithfulness,
                "contextual_recall": r.contextual_recall,
                "overall": r.overall_score,
            }
        return {"answer_relevancy": 0.0, "faithfulness": 0.0, "contextual_recall": 0.0, "overall": 0.0}


# ═══════════════════════════════════════════════════════════════════════════════
# Bootstrap 演示生成器
# ═══════════════════════════════════════════════════════════════════════════════

class BootstrapDemonstrator:
    """
    冷启动演示生成器。

    对每个样本执行三步操作：
      1. 学生 RAG 检索 + 生成 → 获取 student_answer + 实际检索上下文
      2. 将学生检索上下文传给教师模型 → 获取高质量 teacher_answer
      3. TeacherScorer 评分 → 标注 positive（分数 ≥ 0.7）或 negative

    为什么用学生检索上下文而非 golden 上下文？
      - golden 数据中 retrieval_context 通常为 null
      - 教师在学生能看到的同一批文档范围内评估才公平
      - 学生的检索质量本身就是瓶颈之一

    返回的 DemoRecord 按分数降序排列，取 top max_demos。
    正例帮助模型学习"好答案是什么样的"，
    负例帮助模型了解"什么是不好的"。
    """

    def __init__(
        self,
        scorer: TeacherScorer,
        positive_threshold: float = 0.7,
        negative_threshold: float = 0.4,
    ):
        self.scorer = scorer
        self.positive_threshold = positive_threshold   # 高于此分为正例
        self.negative_threshold = negative_threshold   # 低于此分为明显负例

    def bootstrap(
        self,
        student_module: dspy.Module,
        teacher_module: dspy.Module,
        examples: List[dspy.Example],
        max_demos: int = 8,
    ) -> List[DemoRecord]:
        """
        从示例中生成 bootstrap demos。

        Args:
            student_module: 学生 RAG 模块（做检索+生成）
            teacher_module: 教师 API 模块（生成参考答案）
            examples: DSPy Example 列表
            max_demos: 最多返回的 demo 数量

        Returns:
            DemoRecord 列表（按分数降序）
        """
        demos = []

        for example in tqdm(examples, desc="Bootstrapping demos"):
            student_answer = ""
            retrieval_context = []

            # 步骤 1：学生模型推理（内部做 BM25+Milvus 检索 → vLLM 生成）
            try:
                student_pred = student_module(
                    input=example.input,
                    retrieval_context=example.retrieval_context or [],
                )
                student_answer = student_pred.answer
                # 捕获学生实际检索到的文档（非 golden 数据中的 null）
                retrieval_context = student_pred.retrieval_context or []
            except Exception as e:
                print(f"[WARN] Student inference failed: {e}")

            # 步骤 2：教师模型基于学生检索上下文生成参考答案
            try:
                teacher_pred = teacher_module(
                    input=example.input,
                    context=retrieval_context,
                )
                teacher_answer = teacher_pred.answer
            except Exception as e:
                print(f"[WARN] Teacher inference failed: {e}")
                teacher_answer = ""

            # 步骤 3：评分并标注正/负例
            expected = example.expected_output or ""
            scores = self.scorer.score(
                input=example.input,
                expected=expected,
                student_answer=student_answer,
                retrieval_context=retrieval_context,
            )

            is_positive = scores["overall"] >= self.positive_threshold

            demos.append(DemoRecord(
                input=example.input,
                retrieval_context=retrieval_context,
                expected_output=expected,
                student_answer=student_answer,
                teacher_answer=teacher_answer,
                score=scores["overall"],
                is_positive=is_positive,
            ))

        # 按评分降序排列，取 top-k
        demos.sort(key=lambda d: d.score, reverse=True)
        return demos[:max_demos]


# ═══════════════════════════════════════════════════════════════════════════════
# 改进提示生成器
# ═══════════════════════════════════════════════════════════════════════════════

class HintGenerator:
    """
    利用教师模型分析学生答案问题，生成改进建议。

    两个核心方法：
      1. generate_hint() — 分析单条问答，输出 JSON 格式的问题诊断
      2. synthesize_improved_prompt() — 基于多条 hint 反馈，用教师模型
         重写 system_prompt，以系统性地改进回答质量

    synthesize_improved_prompt 的安全检查：
      - 新 prompt 长度必须 ≥ 10 字符且 ≤ 2000 字符
      - 不符合要求时返回原 prompt（避免错误的过短/过长输出）
    """

    # 教师模型分析答案质量时使用的 system prompt
    SYSTEM_PROMPT = """你是一个严格的知识库问答质量评估员和提示词优化专家。
给定一个问题、学生答案和标准答案，你需要：
1. 分析学生答案的问题
2. 提供改进提示词的具体建议
3. 给出更好的答案示例

请以JSON格式输出，包含以下字段：
- issue: 学生答案的主要问题
- hint: 改进提示词的具体建议
- improved_answer: 更好的答案示例
- reasoning: 评分推理
- scores: {"answer_relevancy": 0-1, "faithfulness": 0-1, "contextual_recall": 0-1}
"""

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        import os
        from openai import OpenAI

        self.model = model
        self._api_key = api_key or os.environ.get("DOUBAO_API_KEY", "")
        self._base_url = base_url or os.environ.get("DOUBAO_BASE_URL", "")
        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)

    def generate_hint(
        self,
        input: str,
        student_answer: str,
        expected: str,
        retrieval_context: List[str],
    ) -> Dict[str, Any]:
        """
        分析一个学生回答，生成改进提示。

        返回的 hint 将在下一轮优化中用于"提示词驱动探索"策略。
        教师模型需要理解：
          - 学生看到了什么上下文（可能导致错误的原因）
          - 学生输出了什么（哪里有问题）
          - 标准答案是什么（应该达到的目标）

        Returns:
            {issue, hint, improved_answer, reasoning, scores}
            失败时返回降级结果
        """
        context_str = "\n".join(
            [f"[{i+1}] {ctx}" for i, ctx in enumerate(retrieval_context)]
        )

        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"【问题】{input}\n\n"
            f"【学生答案】{student_answer}\n\n"
            f"【标准答案】{expected}\n\n"
            f"【检索上下文】\n{context_str}\n\n"
            f"请以JSON格式输出分析结果。"
        )

        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            raw = completion.choices[0].message.content or "{}"
            result = json.loads(raw)
            return result
        except Exception as e:
            print(f"[WARN] Hint generation failed: {e}")
            # 降级返回：给出通用建议
            return {
                "issue": "未知问题",
                "hint": "请参考标准答案生成答案",
                "improved_answer": expected,
                "reasoning": f"生成失败: {e}",
                "scores": {"answer_relevancy": 0.0, "faithfulness": 0.0, "contextual_recall": 0.0},
            }

    def synthesize_improved_prompt(
        self,
        current_prompt: str,
        hints: List[Dict[str, Any]],
    ) -> str:
        """
        基于多条 hint 反馈，重写 system_prompt。

        教师模型作为"提示词工程师"，分析学生答案中最常见的问题，
        然后重写 system_prompt 来系统性地解决这些问题。

        输入限制：
          - 最多用 3 条 hint（取最严重的），防止 prompt 过长
          - 要求教师"只输出改进后的提示词文本"，避免解释性文字

        安全检查：
          - 长度 < 10 或 > 2000 → 返回原 prompt（忽略异常输出）
          - API 调用失败 → 返回原 prompt（不中断优化）
        """
        if not hints:
            return current_prompt

        # 取前 3 条最严重的 hint
        hints_text = "\n\n".join([
            f"问题{i+1}: {h.get('issue', 'N/A')}\n"
            f"改进建议: {h.get('hint', 'N/A')}"
            for i, h in enumerate(hints[:3])
        ])

        system_msg = (
            "你是一个提示词优化专家。根据以下评估反馈，重写系统提示词以改进模型回答质量。"
            "改进后的提示词应：1) 针对反馈中指出的问题，2) 保持简洁清晰，3) 使用中文。"
            "请只输出改进后的系统提示词文本，不要输出任何解释。"
        )

        user_prompt = (
            f"{system_msg}\n\n"
            f"【当前系统提示词】\n{current_prompt}\n\n"
            f"【评估反馈 - 学生模型最差的回答】\n{hints_text}\n\n"
            f"改进后的系统提示词："
        )

        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.3,
                max_tokens=1024,
            )
            new_prompt = (completion.choices[0].message.content or "").strip()
            # 安全检查：拒绝明显异常的提示词
            if len(new_prompt) < 10 or len(new_prompt) > 2000:
                return current_prompt
            return new_prompt
        except Exception as e:
            print(f"[WARN] Prompt synthesis failed: {e}")
            return current_prompt


# ═══════════════════════════════════════════════════════════════════════════════
# DYPS 优化器（核心类）
# ═══════════════════════════════════════════════════════════════════════════════

class DYPSOptimizer:
    """
    DYPS (Dynamic Prompt Selection) 提示词优化器。

    这是整个框架的核心，编排以下流程：

    ┌──────────────────────────────────────────────────────────┐
    │ cold_start(examples)                                      │
    │   └─ BootstrapDemonstrator.bootstrap() → DemoRecord 池    │
    │                                                           │
    │ optimize_async(train, dev, num_trials)                    │
    │   for trial in 1..num_trials:                             │
    │     ├─ generate_new_params(history, hint_feedback)        │
    │     │   ├─ 65% Exploit（保留提示词，小幅扰动）              │
    │     │   ├─ 20% Random Explore（大幅随机）                  │
    │     │   └─ 15% Hint-Driven（教师重写 system_prompt）      │
    │     ├─ system_prompt 去重检查                              │
    │     ├─ StudentRAG(prompt_params) → dev 集推理              │
    │     ├─ Evaluator.evaluate() → DeepEval 三维评分            │
    │     ├─ HintGenerator.generate_hint() × 3（最差 case）      │
    │     └─ 早停判断 teacher_win_rate < (1 - threshold)         │
    │                                                           │
    │ 返回 OptimizationResult(best_params, best_score, ...)     │
    └──────────────────────────────────────────────────────────┘

    使用示例：
        optimizer = DYPSOptimizer(
            student_module=student_rag,
            teacher_module=teacher_lm,
            judge=deep_eval_judge,
            dataset=golden_dataset,
            config={"num_trials": 50, "max_demos": 8},
        )
        result = await optimizer.optimize_async(train, dev)
        print(f"Best params: {result.best_prompt_params}")
    """

    def __init__(
        self,
        student_module: dspy.Module,
        teacher_module: dspy.Module,
        judge: Optional[DeepEvalJudge] = None,
        dataset: Optional[GoldenDataset] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        # 模型
        self.student = student_module      # 学生：本地 RAG pipeline
        self.teacher = teacher_module      # 教师：API 模型
        self.judge = judge                 # 评测员：DeepEval 包装的 API 模型
        self.dataset = dataset

        # 超参数（可运行时覆盖）
        cfg = config or {}
        self.num_trials = cfg.get("num_trials", 50)
        self.max_demos = cfg.get("max_demos", 8)
        self.cold_start_samples = cfg.get("cold_start_samples", 10)
        self.teacher_win_threshold = cfg.get("teacher_win_threshold", 0.7)
        self.min_improvement = cfg.get("min_improvement", 0.01)

        # 子组件
        self.scorer = TeacherScorer(judge=judge)
        self.demonstrator = BootstrapDemonstrator(scorer=self.scorer)
        self.hint_gen = HintGenerator()

        # 内部状态（跨 trial 持久化）
        self._demos: List[DemoRecord] = []                     # 当前 demo 示例池
        self._trials: List[OptimizationTrial] = []              # 所有 trial 记录
        self._history: List[Dict[str, Any]] = []                # 评分历史（精简）
        self._best_score = 0.0                                  # 最佳 dev 评分
        self._best_params: Dict[str, Any] = {}                  # 最佳参数
        self._last_hint_feedback: Optional[List[Dict[str, Any]]] = None  # 上一轮的 hint 反馈
        self._seen_prompts: set = set()                         # 已见过的 system_prompt 去重集合
        self.initial_system_prompt = cfg.get("initial_system_prompt")

    # ── 冷启动 ────────────────────────────────────────────────────────────

    def cold_start(self, examples: List[dspy.Example]) -> List[DemoRecord]:
        """
        冷启动：用少量样本生成初始 demo 池。

        这是优化的起点——建立对模型当前能力的基线认知。
        返回的 DemoRecord 中包含了：
          - 学生对每个问题的回答和得分
          - 教师的参考答案
          - 正/负例标注
        """
        print(f"[DYPS] Cold start with {len(examples)} examples...")
        demos = self.demonstrator.bootstrap(
            student_module=self.student,
            teacher_module=self.teacher,
            examples=examples,
            max_demos=self.max_demos,
        )
        self._demos = demos

        positive = sum(1 for d in demos if d.is_positive)
        print(f"[DYPS] Cold start complete: {len(demos)} demos, {positive} positive")
        return demos

    # ── Demo 池统计 ──────────────────────────────────────────────────────

    def score_demos(self) -> Dict[str, float]:
        """计算当前 demo 池的统计指标"""
        if not self._demos:
            return {"overall": 0.0}

        scores = [d.score for d in self._demos]
        return {
            "overall": sum(scores) / len(scores),
            "positive_rate": sum(1 for d in self._demos if d.is_positive) / len(self._demos),
            "max_score": max(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
        }

    # ── 参数生成（核心搜索策略）────────────────────────────────────────────

    def generate_new_params(
        self,
        history: List[Dict[str, Any]],
        hint_feedback: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        基于历史和 hint 反馈生成下一轮的参数。

        三种搜索策略（按概率选择）：

        ┌──────────┬──────┬─────────────────────────────────────────┐
        │ 策略      │ 概率  │ 行为                                      │
        ├──────────┼──────┼─────────────────────────────────────────┤
        │ Exploit   │ 65%  │ 保留最佳 system_prompt，小幅度扰动        │
        │           │      │ temperature ±0.05, max_tokens ±128       │
        │ Random    │ 20%  │ 大幅度随机调参，40%概率同时重写提示词      │
        │           │      │ temperature ±0.2, max_tokens ±512         │
        │ HintDriven│ 15%  │ 教师分析最差 case → 完全重写 system_prompt │
        └──────────┴──────┴─────────────────────────────────────────┘

        设计原理：
          - Exploit 占 65%：大多数时候应该微调已知好的参数（保守利用）
          - Random 占 20%：需要一定探索性防止陷入局部最优
          - HintDriven 仅 15%：重写 system_prompt 成本高（API 调用），
            且过度改变可能导致不稳定，所以概率最低
        """
        # 无历史 → 返回初始参数
        if not history:
            return {
                "system_prompt": self.initial_system_prompt or DEFAULT_SYSTEM_PROMPT,
                "temperature": random.uniform(0.0, 0.3),
                "max_tokens": random.randint(512, 2048),
            }

        # 从最后一轮历史中获取最佳参数
        best = history[-1]["params"]
        new_params = copy.deepcopy(best)
        hints_available = bool(hint_feedback and len(hint_feedback) > 0)

        roll = random.random()

        if roll < 0.15 and hints_available:
            # 策略 1 (15%)：提示词驱动探索
            # 教师重写 system_prompt + 小幅参数扰动
            new_params["system_prompt"] = self._synthesize_improved_prompt(
                current_prompt=best.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
                hints=hint_feedback,
            )
            self._mutate_inference_params(new_params, aggressive=False)

        elif roll < 0.35:
            # 策略 2 (20%)：随机探索
            # 40% 概率同时重写 system_prompt + 大幅扰动参数
            if random.random() < 0.4 and hints_available:
                new_params["system_prompt"] = self._synthesize_improved_prompt(
                    current_prompt=best.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
                    hints=hint_feedback,
                )
            self._mutate_inference_params(new_params, aggressive=True)

        else:
            # 策略 3 (65%)：利用（Exploit）
            # 保持 system_prompt 不变，仅小幅扰动 temperature 和 max_tokens
            self._mutate_inference_params(new_params, aggressive=False)

        return new_params

    def _mutate_inference_params(
        self,
        params: Dict[str, Any],
        aggressive: bool = False,
    ) -> None:
        """
        就地扰动 temperature 和 max_tokens 参数。

        每个参数 60% 概率被扰动（不是 100%，保留一些不变化的维度）。

        aggressive=False: temperature ±0.05, max_tokens ±128（微调）
        aggressive=True:  temperature ±0.2,  max_tokens ±512（探索）
        """
        temp_scale = 0.2 if aggressive else 0.05
        token_scale = 512 if aggressive else 128

        if random.random() < 0.6:
            params["temperature"] = max(0.0, min(1.0,
                params.get("temperature", 0.1) + random.uniform(-temp_scale, temp_scale)))
        if random.random() < 0.6:
            params["max_tokens"] = max(256, min(4096,
                int(params.get("max_tokens", 1024) + random.randint(-token_scale, token_scale))))

    def _synthesize_improved_prompt(
        self,
        current_prompt: str,
        hints: List[Dict[str, Any]],
    ) -> str:
        """转发到 HintGenerator 进行 prompt 重写"""
        return self.hint_gen.synthesize_improved_prompt(current_prompt, hints)

    # ── 异步优化主循环 ────────────────────────────────────────────────────

    async def optimize_async(
        self,
        train_examples: List[dspy.Example],
        dev_examples: List[dspy.Example],
        num_trials: Optional[int] = None,
    ) -> OptimizationResult:
        """
        异步优化主循环。

        Args:
            train_examples: 训练集（用于冷启动和可能的重采样）
            dev_examples: 开发集（用于每轮 trial 的评测）
            num_trials: 覆盖默认的 trial 数量

        Returns:
            OptimizationResult: 最佳参数、最佳分、全程历史

        每轮 trial 的详细步骤：

        1. generate_new_params() → 基于历史和 hint 反馈生成候选参数
        2. 去重检查 → 如果 system_prompt 已见过，加重参数扰动
        3. StudentRAG(prompt_params) → 在 dev 集上批量推理
        4. Evaluator.evaluate() → DeepEval 三维评测
        5. 计算 teacher_win_rate（教师在多少比例样本上显著优于学生）
        6. HintGenerator.generate_hint() × 3 → 分析最差的 3 条
        7. 早停判断 → teacher_win_rate < (1 - threshold) 时退出

        早停条件解读：
          如果 teacher_win_rate < (1 - 0.7) = 0.3，即不足 30% 的样本上
          教师显著优于学生，说明学生已经足够好，可以停止优化。
        """
        trials = num_trials or self.num_trials

        # 冷启动（如果尚未执行）
        if not self._demos:
            cold_examples = train_examples[:self.cold_start_samples]
            self.cold_start(cold_examples)

        print(f"[DYPS] Starting optimization for {trials} trials...")

        for trial_idx in range(trials):
            # ── 步骤 1：生成新参数 ──
            params = self.generate_new_params(
                history=self._history,
                hint_feedback=self._last_hint_feedback,
            )

            # ── 步骤 2：system_prompt 去重 ──
            # 相同提示词产生相同结果 → 浪费 API 调用
            # 如果已见过，加重扰动使其不同
            prompt_key = params.get("system_prompt", "")[:200]
            if prompt_key in self._seen_prompts and len(self._seen_prompts) > 0:
                self._mutate_inference_params(params, aggressive=True)
                prompt_key = params.get("system_prompt", "")[:200]
            self._seen_prompts.add(prompt_key)

            # ── 步骤 3：在 dev 集上推理 ──
            dev_records = []
            for ex in dev_examples:
                retrieval_context = []
                try:
                    pred = self.student(
                        input=ex.input,
                        retrieval_context=ex.retrieval_context or [],
                        prompt_params=params,  # 动态注入优化参数
                    )
                    actual = pred.answer
                    # 使用学生实际检索的上下文（非 golden 数据中的 null）
                    retrieval_context = pred.retrieval_context or []
                except Exception as e:
                    print(f"[WARN] Trial {trial_idx} inference failed: {e}")
                    actual = ""

                dev_records.append({
                    "input": ex.input,
                    "expected_output": ex.expected_output or "",
                    "actual_output": actual,
                    "retrieval_context": retrieval_context,
                    "unique_id": getattr(ex, "unique_id", str(trial_idx)),
                    "category": getattr(ex, "category", ""),
                })

            # ── 步骤 4：DeepEval 评测 ──
            if self.judge:
                evaluator = Evaluator(self.judge)
                results = await evaluator.evaluate(dev_records, desc=f"Trial {trial_idx+1}/{trials}")
                overall = sum(r.overall_score for r in results) / len(results)

                # teacher_win_rate: answer_relevancy < 0.5 表示教师显著优于学生
                teacher_wins = 0
                for r in results:
                    if r.answer_relevancy < 0.5:
                        teacher_wins += 1
                teacher_win_rate = teacher_wins / len(results)
            else:
                # 无 judge 时回退到 demo 池分数
                demo_scores = self.score_demos()
                overall = demo_scores["overall"]
                teacher_win_rate = 1.0 - demo_scores["positive_rate"]

            # 记录 trial 结果
            trial = OptimizationTrial(
                prompt_params=params,
                demos=copy.deepcopy(self._demos),
                score=overall,
                teacher_win_rate=teacher_win_rate,
            )
            self._trials.append(trial)
            self._history.append({
                "trial": trial_idx,
                "params": params,
                "score": overall,
                "teacher_win_rate": teacher_win_rate,
            })

            print(f"Trial {trial_idx+1}/{trials}: score={overall:.4f}, teacher_win_rate={teacher_win_rate:.2f}")

            # ── 步骤 6：生成 hint 反馈（为下一轮准备）──
            if self.judge and results:
                # 取评分最差的 3 条，教师分析问题并生成改进建议
                sorted_results = sorted(
                    zip(dev_records, results),
                    key=lambda pair: pair[1].overall_score,
                )
                worst_pairs = sorted_results[:3]

                hint_feedback = []
                for record, eval_result in worst_pairs:
                    try:
                        hint = self.hint_gen.generate_hint(
                            input=record["input"],
                            student_answer=record["actual_output"],
                            expected=record["expected_output"],
                            retrieval_context=record.get("retrieval_context", []),
                        )
                        hint_feedback.append(hint)
                    except Exception as e:
                        print(f"[WARN] Hint generation failed: {e}")

                self._last_hint_feedback = hint_feedback if hint_feedback else None
            else:
                self._last_hint_feedback = None

            # ── 步骤 7：早停判断 ──
            # teacher_win_rate 低 → 学生在大多数样本上已经接近或超过教师
            if teacher_win_rate < (1.0 - self.teacher_win_threshold):
                print(
                    f"[DYPS] Early stopping at trial {trial_idx+1}: "
                    f"student performance sufficient "
                    f"(improvement needed on {teacher_win_rate:.1%} of examples)"
                )
                break

        # ── 汇总最佳结果 ──
        best_trial = max(self._trials, key=lambda t: t.score)
        self._best_score = best_trial.score
        self._best_params = best_trial.prompt_params

        print(f"\n[DYPS] Optimization complete. Best score: {self._best_score:.4f}")

        return OptimizationResult(
            best_prompt_params=self._best_params,
            best_score=self._best_score,
            trials=self._trials,
            history=self._history,
            final_metrics=self.score_demos(),
        )

    # ── 同步包装器 ────────────────────────────────────────────────────────

    def optimize(
        self,
        train_examples: List[dspy.Example],
        dev_examples: List[dspy.Example],
        num_trials: Optional[int] = None,
    ) -> OptimizationResult:
        """同步版本的优化方法（内部调用 asyncio.run）"""
        import asyncio
        return asyncio.run(
            self.optimize_async(train_examples, dev_examples, num_trials)
        )

    # ── 结果访问与保存 ────────────────────────────────────────────────────

    def get_best_params(self) -> Dict[str, Any]:
        """获取当前找到的最佳参数"""
        return self._best_params

    def save_results(self, output_path: Path) -> None:
        """将优化结果保存为 JSON 文件"""
        result = {
            "best_params": self._best_params,
            "best_score": self._best_score,
            "num_trials": len(self._trials),
            "history": self._history,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[DYPS] Results saved to {output_path}")
