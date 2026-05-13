# -*- coding: utf-8 -*-
"""
DSPy Teleprompter for RAG prompt optimization using teacher model guidance.

This module implements the core DYPS pipeline:
1. Bootstrap demonstrations using teacher model
2. Score student answers with teacher model (or DeepEval)
3. Generate hints to improve prompts
4. Optimize prompt parameters

Key classes:
- TeacherScorer: Scores student answers using teacher model
- BootstrapDemonstrator: Generates high-quality training examples
- HintGenerator: Generates improvement hints from teacher model
- DYPSOptimizer: Main optimization loop
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


# ──────────────────────────── 默认提示词 ────────────────────────────

DEFAULT_SYSTEM_PROMPT = (
    "你是一个知识库问答助手。请基于检索到的上下文信息，"
    "准确、简洁地回答问题。如果无法从知识库中获得答案，请如实说明。"
)


# ──────────────────────────── 数据类型 ────────────────────────────

@dataclass
class DemoRecord:
    """A demonstration record for bootstrap."""
    input: str
    retrieval_context: List[str]
    expected_output: str
    student_answer: str
    teacher_answer: str
    score: float
    is_positive: bool  # Teacher thinks this is a good demo


@dataclass
class OptimizationTrial:
    """A single optimization trial."""
    prompt_params: Dict[str, Any]
    demos: List[DemoRecord]
    score: float
    teacher_win_rate: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationResult:
    """Result of a full optimization run."""
    best_prompt_params: Dict[str, Any]
    best_score: float
    trials: List[OptimizationTrial]
    history: List[Dict[str, Any]]  # Score history over iterations
    final_metrics: Dict[str, float]


# ──────────────────────────── 教师评分器 ────────────────────────────

class TeacherScorer:
    """
    Uses teacher model to score student answers.

    The teacher model acts as a "gold standard" to score how well
    the student model performs, guiding the optimization process.
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
        Score a student answer against expected answer.

        Returns:
            Dict with answer_relevancy, faithfulness, contextual_recall, overall
        """
        # Use DeepEval judge if available
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

        # Fallback to rule-based scoring
        return self._rule_based_score(expected, student_answer)

    def _rule_based_score(
        self,
        expected: str,
        student_answer: str,
    ) -> Dict[str, float]:
        """Simple rule-based scoring when no judge model available."""
        # Check keyword overlap
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
        """Async version of score."""
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


# ──────────────────────────── Bootstrap 演示生成器 ────────────────────────────

class BootstrapDemonstrator:
    """
    Generates high-quality demonstrations using teacher model.

    For each sample, it:
    1. Runs student to get an answer
    2. Runs teacher to get a reference answer
    3. Scores the student answer
    4. Stores high-quality demos for optimization
    """

    def __init__(
        self,
        scorer: TeacherScorer,
        positive_threshold: float = 0.7,
        negative_threshold: float = 0.4,
    ):
        self.scorer = scorer
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold

    def bootstrap(
        self,
        student_module: dspy.Module,
        teacher_module: dspy.Module,
        examples: List[dspy.Example],
        max_demos: int = 8,
    ) -> List[DemoRecord]:
        """
        Generate bootstrap demonstrations.

        Args:
            student_module: The student RAG module
            teacher_module: The teacher API module
            examples: DSPy examples to use
            max_demos: Maximum number of demos to return

        Returns:
            List of DemoRecord with positive and negative examples
        """
        demos = []

        for example in tqdm(examples, desc="Bootstrapping demos"):
            student_answer = ""
            retrieval_context = []

            # Get student answer (student RAG does its own BM25+Milvus retrieval internally)
            try:
                student_pred = student_module(
                    input=example.input,
                    retrieval_context=example.retrieval_context or [],
                )
                student_answer = student_pred.answer
                # Use the actual retrieved context from student RAG for teacher and scoring
                retrieval_context = student_pred.retrieval_context or []
            except Exception as e:
                print(f"[WARN] Student inference failed: {e}")

            # Get teacher answer (pass student-retrieved context for high-quality reference)
            try:
                teacher_pred = teacher_module(
                    input=example.input,
                    context=retrieval_context,
                )
                teacher_answer = teacher_pred.answer
            except Exception as e:
                print(f"[WARN] Teacher inference failed: {e}")
                teacher_answer = ""

            # Score
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

        # Sort by score and return top demos
        demos.sort(key=lambda d: d.score, reverse=True)
        return demos[:max_demos]


# ──────────────────────────── 提示生成器 ────────────────────────────

class HintGenerator:
    """
    Generates improvement hints using teacher model.

    Analyzes student answers and provides actionable feedback
    for improving prompts.
    """

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
        Generate a hint for improving the student answer.

        Returns:
            Dict with issue, hint, improved_answer, reasoning, scores
        """
        context_str = "\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(retrieval_context)])

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
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            raw = completion.choices[0].message.content or "{}"
            result = json.loads(raw)
            return result
        except Exception as e:
            print(f"[WARN] Hint generation failed: {e}")
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
        Synthesize an improved system_prompt from aggregated evaluation feedback.

        The teacher model rewrites the system_prompt to address the issues
        identified in the worst-performing examples.
        """
        if not hints:
            return current_prompt

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
            if len(new_prompt) < 10 or len(new_prompt) > 2000:
                return current_prompt
            return new_prompt
        except Exception as e:
            print(f"[WARN] Prompt synthesis failed: {e}")
            return current_prompt


# ──────────────────────────── DYPS 优化器 ────────────────────────────

class DYPSOptimizer:
    """
    Main optimization loop for RAG prompt engineering using teacher guidance.

    The DYPS (Dynamic Prompt Selection) approach:
    1. Cold start: Generate initial demos using teacher model
    2. Bootstrap: Build demos with diverse (good + bad) examples
    3. Optimize: Use teacher feedback to improve prompt parameters
    4. Evaluate: Use DeepEval metrics to measure improvement

    Usage:
        optimizer = DYPSOptimizer(
            student=student_rag,
            teacher=teacher_lm,
            judge=deepseek_judge,
            dataset=goldens,
        )
        result = optimizer.optimize(num_trials=50)
    """

    def __init__(
        self,
        student_module: dspy.Module,
        teacher_module: dspy.Module,
        judge: Optional[DeepEvalJudge] = None,
        dataset: Optional[GoldenDataset] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.student = student_module
        self.teacher = teacher_module
        self.judge = judge
        self.dataset = dataset

        cfg = config or {}
        self.num_trials = cfg.get("num_trials", 50)
        self.max_demos = cfg.get("max_demos", 8)
        self.cold_start_samples = cfg.get("cold_start_samples", 10)
        self.teacher_win_threshold = cfg.get("teacher_win_threshold", 0.7)
        self.min_improvement = cfg.get("min_improvement", 0.01)

        # Components
        self.scorer = TeacherScorer(judge=judge)
        self.demonstrator = BootstrapDemonstrator(scorer=self.scorer)
        self.hint_gen = HintGenerator()

        # State
        self._demos: List[DemoRecord] = []
        self._trials: List[OptimizationTrial] = []
        self._history: List[Dict[str, Any]] = []
        self._best_score = 0.0
        self._best_params: Dict[str, Any] = {}
        self._last_hint_feedback: Optional[List[Dict[str, Any]]] = None
        self._seen_prompts: set = set()
        self.initial_system_prompt = cfg.get("initial_system_prompt")

    def cold_start(self, examples: List[dspy.Example]) -> List[DemoRecord]:
        """
        Cold start: Generate initial demonstrations.

        Uses teacher model to provide high-quality examples.
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

    def score_demos(self) -> Dict[str, float]:
        """Calculate overall score from current demos."""
        if not self._demos:
            return {"overall": 0.0}

        scores = [d.score for d in self._demos]
        return {
            "overall": sum(scores) / len(scores),
            "positive_rate": sum(1 for d in self._demos if d.is_positive) / len(self._demos),
            "max_score": max(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
        }

    def generate_new_params(
        self,
        history: List[Dict[str, Any]],
        hint_feedback: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate new prompt parameters based on history and hint feedback.

        Three-way strategy:
        - 15% Hint-driven exploration: rewrite system_prompt using teacher feedback
        - 20% Random exploration: larger parameter jumps
        - 65% Exploit: keep best system_prompt, small parameter tweaks
        """
        if not history:
            return {
                "system_prompt": self.initial_system_prompt or DEFAULT_SYSTEM_PROMPT,
                "temperature": random.uniform(0.0, 0.3),
                "max_tokens": random.randint(512, 2048),
            }

        best = history[-1]["params"]
        new_params = copy.deepcopy(best)
        hints_available = bool(hint_feedback and len(hint_feedback) > 0)

        roll = random.random()

        if roll < 0.15 and hints_available:
            # Hint-driven prompt exploration: rewrite system_prompt via teacher
            new_params["system_prompt"] = self._synthesize_improved_prompt(
                current_prompt=best.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
                hints=hint_feedback,
            )
            self._mutate_inference_params(new_params, aggressive=False)

        elif roll < 0.35:
            # Random exploration: larger parameter jumps
            if random.random() < 0.4 and hints_available:
                new_params["system_prompt"] = self._synthesize_improved_prompt(
                    current_prompt=best.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
                    hints=hint_feedback,
                )
            self._mutate_inference_params(new_params, aggressive=True)

        else:
            # Exploit: keep best system_prompt, small parameter tweaks
            self._mutate_inference_params(new_params, aggressive=False)

        return new_params

    def _mutate_inference_params(
        self,
        params: Dict[str, Any],
        aggressive: bool = False,
    ) -> None:
        """Mutate temperature and max_tokens in-place."""
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
        """Forward to HintGenerator for prompt synthesis."""
        return self.hint_gen.synthesize_improved_prompt(current_prompt, hints)

    async def optimize_async(
        self,
        train_examples: List[dspy.Example],
        dev_examples: List[dspy.Example],
        num_trials: Optional[int] = None,
    ) -> OptimizationResult:
        """
        Run async optimization loop.

        Args:
            train_examples: Training examples for bootstrap
            dev_examples: Development examples for evaluation

        Returns:
            OptimizationResult with best params and history
        """
        trials = num_trials or self.num_trials

        # Cold start
        if not self._demos:
            cold_examples = train_examples[:self.cold_start_samples]
            self.cold_start(cold_examples)

        print(f"[DYPS] Starting optimization for {trials} trials...")

        for trial_idx in range(trials):
            # Generate new params (feed hint feedback from previous trial)
            params = self.generate_new_params(
                history=self._history,
                hint_feedback=self._last_hint_feedback,
            )

            # Deduplicate system prompts to avoid redundant API calls
            prompt_key = params.get("system_prompt", "")[:200]
            if prompt_key in self._seen_prompts and len(self._seen_prompts) > 0:
                self._mutate_inference_params(params, aggressive=True)
                prompt_key = params.get("system_prompt", "")[:200]
            self._seen_prompts.add(prompt_key)

            # Evaluate on dev set
            dev_records = []
            for ex in dev_examples:
                retrieval_context = []
                try:
                    pred = self.student(
                        input=ex.input,
                        retrieval_context=ex.retrieval_context or [],
                        prompt_params=params,
                    )
                    actual = pred.answer
                    # Use student RAG's actual retrieved context for evaluation
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

            # Use evaluator if judge available
            if self.judge:
                evaluator = Evaluator(self.judge)
                results = await evaluator.evaluate(dev_records, desc=f"Trial {trial_idx+1}/{trials}")
                overall = sum(r.overall_score for r in results) / len(results)

                # Teacher win rate: how often teacher would give better answer
                teacher_wins = 0
                for r in results:
                    if r.answer_relevancy < 0.5:
                        teacher_wins += 1
                teacher_win_rate = teacher_wins / len(results)
            else:
                # Simple scoring
                demo_scores = self.score_demos()
                overall = demo_scores["overall"]
                teacher_win_rate = 1.0 - demo_scores["positive_rate"]

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

            # Generate hint feedback for next trial (analyze worst examples)
            if self.judge and results:
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

            # Early stopping: exit when student performs well enough
            # teacher_win_rate measures the fraction of examples where teacher
            # would significantly outperform; low value means student is good
            if teacher_win_rate < (1.0 - self.teacher_win_threshold):
                print(
                    f"[DYPS] Early stopping at trial {trial_idx+1}: "
                    f"student performance sufficient "
                    f"(improvement needed on {teacher_win_rate:.1%} of examples)"
                )
                break

        # Find best
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

    def optimize(
        self,
        train_examples: List[dspy.Example],
        dev_examples: List[dspy.Example],
        num_trials: Optional[int] = None,
    ) -> OptimizationResult:
        """Synchronous wrapper for optimize_async()."""
        import asyncio
        return asyncio.run(
            self.optimize_async(train_examples, dev_examples, num_trials)
        )

    def get_best_params(self) -> Dict[str, Any]:
        """Get the best prompt parameters found so far."""
        return self._best_params

    def save_results(self, output_path: Path) -> None:
        """Save optimization results to file."""
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