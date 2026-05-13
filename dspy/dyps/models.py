# -*- coding: utf-8 -*-
"""
Model wrappers for DSPy DYPS.

Provides three model types:
1. StudentRAG - the local RAG pipeline (student model to be optimized)
2. TeacherLM  - a strong API model used as teacher for scoring / hints
3. DSPyLM     - wraps any OpenAI-compatible API as a DSPy LM for DSPy native use
"""

import os
import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import dspy
from openai import OpenAI

# Add RAG project to path for importing HybridRAGPipeline
RAG_ROOT = Path("/root/autodl-tmp/RAG")
if str(RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_ROOT))


# ──────────────────────────── 学生模型：本地 RAG ────────────────────────────

class StudentRAG(dspy.Module):
    """
    Student model: wraps the local HybridRAGPipeline.

    This is the model whose prompt we want to optimize via DYPS.
    DSPy will call this module; we intercept the prompt for optimization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        cfg = config or {}

        from src.utils import HybridRAGConfig, HybridRAGPipeline

        rag_config = HybridRAGConfig(
            bm25_topk=cfg.get("bm25_topk", 5),
            milvus_topk=cfg.get("milvus_topk", 10),
            rerank_topk=cfg.get("rerank_topk", 5),
        )
        self.pipeline = HybridRAGPipeline(config=rag_config)

    def forward(self, input: str, prompt_params: Optional[Dict[str, Any]] = None, **kwargs) -> dspy.Prediction:
        """
        Run the student RAG pipeline.

        Args:
            input: User query (question)
            prompt_params: Optional dict with keys: temperature, max_tokens, use_cot

        Returns:
            DSPy Prediction with 'answer' and 'retrieval_context' fields
        """
        params = prompt_params or {}
        temperature = params.get("temperature")
        max_tokens = params.get("max_tokens")
        system_prompt = params.get("system_prompt") or None

        result = self.pipeline.answer(input, temperature=temperature, max_tokens=max_tokens, system_prompt=system_prompt)

        # Extract answer text (post-processed, without citations)
        answer_text = result["answer"]["answer"]

        return dspy.Prediction(
            answer=answer_text,
            retrieval_context=result["retrieval_context"],
        )


# ──────────────────────────── 教师模型：API 调用 ────────────────────────────

class TeacherLM(dspy.Module):
    """
    Teacher model: uses a strong API model to generate reference answers.

    The teacher provides high-quality answers that guide the student
    toward better prompt engineering. Used for scoring and bootstrapping.
    """

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        super().__init__()

        self.model = model
        self.api_key = api_key or os.environ.get("DOUBAO_API_KEY", "")
        self.base_url = base_url or os.environ.get("DOUBAO_BASE_URL", "")
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def forward(self, input: str, context: List[str]) -> dspy.Prediction:
        """
        Generate a teacher answer using context + query.

        Args:
            input: User query
            context: List of retrieved context strings

        Returns:
            DSPy Prediction with 'answer' field
        """
        context_str = "\n".join([f"[{i + 1}] {ctx}" for i, ctx in enumerate(context)])
        prompt = self._build_prompt(input, context_str)
        response = self._call(prompt)

        return dspy.Prediction(answer=response.strip())

    def _build_prompt(self, query: str, context: str) -> str:
        return (
            f"你是一个严格的知识库问答助手。请根据以下检索到的上下文信息，"
            f"准确、完整地回答用户问题。\n\n"
            f"【检索上下文】\n{context}\n\n"
            f"【用户问题】{query}\n\n"
            f"请直接给出答案，不要提及你是基于检索结果回答。"
        )

    def _call(self, prompt: str) -> str:
        """Call the API."""
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return completion.choices[0].message.content or ""

    def grade(self, input: str, expected: str, actual: str) -> Dict[str, Any]:
        """
        Grade a student answer against expected answer.

        Returns a dict with scores for different metrics.
        """
        grading_prompt = (
            f"你是一个严格的质量评估员。请评估以下答案的质量。\n\n"
            f"【问题】{input}\n\n"
            f"【标准答案】{expected}\n\n"
            f"【待评测答案】{actual}\n\n"
            f"请从以下三个维度打分（0-1分，1分最优）：\n"
            f"1. 答案相关性（Answer Relevancy）：答案是否针对问题本身\n"
            f"2. 忠诚度（Faithfulness）：答案是否忠实于标准答案，没有编造\n"
            f"3. 上下文召回（Contextual Recall）：答案是否覆盖了标准答案的关键信息\n\n"
            f"请以JSON格式输出：{{\"answer_relevancy\": 0.0, \"faithfulness\": 0.0, \"contextual_recall\": 0.0}}"
        )
        raw = self._call(grading_prompt)
        try:
            # Extract JSON from response
            import json as _json
            match = re.search(r'\{[^}]+\}', raw)
            if match:
                scores = _json.loads(match.group())
                return scores
        except Exception:
            pass
        return {"answer_relevancy": 0.0, "faithfulness": 0.0, "contextual_recall": 0.0}


# ──────────────────────────── DSPy LM 包装器 ────────────────────────────

class DSPyLMWrapper:
    """
    Wraps an OpenAI-compatible API as a DSPy LM for DSPy native features.

    Usage:
        lm = DSPyLMWrapper(
            model="openai/deepseek-v4-flash",
            api_key=os.environ["DOUBAO_API_KEY"],
            base_url=os.environ.get("DOUBAO_BASE_URL"),
        )
        dspy.configure(lm=lm)
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        # DSPy LM expects model string like "openai/gpt-4"
        self.model_name = model  # e.g. "openai/deepseek-v4-flash"
        self._api_key = api_key or os.environ.get("DOUBAO_API_KEY", "")
        self._base_url = api_base or os.environ.get("DOUBAO_BASE_URL", "")
        self.temperature = temperature
        self.max_tokens = max_tokens

        # LiteLLM (DSPy 3.x backend) uses api_base not base_url for custom endpoints
        self._lm = dspy.LM(
            model=model,
            api_key=self._api_key,
            api_base=self._base_url,
            temperature=temperature,
        )

    def configure(self) -> None:
        """Configure DSPy to use this LM."""
        dspy.configure(lm=self._lm)

    def __getattr__(self, name: str):
        """Proxy attribute access to the underlying DSPy LM."""
        return getattr(self._lm, name)
