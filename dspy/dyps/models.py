# -*- coding: utf-8 -*-
"""
DSPy DYPS 模型封装模块
======================

提供三类模型的统一封装，分别对应教师-学生架构中的不同角色：

  1. StudentRAG (dspy.Module)
     ─ 包装本地 HybridRAGPipeline（BM25 + Milvus + BGE Reranker + Qwen3-8B vLLM）
     ─ 这是 DYPS 要优化的目标模型
     ─ forward() 接收 prompt_params 字典，动态注入 system_prompt / temperature / max_tokens

  2. TeacherLM (dspy.Module)
     ─ 通过 API 调用 deepseek-v4-flash
     ─ 三大职责：生成参考答案、评分学生答案、作为知识源
     ─ 学生用自己的检索上下文请求教师，而非依赖 golden 数据中的上下文

  3. DSPyLMWrapper
     ─ 将 API 模型包装为 DSPy 原生的 LM 对象
     ─ 供 DSPy 内部操作使用（bootstrap demos、ChainOfThought 等）
     ─ 内部通过 LiteLLM（DSPy 3.x 后端）路由请求

参数注入链路（核心设计）：
  DYPSOptimizer
    → StudentRAG.forward(input, prompt_params={system_prompt, temperature, max_tokens})
      → HybridRAGPipeline.answer(query, temperature=..., max_tokens=..., system_prompt=...)
        → request_chat(query, context, temperature, max_tokens, system_prompt)
          → vLLM API (Qwen3-8B 本地推理)
"""

import os
import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import dspy
from openai import OpenAI

# 将外部 RAG 项目加入 sys.path，以便导入 HybridRAGPipeline
RAG_ROOT = Path("/root/autodl-tmp/RAG")
if str(RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
# 学生模型：本地 RAG Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class StudentRAG(dspy.Module):
    """
    学生模型（被优化的对象）。

    包装了本地 HybridRAGPipeline，该 pipeline 包含：
      1. BM25 关键词检索 → 快速召回候选文档
      2. Milvus 向量检索 → 语义相似度召回
      3. BGE Reranker 重排序 → 融合并精选 top-k
      4. Qwen3-8B vLLM 推理 → 基于检索上下文生成答案

    DSPy 调用流程：
      student(input="问题", prompt_params={...})  →  dspy.Prediction(answer, retrieval_context)

    prompt_params 参数注入是 DYPS 优化的关键：
      - system_prompt: 替换默认 system prompt，引导模型回答风格（核心优化目标）
      - temperature: 控制生成随机性
      - max_tokens: 限制输出长度
      这些参数透明穿透到 vLLM，不需要修改任何文件或配置。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        cfg = config or {}

        from src.utils import HybridRAGConfig, HybridRAGPipeline

        # 用配置构建 RAG pipeline（BM25/Milvus/Rerank 的 topk 参数）
        rag_config = HybridRAGConfig(
            bm25_topk=cfg.get("bm25_topk", 5),
            milvus_topk=cfg.get("milvus_topk", 10),
            rerank_topk=cfg.get("rerank_topk", 5),
        )
        self.pipeline = HybridRAGPipeline(config=rag_config)

    def forward(
        self,
        input: str,
        prompt_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> dspy.Prediction:
        """
        学生模型前向推理。

        Args:
            input: 用户问题（question / query）
            prompt_params: 可选参数字典，DYPS 优化器注入：
                {
                    "system_prompt": str | None,   # 自定义 system prompt
                    "temperature": float | None,    # LLM 温度（0.0~1.0）
                    "max_tokens": int | None,       # 最大输出 token 数
                }
            **kwargs: 接受额外参数（如 retrieval_context），保持接口兼容

        Returns:
            dspy.Prediction:
                - answer: 生成的答案文本（已后处理，不含引用标记）
                - retrieval_context: 学生实际检索到的文档列表（供教师模型和评测使用）
        """
        params = prompt_params or {}
        temperature = params.get("temperature")
        max_tokens = params.get("max_tokens")
        system_prompt = params.get("system_prompt") or None

        # 调用 RAG pipeline，参数直接穿透到 vLLM
        result = self.pipeline.answer(
            input,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )

        # result["answer"]["answer"] 是后处理过的纯文本答案（无引用标记）
        answer_text = result["answer"]["answer"]

        return dspy.Prediction(
            answer=answer_text,
            retrieval_context=result["retrieval_context"],
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 教师模型：强 API 模型
# ═══════════════════════════════════════════════════════════════════════════════

class TeacherLM(dspy.Module):
    """
    教师模型（提供监督信号）。

    使用 deepseek-v4-flash API，在 DYPS 中承担三个关键角色：
      1. 生成参考答案 — 基于学生检索到的上下文生成高质量答案（BootstrapDemonstrator）
      2. 评分学生答案 — grade() 方法从三个维度打分（TeacherScorer 的备选方案）
      3. 知识源 — 提供更强的推理能力来指导弱模型

    设计决策：教师模型的提示词是固定的中文模板，不需要被优化——
    因为教师本身就是能力强的大模型，提示词优化对它影响很小。

    关键区别：教师用学生检索到的上下文（而非 golden 数据中的上下文），
    这确保了教师在"学生能看到的信息"范围内评估学生，更具指导意义。
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

        # OpenAI 兼容客户端（用于非 DSPy 的独立 API 调用）
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def forward(self, input: str, context: List[str]) -> dspy.Prediction:
        """
        教师模型生成参考答案。

        Args:
            input: 用户问题
            context: 检索上下文列表（应为学生实际检索到的文档）

        Returns:
            dspy.Prediction(answer=教师参考答案)
        """
        context_str = "\n".join(
            [f"[{i + 1}] {ctx}" for i, ctx in enumerate(context)]
        )
        prompt = self._build_prompt(input, context_str)
        response = self._call(prompt)

        return dspy.Prediction(answer=response.strip())

    def _build_prompt(self, query: str, context: str) -> str:
        """
        构建教师推理的提示词。

        提示词设计要点：
          - 角色设定为"严格的知识库问答助手"
          - 强调基于检索上下文回答（不准编造）
          - 要求学生直接给出答案（避免"根据检索结果..."等套话）
        """
        return (
            f"你是一个严格的知识库问答助手。请根据以下检索到的上下文信息，"
            f"准确、完整地回答用户问题。\n\n"
            f"【检索上下文】\n{context}\n\n"
            f"【用户问题】{query}\n\n"
            f"请直接给出答案，不要提及你是基于检索结果回答。"
        )

    def _call(self, prompt: str) -> str:
        """调用 API 并提取回复文本"""
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
        教师评分学生答案（三维打分）。

        评分维度：
          1. 答案相关性 (Answer Relevancy): 答案是否针对问题本身
          2. 忠诚度 (Faithfulness): 答案是否忠实于标准答案，没有编造
          3. 上下文召回 (Contextual Recall): 答案是否覆盖了标准答案的关键信息

        注意：此方法通过构造 prompt 让教师模型打分，而非使用 DeepEval。
        通常优先使用 DeepEvalJudge（包装为 DeepEval 兼容接口），此方法作为备选。

        Returns:
            {"answer_relevancy": float, "faithfulness": float, "contextual_recall": float}
            解析失败时返回全 0.0
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
            import json as _json
            # 从回复中提取 JSON 对象（容错处理：可能混在其他文本中）
            match = re.search(r'\{[^}]+\}', raw)
            if match:
                scores = _json.loads(match.group())
                return scores
        except Exception:
            pass
        # 解析失败时返回零分（而非崩溃），保证优化流程可以继续
        return {"answer_relevancy": 0.0, "faithfulness": 0.0, "contextual_recall": 0.0}


# ═══════════════════════════════════════════════════════════════════════════════
# DSPy LM 包装器
# ═══════════════════════════════════════════════════════════════════════════════

class DSPyLMWrapper:
    """
    将 OpenAI 兼容 API 包装为 DSPy 原生 LM。

    为什么需要这个包装器？
      DSPy 框架内部需要 LM 对象来执行一些内置操作：
        - 编译 prompt（自动生成 few-shot 示例）
        - Bootstrap 生成演示（从少量样本泛化）
        - ChainOfThought 推理链

    使用 LiteLLM 作为 DSPy 3.x 的后端，模型名格式为 "openai/<model>"。
    api_base 参数（非 base_url）是 LiteLLM 识别自定义端点的标准方式。

    使用示例：
        lm = DSPyLMWrapper(
            model="openai/deepseek-v4-flash",
            api_key=os.environ["DOUBAO_API_KEY"],
            api_base=os.environ.get("DOUBAO_BASE_URL"),
        )
        lm.configure()  # 等价于 dspy.configure(lm=...)
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self.model_name = model  # e.g. "openai/deepseek-v4-flash"
        self._api_key = api_key or os.environ.get("DOUBAO_API_KEY", "")
        self._base_url = api_base or os.environ.get("DOUBAO_BASE_URL", "")
        self.temperature = temperature
        self.max_tokens = max_tokens

        # LiteLLM（DSPy 3.x 后端）使用 api_base 参数来指定自定义 endpoint
        self._lm = dspy.LM(
            model=model,
            api_key=self._api_key,
            api_base=self._base_url,  # 注意：不是 base_url
            temperature=temperature,
        )

    def configure(self) -> None:
        """将包装的 LM 注册为 DSPy 全局默认 LM"""
        dspy.configure(lm=self._lm)

    def __getattr__(self, name: str):
        """
        属性访问代理。

        将未定义的属性访问转发给底层的 dspy.LM 对象，
        这样 DSPyLMWrapper 可以像原生 LM 一样使用。
        """
        return getattr(self._lm, name)
