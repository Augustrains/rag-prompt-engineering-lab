# -*- coding: utf-8 -*-
"""
DSPy DYPS RAG Prompt Optimization Package.

三个模型架构：
- StudentRAG：本地 RAG pipeline wrapper（BM25 + Milvus + BGE Reranker + 本地 Qwen3-8B vLLM）
- TeacherLM：API 模型（deepseek-v4-flash），用于生成参考答案、评分、生成 hints
- DSPyLMWrapper：API 模型封装，供 DSPy 内部操作使用（如 bootstrap demos）
"""

from .config import (
    GOLDEN_PATH,
    OUTPUT_DIR,
    TeacherConfig,
    StudentConfig,
    DSPyLMConfig,
    EvalConfig,
    DypsConfig,
    Config,
)
from .data import GoldenDataset, GoldenRecord, load_goldens
from .models import StudentRAG, TeacherLM, DSPyLMWrapper
from .signatures import (
    RAGSignature,
    TeacherGradingSignature,
    HintGenerationSignature,
    CompositeRAGSignature,
)
from .evaluator import (
    DeepEvalJudge,
    EvalResult,
    Evaluator,
    build_judge,
    load_env_config,
)
from .teleprompter import (
    DemoRecord,
    OptimizationTrial,
    OptimizationResult,
    TeacherScorer,
    BootstrapDemonstrator,
    HintGenerator,
    DYPSOptimizer,
)

__all__ = [
    # config
    "GOLDEN_PATH",
    "OUTPUT_DIR",
    "TeacherConfig",
    "StudentConfig",
    "DSPyLMConfig",
    "EvalConfig",
    "DypsConfig",
    "Config",
    # data
    "GoldenDataset",
    "GoldenRecord",
    "load_goldens",
    # models
    "StudentRAG",
    "TeacherLM",
    "DSPyLMWrapper",
    # signatures
    "RAGSignature",
    "TeacherGradingSignature",
    "HintGenerationSignature",
    "CompositeRAGSignature",
    # evaluator
    "DeepEvalJudge",
    "EvalResult",
    "Evaluator",
    "build_judge",
    "load_env_config",
    # teleprompter
    "DemoRecord",
    "OptimizationTrial",
    "OptimizationResult",
    "TeacherScorer",
    "BootstrapDemonstrator",
    "HintGenerator",
    "DYPSOptimizer",
]
