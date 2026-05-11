# -*- coding: utf-8 -*-
"""
Configuration for DSPy DYPS RAG prompt optimization.

This module centralizes all configuration needed for:
- Student/Teacher model settings
- Data paths
- Evaluation parameters
- Optimization hyperparameters
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ──────────────────────────── 数据路径 ────────────────────────────

GOLDEN_PATH = Path("/root/autodl-tmp/rag-prompt-engineering-lab-upload/data/golden/goldens.jsonl")

# RAG 模型（学生）配置：从 RAG 项目读取
RAG_CONFIG_PATH = Path("/root/autodl-tmp/RAG/config.ini")

# 输出路径
OUTPUT_DIR = Path("/root/autodl-tmp/rag-prompt-engineering-lab-upload/data/dyps")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────── 教师模型（API 调用） ────────────────────────────

@dataclass
class TeacherConfig:
    """Teacher model configuration (stronger model via API)."""
    model: str = "deepseek-v4-flash"           # 模型名
    api_key_env: str = "DOUBAO_API_KEY"         # API Key 环境变量
    base_url_env: str = "DOUBAO_BASE_URL"       # Base URL 环境变量
    temperature: float = 0.0                   # 生成温度
    max_tokens: int = 2048                      # 最大 token 数

    @property
    def api_key(self) -> str:
        import os
        return os.environ[self.api_key_env]

    @property
    def base_url(self) -> str:
        import os
        return os.environ.get(self.base_url_env, "")


# ──────────────────────────── 学生模型（本地 RAG） ────────────────────────────

@dataclass
class StudentConfig:
    """Student model configuration (local RAG pipeline)."""
    # RAG pipeline 使用本地 vLLM 模型，无需额外配置
    bm25_topk: int = 5
    milvus_topk: int = 10
    rerank_topk: int = 5


# ──────────────────────────── DSPy LM 包装器 ────────────────────────────

@dataclass
class DSPyLMConfig:
    """DSPy language model configuration."""
    model: str = "openai/deepseek-v4-flash"
    api_key_env: str = "DOUBAO_API_KEY"
    base_url_env: str = "DOUBAO_BASE_URL"
    temperature: float = 0.0


# ──────────────────────────── 评测配置 ────────────────────────────

@dataclass
class EvalConfig:
    """Evaluation configuration."""
    max_concurrency: int = 1          # 并发数
    max_retries: int = 3             # 重试次数
    thresholds: dict = field(default_factory=lambda: {
        "answer_relevancy": 0.5,
        "faithfulness": 0.5,
        "contextual_recall": 0.6,
    })


# ──────────────────────────── DYPS 优化配置 ────────────────────────────

@dataclass
class DypsConfig:
    """DSPy DYPS optimization configuration."""
    # 提示词相关
    initial_prompt_path: Optional[Path] = None  # 如果指定，则从文件加载初始提示词
    max_prompt_length: int = 4096                # 提示词最大长度

    # 优化相关
    num_trials: int = 50                         # 并行 trials 数量
    num_threads: int = 4                         # 并行线程数
    max_bootstrapped_demos: int = 4              # 最多保留多少个 bootstrap demos
    max_labeled_demos: int = 8                   # 最多保留多少个 labeled demos

    # 评分相关
    metric_name: str = "composite"               # 评分指标名
    teacher_win_rate_threshold: float = 0.7      # 教师胜率阈值（达到后停止优化）
    min_improvement_threshold: float = 0.01      # 最小提升阈值

    # 冷启动
    cold_start_num_samples: int = 10             # 用于冷启动的样本数

    # 评分权重
    score_weights: dict = field(default_factory=lambda: {
        "answer_relevancy": 0.3,
        "faithfulness": 0.4,
        "contextual_recall": 0.3,
    })


# ──────────────────────────── 汇总配置 ────────────────────────────

@dataclass
class Config:
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    dspy_lm: DSPyLMConfig = field(default_factory=DSPyLMConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    dyps: DypsConfig = field(default_factory=DypsConfig)
