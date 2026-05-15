# -*- coding: utf-8 -*-
"""
DSPy DYPS RAG 提示词优化 — 配置中心
=====================================

本模块集中管理所有配置项，覆盖以下维度：
  - 数据路径（Golden 数据集、RAG 配置、输出目录）
  - 教师模型配置（deepseek-v4-flash API，用于评分和生成 hints）
  - 学生模型配置（本地 HybridRAGPipeline：BM25 + Milvus + BGE Reranker + Qwen3-8B vLLM）
  - DSPy LM 配置（包装 API 模型供 DSPy 内部使用，如 bootstrap demos）
  - 评测配置（DeepEval 三维指标阈值、并发数、重试次数）
  - DYPS 优化超参数（trials 数量、冷启动样本数、教师胜率阈值等）

使用 dataclass 实现，支持默认值 + 运行时覆盖。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# 数据路径（硬编码，指向项目固定位置）
# ═══════════════════════════════════════════════════════════════════════════════

# Golden 数据集：由 build_golden_data.py 从原始 QA 对构建，每条包含 input/expected_output/retrieval_context
GOLDEN_PATH = Path("/root/autodl-tmp/rag-prompt-engineering-lab-upload/data/golden/goldens.jsonl")

# RAG 项目配置文件（包含 DOUBAO_API_KEY / DOUBAO_BASE_URL / DOUBAO_MODEL_NAME 等环境变量）
RAG_CONFIG_PATH = Path("/root/autodl-tmp/RAG/config.ini")

# DYPS 优化结果输出目录
OUTPUT_DIR = Path("/root/autodl-tmp/rag-prompt-engineering-lab-upload/data/dyps")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 教师模型配置
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TeacherConfig:
    """
    教师模型（Teacher）配置。

    教师是更强的 API 模型（deepseek-v4-flash），在 DYPS 中承担三个角色：
    1. 生成参考答案（供 BootstrapDemonstrator 使用）
    2. 评分学生答案（TeacherScorer / grade 方法）
    3. 生成改进 hints（HintGenerator）

    使用 temperature=0.0 确保教师输出稳定、可复现。
    """
    model: str = "deepseek-v4-flash"       # API 模型名
    api_key_env: str = "DOUBAO_API_KEY"     # 从哪个环境变量读取 API Key
    base_url_env: str = "DOUBAO_BASE_URL"   # 从哪个环境变量读取 Base URL
    temperature: float = 0.0                # 生成温度（0 = 确定性输出）
    max_tokens: int = 2048                  # 最大输出 token 数

    @property
    def api_key(self) -> str:
        """从环境变量动态读取 API Key（不在配置文件中硬编码敏感信息）"""
        import os
        return os.environ[self.api_key_env]

    @property
    def base_url(self) -> str:
        """从环境变量动态读取 Base URL"""
        import os
        return os.environ.get(self.base_url_env, "")


# ═══════════════════════════════════════════════════════════════════════════════
# 学生模型配置
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StudentConfig:
    """
    学生模型（Student）配置。

    学生是本地 HybridRAGPipeline，包含：
      - BM25 关键词检索（bm25_topk）
      - Milvus 向量检索（milvus_topk）
      - BGE Reranker 重排序（rerank_topk）
      - Qwen3-8B vLLM 本地推理

    这些 topk 参数控制检索阶段的召回量，直接影响最终答案质量。
    """
    bm25_topk: int = 5       # BM25 关键词检索返回 top-k 文档数
    milvus_topk: int = 10    # Milvus 向量检索返回 top-k 文档数
    rerank_topk: int = 5     # BGE Reranker 重排序后保留的文档数


# ═══════════════════════════════════════════════════════════════════════════════
# DSPy LM 包装器配置
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DSPyLMConfig:
    """
    DSPy 语言模型配置。

    DSPy 内部需要 LM 来执行 bootstrap demos、ChainOfThought 等操作。
    使用 "openai/<model>" 前缀让 LiteLLM（DSPy 3.x 后端）识别路由。
    """
    model: str = "openai/deepseek-v4-flash"  # LiteLLM 格式的模型标识
    api_key_env: str = "DOUBAO_API_KEY"
    base_url_env: str = "DOUBAO_BASE_URL"
    temperature: float = 0.0                 # DSPy 操作使用确定性输出


# ═══════════════════════════════════════════════════════════════════════════════
# 评测配置
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalConfig:
    """
    DeepEval 评测配置。

    三维指标及其阈值：
      - AnswerRelevancy (0.5)：答案是否切题
      - Faithfulness (0.5)：答案是否忠实于检索上下文，无幻觉
      - ContextualRecall (0.6)：答案是否覆盖预期答案的关键信息

    注意：AnswerRelevancy 对"正确输出无答案"场景存在评分盲点，
    可能给 0 分，这是指标设计问题而非提示词质量问题。
    """
    max_concurrency: int = 1                # 异步评测并发数（建议 1，降低非法 JSON 概率）
    max_retries: int = 3                    # 每个 metric 失败后最大重试次数
    thresholds: dict = field(default_factory=lambda: {
        "answer_relevancy": 0.5,
        "faithfulness": 0.5,
        "contextual_recall": 0.6,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# DYPS 优化超参数
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DypsConfig:
    """
    DYPS（Dynamic Prompt Selection）优化超参数。

    核心概念：
      - Cold Start：用少量样本让教师生成初始 bootstrap demos，建立评分基线
      - Trials：每轮 trial 生成新参数 → 学生推理 → DeepEval 评分 → 教师反馈
      - Teacher Win Rate：教师能显著优于学生的样本比例，低值表示学生已足够好
      - Early Stopping：当 teacher_win_rate < (1 - threshold) 时提前终止

    评分权重设计理由：
      - Faithfulness (0.4) 权重最高：幻觉是 RAG 最致命的问题
      - AnswerRelevancy (0.3)：切题性同样重要
      - ContextualRecall (0.3)：信息覆盖度
    """
    # 提示词相关
    initial_prompt_path: Optional[Path] = None  # 自定义初始提示词文件路径
    max_prompt_length: int = 4096               # system_prompt 最大长度（字符）

    # 优化循环
    num_trials: int = 50                         # 最大优化轮数
    num_threads: int = 4                         # 并行线程数（预留）
    max_bootstrapped_demos: int = 4              # 每轮保留的 bootstrap 示例数
    max_labeled_demos: int = 8                   # 每轮保留的 labeled 示例数

    # 评分与收敛
    metric_name: str = "composite"               # 综合评分指标名
    teacher_win_rate_threshold: float = 0.7      # 教师胜率阈值（高于此值继续优化）
    min_improvement_threshold: float = 0.01      # 最小提升阈值（低于此值视为无改进）

    # 冷启动
    cold_start_num_samples: int = 10             # 冷启动使用的训练样本数

    # 综合评分权重
    score_weights: dict = field(default_factory=lambda: {
        "answer_relevancy": 0.3,
        "faithfulness": 0.4,
        "contextual_recall": 0.3,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# 汇总配置容器
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """
    顶层配置聚合类。

    将所有子配置汇总到一个对象中，方便传递和访问。
    所有子配置都有默认值，可以按需覆盖。
    """
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    dspy_lm: DSPyLMConfig = field(default_factory=DSPyLMConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    dyps: DypsConfig = field(default_factory=DypsConfig)
