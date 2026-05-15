# -*- coding: utf-8 -*-
"""
DSPy DYPS RAG 提示词优化 — 主入口脚本
======================================

本脚本编排完整的 DYPS 优化流程：

  setup_environment()           → 加载 config.ini 环境变量
  load_dataset()                → GoldenDataset + 分层切分 train/dev/test
  setup_models()                → StudentRAG + TeacherLM + DeepEvalJudge + DSPyLMWrapper
  run_cold_start()              → BootstrapDemonstrator 生成初始 demo 池
  run_optimization()            → DYPSOptimizer.optimize_async() 迭代优化
  evaluate_on_test()            → 在 test 集上用最佳参数评测
  save_summary()                → 输出 run_summary.json

使用示例：
    # 默认参数
    python run_dyps.py

    # 自定义优化参数
    python run_dyps.py --num-trials 50 --cold-start 10 --train-ratio 0.7

    # 跳过测试集评测（节省 token）
    python run_dyps.py --skip-test-eval

    # 仅验证配置
    python run_dyps.py --dry-run

环境要求：
    - conda 环境 "rag"（Python 3.12）
    - /root/autodl-tmp/RAG/config.ini 中配置 DOUBAO_API_KEY / DOUBAO_BASE_URL
    - 本地 vLLM + Qwen3-8B 服务运行中
    - MongoDB + Milvus 运行中（供 HybridRAGPipeline 检索）
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm

# 将外部 RAG 项目加入 sys.path
RAG_ROOT = Path("/root/autodl-tmp/RAG")
if str(RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_ROOT))

import dspy

from dyps.config import (
    GOLDEN_PATH,
    OUTPUT_DIR,
    DSPyLMConfig,
    DypsConfig,
    EvalConfig,
    StudentConfig,
    TeacherConfig,
)
from dyps.data import GoldenDataset, save_jsonl
from dyps.evaluator import DeepEvalJudge, Evaluator, build_judge, load_env_config
from dyps.models import DSPyLMWrapper, StudentRAG, TeacherLM
from dyps.signatures import RAGSignature
from dyps.teleprompter import DYPSOptimizer


# ═══════════════════════════════════════════════════════════════════════════════
# 命令行参数解析
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    参数分为五组：
      - Data: 数据集路径、切分比例、随机种子
      - Model: 教师模型、学生 RAG 的 topk 参数、DSPy LM 配置
      - Optimization: trials 数量、冷启动、max_demos、早停阈值
      - Evaluation: 评测并发数
      - Misc: dry-run、skip-test-eval
    """
    parser = argparse.ArgumentParser(
        description="DSPy DYPS RAG Prompt Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── 数据参数 ──
    parser.add_argument(
        "--golden-path", type=Path, default=GOLDEN_PATH,
        help="Golden 数据集 JSONL 文件路径",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="结果输出目录",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.7,
        help="训练集比例（默认 0.7）",
    )
    parser.add_argument(
        "--dev-ratio", type=float, default=0.15,
        help="开发集比例（默认 0.15），测试集 = 1 - train - dev",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（保证可复现）",
    )

    # ── 模型参数 ──
    parser.add_argument(
        "--teacher-model", type=str, default="openai/deepseek-v4-flash",
        help="教师模型名（DSPy 格式，如 openai/deepseek-v4-flash）",
    )
    parser.add_argument(
        "--student-bm25-topk", type=int, default=5,
        help="BM25 关键词检索 top-k",
    )
    parser.add_argument(
        "--student-milvus-topk", type=int, default=10,
        help="Milvus 向量检索 top-k",
    )
    parser.add_argument(
        "--student-rerank-topk", type=int, default=5,
        help="BGE Reranker 重排序后保留 top-k",
    )

    # ── 优化参数 ──
    parser.add_argument(
        "--num-trials", type=int, default=30,
        help="优化 trial 数量（默认 30）",
    )
    parser.add_argument(
        "--cold-start", type=int, default=10,
        help="冷启动使用的训练样本数（默认 10）",
    )
    parser.add_argument(
        "--max-demos", type=int, default=8,
        help="最大 bootstrap demo 数量（默认 8）",
    )
    parser.add_argument(
        "--teacher-win-threshold", type=float, default=0.7,
        help="教师胜率阈值（低于 1-threshold 时早停）",
    )
    parser.add_argument(
        "--initial-prompt", type=str, default=None,
        help="自定义初始 system_prompt。不指定则使用内置默认值",
    )

    # ── 评测参数 ──
    parser.add_argument(
        "--max-concurrency", type=int, default=1,
        help="评测最大并发数（默认 1，避免 API 限流）",
    )

    # ── DSPy 参数 ──
    parser.add_argument(
        "--dspy-model", type=str, default="openai/deepseek-v4-flash",
        help="DSPy 内部 LM 模型名",
    )
    parser.add_argument(
        "--dspy-temperature", type=float, default=0.0,
        help="DSPy LM 温度（默认 0.0 = 确定性输出）",
    )

    # ── 杂项 ──
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅打印配置并验证环境，不执行实际优化",
    )
    parser.add_argument(
        "--skip-test-eval", action="store_true",
        help="跳过测试集评测（节省 API token）",
    )

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# 环境配置
# ═══════════════════════════════════════════════════════════════════════════════

def setup_environment() -> None:
    """
    配置运行环境。

    1. 检查必要的环境变量（DOUBAO_API_KEY / DOUBAO_BASE_URL / DOUBAO_MODEL_NAME）
    2. 如果缺失，尝试从 /root/autodl-tmp/RAG/config.ini 加载
    3. 同步设置 OPENAI_API_KEY / OPENAI_BASE_URL（供 LiteLLM / DSPy 3.x 后端使用）

    LiteLLM 兼容性说明：
      DSPy 3.x 使用 LiteLLM 作为后端。当模型前缀为 "openai/" 时，
      LiteLLM 会读取 OPENAI_API_KEY 和 OPENAI_BASE_URL 环境变量。
      因此需要将 DOUBAO_* 环境变量同步为 OPENAI_*。
    """
    required = ["DOUBAO_API_KEY", "DOUBAO_BASE_URL", "DOUBAO_MODEL_NAME"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"[WARN] Missing env vars: {missing}")
        print("Trying to load from config.ini...")

        config_path = Path("/root/autodl-tmp/RAG/config.ini")
        if config_path.exists():
            env = load_env_config(config_path)
            for key in missing:
                if key in env:
                    os.environ[key] = env[key]
                    print(f"  Loaded {key} from config.ini")

    # 安全检查：API Key 是调用教师模型和评测的必需条件
    if not os.environ.get("DOUBAO_API_KEY"):
        raise ValueError(
            "DOUBAO_API_KEY not found. "
            "Set it in /root/autodl-tmp/RAG/config.ini or environment."
        )

    # LiteLLM 兼容性：将 DOUBAO 凭据同步为 OPENAI 标准环境变量
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["DOUBAO_API_KEY"]
    if not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = os.environ["DOUBAO_BASE_URL"]


# ═══════════════════════════════════════════════════════════════════════════════
# 数据集加载
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(args: argparse.Namespace) -> GoldenDataset:
    """
    加载 Golden 数据集并切分。

    返回 (dataset, train, dev, test) 四元组。
    dataset 保留完整数据集引用，供 optimizer 统计使用。
    """
    print(f"\n[Data] Loading golden dataset from: {args.golden_path}")
    dataset = GoldenDataset(path=args.golden_path)
    print(f"[Data] Loaded {len(dataset)} records")
    print(f"[Data] Stats: {dataset.stats()}")

    train, dev, test = dataset.split(
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )
    print(f"[Data] Split: train={len(train)}, dev={len(dev)}, test={len(test)}")
    return dataset, train, dev, test


# ═══════════════════════════════════════════════════════════════════════════════
# 模型初始化
# ═══════════════════════════════════════════════════════════════════════════════

def setup_models(args: argparse.Namespace) -> Tuple[StudentRAG, TeacherLM, DeepEvalJudge, DSPyLMWrapper]:
    """
    初始化四个模型组件。

    返回：
      - student: 本地 RAG pipeline（被优化的对象）
      - teacher: API 模型（提供监督信号）
      - judge: DeepEval 包装的 API 模型（评测员）
      - dspy_lm: DSPy 原生 LM（供 DSPy 内部操作）

    注意：StudentRAG 初始化时会连接 MongoDB + Milvus + vLLM，
    需要确保这些服务已启动。
    """
    print("\n[Model] Setting up models...")

    # ── 学生模型：本地 RAG Pipeline ──
    student_config = {
        "bm25_topk": args.student_bm25_topk,
        "milvus_topk": args.student_milvus_topk,
        "rerank_topk": args.student_rerank_topk,
    }
    student = StudentRAG(config=student_config)
    print(f"[Model] Student RAG: HybridRAGPipeline "
          f"(bm25={args.student_bm25_topk}, milvus={args.student_milvus_topk}, "
          f"rerank={args.student_rerank_topk})")

    # ── 教师模型：API 调用 ──
    env = load_env_config(Path("/root/autodl-tmp/RAG/config.ini"))
    teacher = TeacherLM(
        model=env.get("DOUBAO_MODEL_NAME", "deepseek-v4-flash"),
        api_key=env.get("DOUBAO_API_KEY", ""),
        base_url=env.get("DOUBAO_BASE_URL", ""),
        temperature=0.0,
    )
    print(f"[Model] Teacher LM: {env.get('DOUBAO_MODEL_NAME')}")

    # ── 评测员：DeepEval Judge ──
    judge = build_judge(Path("/root/autodl-tmp/RAG/config.ini"))
    print(f"[Model] DeepEval Judge: {judge.get_model_name()}")

    # ── DSPy LM：供 DSPy 内部操作 ──
    dspy_lm = DSPyLMWrapper(
        model=args.dspy_model,
        api_key=os.environ.get("DOUBAO_API_KEY"),
        api_base=os.environ.get("DOUBAO_BASE_URL"),
        temperature=args.dspy_temperature,
    )
    print(f"[Model] DSPy LM: {args.dspy_model}")

    return student, teacher, judge, dspy_lm


# ═══════════════════════════════════════════════════════════════════════════════
# 冷启动
# ═══════════════════════════════════════════════════════════════════════════════

def run_cold_start(
    optimizer: DYPSOptimizer,
    train_examples: list,
    cold_start: int,
) -> Dict[str, float]:
    """
    执行冷启动并返回训练基线指标。

    冷启动过程：
      1. 取前 cold_start 条训练样本
      2. BootstrapDemonstrator 对每条样本：
         a. 学生检索 + 生成答案
         b. 教师基于学生检索上下文生成参考答案
         c. DeepEval 评分
      3. 返回 demo 池的基线统计

    基线指标用于在优化后对比改进幅度。
    """
    print(f"\n[Bootstrap] Cold start with {cold_start} examples...")
    start = time.time()

    cold_examples = train_examples[:cold_start]
    demos = optimizer.cold_start(cold_examples)

    elapsed = time.time() - start
    print(f"[Bootstrap] Cold start complete in {elapsed:.1f}s")
    print(f"[Bootstrap] Generated {len(demos)} demos")
    positive = sum(1 for d in demos if d.is_positive)
    print(f"[Bootstrap] Positive demos: {positive}/{len(demos)}")

    # 打印前 3 条 demo 便于人工检查
    for i, demo in enumerate(demos[:3]):
        print(f"\n  Demo {i+1}: score={demo.score:.3f}, positive={demo.is_positive}")
        print(f"    Q: {demo.input[:60]}...")
        print(f"    A: {demo.teacher_answer[:80]}...")

    scores = [d.score for d in demos]
    baseline = {
        "train_baseline_score": sum(scores) / len(scores) if scores else 0.0,
        "train_baseline_positive_rate": positive / len(demos) if demos else 0.0,
        "train_baseline_samples": len(demos),
    }
    print(f"\n[Train Baseline] avg_score={baseline['train_baseline_score']:.4f}, "
          f"positive_rate={baseline['train_baseline_positive_rate']:.2%}")
    return baseline


# ═══════════════════════════════════════════════════════════════════════════════
# 优化主循环
# ═══════════════════════════════════════════════════════════════════════════════

async def run_optimization(
    optimizer: DYPSOptimizer,
    train_examples: list,
    dev_examples: list,
    args: argparse.Namespace,
):
    """
    运行 DYPS 优化主循环。

    调用 optimizer.optimize_async()，在 dev 集上迭代搜索最佳参数。
    返回 OptimizationResult（包含最佳参数、最佳分、trial 历史）。
    """
    print(f"\n[Optimization] Starting {args.num_trials} trials...")
    start = time.time()

    result = await optimizer.optimize_async(
        train_examples=train_examples,
        dev_examples=dev_examples,
        num_trials=args.num_trials,
    )

    elapsed = time.time() - start
    print(f"\n[Optimization] Complete in {elapsed:.1f}s")
    print(f"[Optimization] Best score: {result.best_score:.4f}")
    print(f"[Optimization] Best params: {result.best_prompt_params}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 测试集评测
# ═══════════════════════════════════════════════════════════════════════════════

async def evaluate_on_test(
    judge: DeepEvalJudge,
    student: StudentRAG,
    test_examples: list,
    args: argparse.Namespace,
    output_dir: Path,
) -> list:
    """
    在测试集上用最佳参数评测最终效果。

    流程：
      1. 对每条 test 样本，用 StudentRAG 生成答案（使用默认参数）
      2. 收集所有答案，用 DeepEval 三维指标评测
      3. 保存结果到 test_eval_results.jsonl
      4. 打印汇总统计

    注意：这里使用默认参数而非最佳参数，因为 evaluate_on_test
    在 optimization 之后调用，但 student 模块在 optimization 过程中
    的 prompt_params 是动态传入的，不影响 student 的默认行为。
    如需用最佳参数评测，应额外传入 prompt_params。
    """
    if args.skip_test_eval:
        print("\n[Test] Skipping test evaluation (--skip-test-eval)")
        return []

    print(f"\n[Test] Evaluating on {len(test_examples)} test examples...")
    start = time.time()

    # ── 生成学生答案 ──
    test_records = []
    for ex in tqdm(test_examples, desc="Generating test answers"):
        retrieval_context = []
        try:
            pred = student(
                input=ex.input,
                retrieval_context=ex.retrieval_context or [],
            )
            actual = pred.answer
            # 使用学生实际检索到的文档（非 golden 数据中的 null）
            retrieval_context = pred.retrieval_context or []
        except Exception as e:
            print(f"[WARN] Test inference failed for "
                  f"{getattr(ex, 'unique_id', '?')}: {e}")
            actual = ""

        test_records.append({
            "unique_id": getattr(ex, "unique_id", ""),
            "category": getattr(ex, "category", ""),
            "input": ex.input,
            "expected_output": ex.expected_output or "",
            "actual_output": actual,
            "retrieval_context": retrieval_context,
        })

    # ── DeepEval 评测 ──
    evaluator = Evaluator(judge, max_concurrency=args.max_concurrency)
    eval_results = await evaluator.evaluate(test_records, desc="Test Evaluation")

    # ── 保存结果 ──
    output_path = output_dir / "test_eval_results.jsonl"
    save_jsonl(output_path, [r.to_dict() for r in eval_results])

    # ── 打印汇总 ──
    summary = evaluator.print_summary(eval_results)

    print(f"\n[Test] Evaluation complete in {time.time() - start:.1f}s")
    print(f"[Test] Results saved to: {output_path}")

    return eval_results


# ═══════════════════════════════════════════════════════════════════════════════
# 结果保存
# ═══════════════════════════════════════════════════════════════════════════════

def save_summary(
    args: argparse.Namespace,
    dataset_stats: dict,
    result: "OptimizationResult",
    test_summary: dict,
    train_baseline: dict,
    output_dir: Path,
) -> Path:
    """
    保存完整运行摘要到 run_summary.json。

    内容包含：
      - 时间戳和运行参数
      - 数据集统计
      - 训练基线（冷启动后的 demo 池指标）
      - 最佳 dev 评分和最佳参数
      - trial 总数
      - 最后 10 轮的历史记录（方便查看收敛趋势）
      - 测试集评测汇总
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "args": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
        },
        "dataset_stats": dataset_stats,
        "train_baseline": train_baseline,
        "best_dev_score": result.best_score,
        "best_params": result.best_prompt_params,
        "num_trials": len(result.trials),
        "history": result.history[-10:],  # 仅保留最后 10 轮（避免文件过大）
        "test_summary": test_summary,
    }

    output_path = output_dir / "run_summary.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[Output] Summary saved to: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    主函数 — 编排完整的 DYPS 优化流程。

    执行顺序：
      1. 解析命令行参数
      2. 配置环境（加载 API 凭据）
      3. 加载并切分数据集
      4. 初始化四个模型组件
      5. 配置 DSPy 全局 LM
      6. 创建 DYPSOptimizer
      7. 冷启动 → 优化循环 → 测试集评测
      8. 保存结果摘要
    """
    args = parse_args()

    # 打印运行配置
    print("=" * 60)
    print("DSPy DYPS RAG Prompt Optimization")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Golden path: {args.golden_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Num trials: {args.num_trials}")
    print(f"Teacher win threshold: {args.teacher_win_threshold}")

    setup_environment()

    if args.dry_run:
        print("\n[dry-run] Configuration validated, exiting.")
        return

    # 1. 加载数据
    dataset, train, dev, test = load_dataset(args)

    # 2. 初始化模型
    student, teacher, judge, dspy_lm = setup_models(args)

    # 3. 配置 DSPy 全局 LM
    dspy_lm.configure()
    print(f"\n[DSPy] Configured with model: {args.dspy_model}")

    # 4. 创建优化器
    optimizer_config = {
        "num_trials": args.num_trials,
        "max_demos": args.max_demos,
        "cold_start_samples": args.cold_start,
        "teacher_win_threshold": args.teacher_win_threshold,
        "initial_system_prompt": args.initial_prompt,
    }
    optimizer = DYPSOptimizer(
        student_module=student,
        teacher_module=teacher,
        judge=judge,
        dataset=dataset,
        config=optimizer_config,
    )

    # 5. 冷启动 → 生成初始 demo 池和基线
    train_baseline = run_cold_start(optimizer, train, args.cold_start)

    # 6. 优化循环 → 迭代搜索最佳参数
    result = asyncio.run(
        run_optimization(optimizer, train, dev, args)
    )

    # 7. 测试集评测 → 验证最佳参数泛化能力
    test_results = asyncio.run(
        evaluate_on_test(judge, student, test, args, args.output_dir)
    )

    # 汇总测试集成绩
    test_summary = {}
    if test_results:
        test_summary = {
            "overall": sum(r.overall_score for r in test_results) / len(test_results),
            "answer_relevancy": sum(r.answer_relevancy for r in test_results) / len(test_results),
            "faithfulness": sum(r.faithfulness for r in test_results) / len(test_results),
            "contextual_recall": sum(r.contextual_recall for r in test_results) / len(test_results),
        }

    # 8. 保存完整摘要
    save_summary(
        args=args,
        dataset_stats=dataset.stats(),
        result=result,
        test_summary=test_summary,
        train_baseline=train_baseline,
        output_dir=args.output_dir,
    )

    # 打印最终对比
    print("\n" + "=" * 60)
    print("DYPS Optimization Complete!")
    print("=" * 60)
    print(f"Train baseline (cold start): {train_baseline['train_baseline_score']:.4f}")
    print(f"Best dev score (optimized):   {result.best_score:.4f}")
    if test_summary:
        print(f"Test final score:             {test_summary.get('overall', 0):.4f}")
        improvement = test_summary.get('overall', 0) - train_baseline['train_baseline_score']
        print(f"Improvement (test - train):   {improvement:+.4f}")
    best = result.best_prompt_params or {}
    print(f"\nBest temperature: {best.get('temperature', 'N/A')}")
    print(f"Best max_tokens: {best.get('max_tokens', 'N/A')}")
    best_prompt = best.get('system_prompt', '')
    if best_prompt:
        truncated = best_prompt[:120] + "..." if len(best_prompt) > 120 else best_prompt
        print(f"Best system_prompt: {truncated}")
        print(f"  (full prompt in run_summary.json)")


if __name__ == "__main__":
    main()
