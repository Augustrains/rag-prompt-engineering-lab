# -*- coding: utf-8 -*-
"""
Main script for DSPy DYPS RAG Prompt Optimization.

Usage:
    # Basic usage (uses defaults)
    python run_dyps.py

    # With custom settings
    python run_dyps.py \
        --num-trials 30 \
        --train-ratio 0.7 \
        --dev-ratio 0.15 \
        --output-dir ./data/dyps \
        --cold-start 10

    # With teacher model override
    python run_dyps.py \
        --teacher-model deepseek-v4-flash \
        --num-trials 50

Workflow:
    1. Load golden dataset
    2. Split into train / dev / test
    3. Configure DSPy LM (teacher model)
    4. Create student RAG module
    5. Cold start: bootstrap demonstrations
    6. Run DYPS optimization trials
    7. Evaluate best params on test set
    8. Save results
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple
from tqdm import tqdm

# Ensure RAG project is on path
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


# ──────────────────────────── 参数解析 ────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DSPy DYPS RAG Prompt Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--golden-path",
        type=Path,
        default=GOLDEN_PATH,
        help="Path to golden dataset JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for results",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.15,
        help="Development set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Model
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="openai/deepseek-v4-flash",
        help="Teacher model name for DSPy (default: openai/deepseek-v4-flash)",
    )
    parser.add_argument(
        "--student-bm25-topk",
        type=int,
        default=5,
        help="BM25 top-k for student RAG (default: 5)",
    )
    parser.add_argument(
        "--student-milvus-topk",
        type=int,
        default=10,
        help="Milvus top-k for student RAG (default: 10)",
    )
    parser.add_argument(
        "--student-rerank-topk",
        type=int,
        default=5,
        help="Rerank top-k for student RAG (default: 5)",
    )

    # Optimization
    parser.add_argument(
        "--num-trials",
        type=int,
        default=30,
        help="Number of optimization trials (default: 30)",
    )
    parser.add_argument(
        "--cold-start",
        type=int,
        default=10,
        help="Number of cold start examples (default: 10)",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=8,
        help="Maximum number of bootstrap demos (default: 8)",
    )
    parser.add_argument(
        "--teacher-win-threshold",
        type=float,
        default=0.7,
        help="Teacher win rate threshold for early stopping (default: 0.7)",
    )
    parser.add_argument(
        "--initial-prompt",
        type=str,
        default=None,
        help="Initial system prompt text. If not set, uses a built-in default.",
    )

    # Evaluation
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Max concurrency for evaluation (default: 1)",
    )

    # DSPy
    parser.add_argument(
        "--dspy-model",
        type=str,
        default="openai/deepseek-v4-flash",
        help="Model for DSPy LM configuration (default: openai/deepseek-v4-flash)",
    )
    parser.add_argument(
        "--dspy-temperature",
        type=float,
        default=0.0,
        help="Temperature for DSPy LM (default: 0.0)",
    )

    # Misc
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print configuration without running",
    )
    parser.add_argument(
        "--skip-test-eval",
        action="store_true",
        help="Skip test set evaluation",
    )

    return parser.parse_args()


# ──────────────────────────── 主流程 ────────────────────────────

def setup_environment() -> None:
    """Setup environment variables and validate configuration."""
    # Check required env vars
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

    # Verify API key is available
    if not os.environ.get("DOUBAO_API_KEY"):
        raise ValueError("DOUBAO_API_KEY not found. Set it in config.ini or environment.")

    # LiteLLM (DSPy 3.x backend) uses OPENAI_API_KEY / OPENAI_BASE_URL for openai/* model prefix
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["DOUBAO_API_KEY"]
    if not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = os.environ["DOUBAO_BASE_URL"]


def load_dataset(args: argparse.Namespace) -> GoldenDataset:
    """Load and split the golden dataset."""
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


def setup_models(args: argparse.Namespace) -> Tuple[StudentRAG, TeacherLM, DeepEvalJudge, DSPyLMWrapper]:
    """Setup student, teacher, judge and DSPy LM."""
    print("\n[Model] Setting up models...")

    # Student RAG (to be optimized)
    student_config = {
        "bm25_topk": args.student_bm25_topk,
        "milvus_topk": args.student_milvus_topk,
        "rerank_topk": args.student_rerank_topk,
    }
    student = StudentRAG(config=student_config)
    print(f"[Model] Student RAG: HybridRAGPipeline (bm25={args.student_bm25_topk}, milvus={args.student_milvus_topk}, rerank={args.student_rerank_topk})")

    # Teacher LM (strong API model)
    env = load_env_config(Path("/root/autodl-tmp/RAG/config.ini"))
    teacher = TeacherLM(
        model=env.get("DOUBAO_MODEL_NAME", "deepseek-v4-flash"),
        api_key=env.get("DOUBAO_API_KEY", ""),
        base_url=env.get("DOUBAO_BASE_URL", ""),
        temperature=0.0,
    )
    print(f"[Model] Teacher LM: {env.get('DOUBAO_MODEL_NAME')}")

    # DeepEval Judge
    judge = build_judge(Path("/root/autodl-tmp/RAG/config.ini"))
    print(f"[Model] DeepEval Judge: {judge.get_model_name()}")

    # DSPy LM (used by DSPy for its internal operations)
    dspy_lm = DSPyLMWrapper(
        model=args.dspy_model,
        api_key=os.environ.get("DOUBAO_API_KEY"),
        api_base=os.environ.get("DOUBAO_BASE_URL"),
        temperature=args.dspy_temperature,
    )
    print(f"[Model] DSPy LM: {args.dspy_model}")

    return student, teacher, judge, dspy_lm


def run_cold_start(
    optimizer: DYPSOptimizer,
    train_examples: list,
    cold_start: int,
) -> None:
    """Run cold start to bootstrap demonstrations."""
    print(f"\n[Bootstrap] Cold start with {cold_start} examples...")
    start = time.time()

    cold_examples = train_examples[:cold_start]
    demos = optimizer.cold_start(cold_examples)

    print(f"[Bootstrap] Cold start complete in {time.time() - start:.1f}s")
    print(f"[Bootstrap] Generated {len(demos)} demos")
    positive = sum(1 for d in demos if d.is_positive)
    print(f"[Bootstrap] Positive demos: {positive}/{len(demos)}")

    for i, demo in enumerate(demos[:3]):
        print(f"\n  Demo {i+1}: score={demo.score:.3f}, positive={demo.is_positive}")
        print(f"    Q: {demo.input[:60]}...")
        print(f"    A: {demo.teacher_answer[:80]}...")


async def run_optimization(
    optimizer: DYPSOptimizer,
    train_examples: list,
    dev_examples: list,
    args: argparse.Namespace,
):
    """Run the main optimization loop."""
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


async def evaluate_on_test(
    judge: DeepEvalJudge,
    student: StudentRAG,
    test_examples: list,
    args: argparse.Namespace,
    output_dir: Path,
) -> list:
    """Evaluate best model on test set."""
    if args.skip_test_eval:
        print("\n[Test] Skipping test evaluation (--skip-test-eval)")
        return []

    print(f"\n[Test] Evaluating on {len(test_examples)} test examples...")
    start = time.time()

    # Generate student answers
    test_records = []
    for ex in tqdm(test_examples, desc="Generating test answers"):
        retrieval_context = []
        try:
            pred = student(
                input=ex.input,
                retrieval_context=ex.retrieval_context or [],
            )
            actual = pred.answer
            # Use student RAG's actual retrieved context for evaluation
            retrieval_context = pred.retrieval_context or []
        except Exception as e:
            print(f"[WARN] Test inference failed for {getattr(ex, 'unique_id', '?')}: {e}")
            actual = ""

        test_records.append({
            "unique_id": getattr(ex, "unique_id", ""),
            "category": getattr(ex, "category", ""),
            "input": ex.input,
            "expected_output": ex.expected_output or "",
            "actual_output": actual,
            "retrieval_context": retrieval_context,
        })

    # Evaluate with DeepEval
    evaluator = Evaluator(
        judge,
        max_concurrency=args.max_concurrency,
    )
    eval_results = await evaluator.evaluate(test_records, desc="Test Evaluation")

    # Save results
    output_path = output_dir / "test_eval_results.jsonl"
    save_jsonl(output_path, [r.to_dict() for r in eval_results])

    # Print summary
    summary = evaluator.print_summary(eval_results)

    print(f"\n[Test] Evaluation complete in {time.time() - start:.1f}s")
    print(f"[Test] Results saved to: {output_path}")

    return eval_results


def save_summary(
    args: argparse.Namespace,
    dataset_stats: dict,
    result: "OptimizationResult",
    test_summary: dict,
    output_dir: Path,
) -> Path:
    """Save a summary of the full run."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "dataset_stats": dataset_stats,
        "best_score": result.best_score,
        "best_params": result.best_prompt_params,
        "num_trials": len(result.trials),
        "history": result.history[-10:],  # Last 10 trials
        "test_summary": test_summary,
    }

    output_path = output_dir / "run_summary.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[Output] Summary saved to: {output_path}")
    return output_path


def main() -> None:
    args = parse_args()

    # Setup
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

    # Load data
    dataset, train, dev, test = load_dataset(args)

    # Setup models
    student, teacher, judge, dspy_lm = setup_models(args)

    # Configure DSPy
    dspy_lm.configure()
    print(f"\n[DSPy] Configured with model: {args.dspy_model}")

    # Create optimizer
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

    # Run cold start
    run_cold_start(optimizer, train, args.cold_start)

    # Run optimization
    result = asyncio.run(
        run_optimization(optimizer, train, dev, args)
    )

    # Evaluate on test set
    test_results = asyncio.run(
        evaluate_on_test(judge, student, test, args, args.output_dir)
    )

    test_summary = {}
    if test_results:
        test_summary = {
            "overall": sum(r.overall_score for r in test_results) / len(test_results),
            "answer_relevancy": sum(r.answer_relevancy for r in test_results) / len(test_results),
            "faithfulness": sum(r.faithfulness for r in test_results) / len(test_results),
            "contextual_recall": sum(r.contextual_recall for r in test_results) / len(test_results),
        }

    # Save summary
    save_summary(
        args=args,
        dataset_stats=dataset.stats(),
        result=result,
        test_summary=test_summary,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("DYPS Optimization Complete!")
    print("=" * 60)
    print(f"Best score (dev): {result.best_score:.4f}")
    best = result.best_prompt_params or {}
    print(f"Best temperature: {best.get('temperature', 'N/A')}")
    print(f"Best max_tokens: {best.get('max_tokens', 'N/A')}")
    best_prompt = best.get('system_prompt', '')
    if best_prompt:
        truncated = best_prompt[:120] + "..." if len(best_prompt) > 120 else best_prompt
        print(f"Best system_prompt: {truncated}")
        print(f"  (full prompt in run_summary.json)")
    if test_summary:
        print(f"Test overall: {test_summary.get('overall', 0):.4f}")


if __name__ == "__main__":
    main()
