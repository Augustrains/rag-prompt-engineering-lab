# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# 版权所有  @丁师兄大模型
# --------------------------------------------

"""Unified pipeline for Golden creation, RAG filling, and DeepEval scoring."""

import argparse
import json
from pathlib import Path
from typing import Optional

from tqdm import tqdm


DEFAULT_RAW_INPUT = "data/qa_pairs/test_qa_pair_verify.json"
DEFAULT_GOLDEN_PATH = "data/golden/goldens.jsonl"
DEFAULT_FILLED_PATH = "data/golden/goldens_filled.jsonl"
DEFAULT_SCORE_PATH = "data/golden/goldens_deepeval_scores.jsonl"
DEFAULT_SUMMARY_PATH = "data/golden/goldens_deepeval_summary.json"
DEFAULT_CONFIG_PATH = "config.ini"


def split_output_path(path: str) -> tuple[str, str]:
    output_path = Path(path)
    if output_path.suffix != ".jsonl":
        raise ValueError("Golden output path must end with .jsonl")
    return str(output_path.parent), output_path.stem


def build_goldens(
    input_path: str,
    output_path: str,
    limit: Optional[int],
) -> str:
    from build_golden_data import build_dataset, load_records

    source_path = Path(input_path)
    output_dir, output_name = split_output_path(output_path)
    records = load_records(source_path)
    dataset = build_dataset(records, source_path, limit)
    saved_path = dataset.save_as("jsonl", directory=output_dir, file_name=output_name)
    print(f"Golden data size: {len(dataset.goldens)}")
    print(f"Golden data saved to: {saved_path}")
    return saved_path


def fill_goldens(
    golden_path: str,
    output_path: str,
    limit: Optional[int],
    offset: int,
    bm25_topk: int,
    milvus_topk: int,
    rerank_topk: int,
) -> str:
    from run_golden_rag import load_jsonl, prediction_record, select_records
    from src.utils import HybridRAGConfig, HybridRAGPipeline

    records = load_jsonl(Path(golden_path))
    selected_records = list(select_records(records, limit, offset))
    if not selected_records:
        raise ValueError("No Golden records selected.")

    pipeline = HybridRAGPipeline(
        HybridRAGConfig(
            bm25_topk=bm25_topk,
            milvus_topk=milvus_topk,
            rerank_topk=rerank_topk,
        )
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fw:
        for golden in tqdm(selected_records, desc="Running RAG"):
            rag_result = pipeline.answer(golden["input"].strip())
            record = prediction_record(golden, rag_result)
            fw.write(json.dumps(record, ensure_ascii=False) + "\n")
            fw.flush()

    print(f"Filled Golden data saved to: {output}")
    print(f"Filled Golden size: {len(selected_records)}")
    return str(output)


def evaluate_goldens(
    input_path: str,
    output_path: str,
    summary_path: str,
    config_path: str,
    limit: Optional[int],
) -> tuple[str, str]:
    from evaluate_golden_deepeval import (
        build_judge_model,
        build_metrics,
        evaluate_record,
        load_jsonl,
        write_jsonl,
        write_summary,
    )

    records = load_jsonl(Path(input_path))
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError("No filled Golden records selected.")

    model = build_judge_model(Path(config_path))
    metrics = build_metrics(model)

    results = []
    for record in tqdm(records, desc="DeepEval"):
        results.append(evaluate_record(record, metrics))

    write_jsonl(Path(output_path), results)
    write_summary(Path(summary_path), results)
    print(f"Evaluation results saved to: {output_path}")
    print(f"Evaluation summary saved to: {summary_path}")
    return output_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Golden build, RAG filling, and DeepEval scoring as reusable stages."
    )
    parser.add_argument(
        "--stage",
        choices=("build", "fill", "evaluate", "all"),
        default="all",
        help="Pipeline stage to run.",
    )
    parser.add_argument("--raw-input", default=DEFAULT_RAW_INPUT, help="Raw evaluation set path.")
    parser.add_argument("--golden-path", default=DEFAULT_GOLDEN_PATH, help="Golden JSONL path.")
    parser.add_argument("--filled-path", default=DEFAULT_FILLED_PATH, help="Filled Golden JSONL path.")
    parser.add_argument("--score-path", default=DEFAULT_SCORE_PATH, help="DeepEval score JSONL path.")
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH, help="DeepEval summary JSON path.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="config.ini path for judge model.")
    parser.add_argument("--build-limit", type=int, default=None, help="Limit records when building Goldens.")
    parser.add_argument("--fill-limit", type=int, default=None, help="Limit records when filling Goldens.")
    parser.add_argument("--eval-limit", type=int, default=None, help="Limit records when evaluating.")
    parser.add_argument("--offset", type=int, default=0, help="Skip N records during filling.")
    parser.add_argument("--bm25-topk", type=int, default=5)
    parser.add_argument("--milvus-topk", type=int, default=10)
    parser.add_argument("--rerank-topk", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    golden_path = args.golden_path
    filled_path = args.filled_path

    if args.stage in ("build", "all"):
        golden_path = build_goldens(
            input_path=args.raw_input,
            output_path=args.golden_path,
            limit=args.build_limit,
        )

    if args.stage in ("fill", "all"):
        filled_path = fill_goldens(
            golden_path=golden_path,
            output_path=args.filled_path,
            limit=args.fill_limit,
            offset=args.offset,
            bm25_topk=args.bm25_topk,
            milvus_topk=args.milvus_topk,
            rerank_topk=args.rerank_topk,
        )

    if args.stage in ("evaluate", "all"):
        evaluate_goldens(
            input_path=filled_path,
            output_path=args.score_path,
            summary_path=args.summary_path,
            config_path=args.config,
            limit=args.eval_limit,
        )


if __name__ == "__main__":
    main()
