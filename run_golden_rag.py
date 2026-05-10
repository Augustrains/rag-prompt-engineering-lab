# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# 版权所有  @丁师兄大模型
# --------------------------------------------

"""Run the existing RAG pipeline on DeepEval Golden data."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tqdm import tqdm

from src.utils import HybridRAGConfig, HybridRAGPipeline


DEFAULT_GOLDEN_PATH = "data/golden/goldens.jsonl"
DEFAULT_OUTPUT_PATH = "data/golden/golden_rag_predictions.jsonl"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_no} must be a JSON object.")
            records.append(record)
    return records


def select_records(records: List[Dict[str, Any]], limit: Optional[int], offset: int) -> Iterable[Dict[str, Any]]:
    selected = records[offset:]
    if limit is not None:
        selected = selected[:limit]
    return selected


def prediction_record(golden: Dict[str, Any], rag_result: Dict[str, Any]) -> Dict[str, Any]:
    pred_answer = rag_result["answer"]
    metadata = golden.get("additional_metadata") or {}
    return {
        "id": metadata.get("unique_id") or metadata.get("id"),
        "input": golden["input"],
        "expected_output": golden.get("expected_output"),
        "actual_output": pred_answer["answer"],
        "retrieval_context": rag_result["retrieval_context"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid RAG predictions on Golden data.")
    parser.add_argument("--golden-path", default=DEFAULT_GOLDEN_PATH, help="Input Golden JSONL path.")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, help="Output prediction JSONL path.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N records after offset.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N records.")
    parser.add_argument("--bm25-topk", type=int, default=5)
    parser.add_argument("--milvus-topk", type=int, default=10)
    parser.add_argument("--rerank-topk", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    golden_path = Path(args.golden_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    goldens = load_jsonl(golden_path)
    selected_goldens = list(select_records(goldens, args.limit, args.offset))
    if not selected_goldens:
        raise ValueError("No Golden records selected.")

    pipeline = HybridRAGPipeline(
        HybridRAGConfig(
            bm25_topk=args.bm25_topk,
            milvus_topk=args.milvus_topk,
            rerank_topk=args.rerank_topk,
        )
    )

    with output_path.open("w", encoding="utf-8") as fw:
        for golden in tqdm(selected_goldens, desc="Running RAG"):
            query = golden["input"].strip()
            rag_result = pipeline.answer(query)
            record = prediction_record(golden, rag_result)
            fw.write(json.dumps(record, ensure_ascii=False) + "\n")
            fw.flush()

    print(f"Predictions saved to: {output_path}")
    print(f"Prediction size: {len(selected_goldens)}")


if __name__ == "__main__":
    main()
