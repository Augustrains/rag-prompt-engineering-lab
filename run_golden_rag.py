"""Run RAG on selected Golden data and save final eval JSONL."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from src.utils import HybridRAGConfig, HybridRAGPipeline


DATA_DIR = Path("/root/autodl-tmp/rag-prompt-engineering-lab-upload/data")
GOLDEN_PATH = DATA_DIR / "golden" / "goldens.jsonl"

CATEGORY_QUOTAS = {
    "充电相关": 4,
    "驾驶操控与底盘": 3,
    "辅助驾驶与智能功能": 3,
    "车辆通用功能": 2,
    "安全与儿童保护": 2,
    "车门锁与后备箱": 2,
    "触控屏与系统设置": 2,
    "警报与故障处理": 2,
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_no} must be a JSON object.")

            records.append(record)

    return records


def get_category(record: Dict[str, Any]) -> Optional[str]:
    metadata = record.get("additional_metadata") or {}

    for key in ("category", "类别", "topic", "主题"):
        value = record.get(key) or metadata.get(key)
        if value:
            return str(value).strip()

    return None


def print_category_stats(records: List[Dict[str, Any]]) -> None:
    stats = defaultdict(int)

    for record in records:
        stats[get_category(record) or "未分类"] += 1

    print("Golden category stats:")
    for category, count in stats.items():
        print(f"- {category}: {count}")


def filter_by_category(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)

    for record in records:
        category = get_category(record)
        if category:
            grouped[category].append(record)

    selected = []

    for category, quota in CATEGORY_QUOTAS.items():
        candidates = grouped.get(category, [])

        if len(candidates) < quota:
            raise ValueError(
                f"Category '{category}' needs {quota} records, "
                f"but only found {len(candidates)} records."
            )

        selected.extend(candidates[:quota])

    return selected


def build_output_record(golden: Dict[str, Any], rag_result: Dict[str, Any]) -> Dict[str, Any]:
    metadata = golden.get("additional_metadata") or {}
    answer = rag_result["answer"]

    return {
        "id": metadata.get("unique_id") or metadata.get("id") or golden.get("id"),
        "category": get_category(golden),
        "input": golden["input"],
        "expected_output": golden.get("expected_output"),
        "actual_output": answer["answer"],
        "retrieval_context": rag_result["retrieval_context"],
    }


def save_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG on Golden data.")
    parser.add_argument(
        "--version",
        required=True,
        help="Version label for this run (e.g. v1, v2). Output goes to data/<version>/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = DATA_DIR / args.version / "goldens_final_eval.jsonl"

    goldens = load_jsonl(GOLDEN_PATH)

    print(f"Loaded Golden records: {len(goldens)}")
    print_category_stats(goldens)

    selected_goldens = filter_by_category(goldens)

    print(f"Selected records: {len(selected_goldens)}")
    print_category_stats(selected_goldens)

    pipeline = HybridRAGPipeline(
        HybridRAGConfig(
            bm25_topk=5,
            milvus_topk=10,
            rerank_topk=5,
        )
    )

    results = []

    for golden in tqdm(selected_goldens, desc="Running RAG"):
        query = golden["input"].strip()
        rag_result = pipeline.answer(query)
        results.append(build_output_record(golden, rag_result))

    save_jsonl(output_path, results)

    print(f"Saved final eval data to: {output_path}")
    print(f"Final record count: {len(results)}")


if __name__ == "__main__":
    main()