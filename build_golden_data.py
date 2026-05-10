
"""Build DeepEval Golden data from an evaluation set.

默认输入为 data/qa_pairs/test_qa_pair_verify.json
输出为data/golden/goldens.jsonl。脚本会兼容当前项目中的评测集字段：

- question/query/input -> Golden.input
- answer/output/expected_output/reference -> Golden.expected_output
- pred.answer/actual_output/response -> Golden.actual_output
- context/retrieval_context -> Golden.retrieval_context
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from deepeval.dataset import EvaluationDataset
from deepeval.dataset.golden import Golden


DEFAULT_INPUT_PATH = "data/qa_pairs/test_qa_pair_verify.json"
DEFAULT_OUTPUT_DIR = "data/golden"
DEFAULT_OUTPUT_NAME = "goldens"


def load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    if path.suffix == ".jsonl":
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

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ("data", "records", "examples", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list, or a dict containing a list field.")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError("Every evaluation item must be a JSON object.")
    return data


def first_text(record: Dict[str, Any], field_names: Iterable[str]) -> Optional[str]:
    for field_name in field_names:
        value = record.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def extract_actual_output(record: Dict[str, Any]) -> Optional[str]:
    direct = first_text(record, ("actual_output", "response", "prediction", "pred_answer"))
    if direct:
        return direct

    pred = record.get("pred")
    if isinstance(pred, dict):
        answer = pred.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()
    elif isinstance(pred, str) and pred.strip():
        return pred.strip()

    return None


def normalize_context(value: Any) -> Optional[List[str]]:
    if value is None:
        return None

    if isinstance(value, list):
        contexts = [str(item).strip() for item in value if str(item).strip()]
        return contexts or None

    if not isinstance(value, str):
        value = str(value)

    value = value.strip()
    if not value:
        return None

    parts = re.split(r"(?m)(?=^\s*\d+[.．])", value)
    contexts = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        part = re.sub(r"^\s*\d+[.．]\s*", "", part).strip()
        if part:
            contexts.append(part)

    return contexts or [value]


def build_metadata(record: Dict[str, Any], source_path: Path, index: int) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "source_path": str(source_path),
        "source_index": index,
    }

    for key in ("unique_id", "id", "keywords", "category", "source", "page"):
        if key in record:
            metadata[key] = record[key]

    return metadata


def record_to_golden(record: Dict[str, Any], source_path: Path, index: int) -> Golden:
    user_input = first_text(record, ("input", "question", "query", "user_input"))
    expected_output = first_text(
        record,
        ("expected_output", "answer", "output", "reference", "ground_truth"),
    )

    if user_input is None:
        raise ValueError(f"Item {index} is missing input/question/query.")
    if expected_output is None:
        raise ValueError(f"Item {index} is missing answer/expected_output/reference.")

    context = normalize_context(record.get("gold_context") or record.get("reference_context"))
    retrieval_context = normalize_context(
        record.get("retrieval_context")
        or record.get("retrieved_contexts")
        or record.get("context")
    )

    return Golden(
        input=user_input,
        expected_output=expected_output,
        actual_output=extract_actual_output(record),
        context=context,
        retrieval_context=retrieval_context,
        additional_metadata=build_metadata(record, source_path, index),
        name=str(record.get("unique_id") or record.get("id") or f"golden-{index}"),
    )


def build_dataset(records: List[Dict[str, Any]], source_path: Path, limit: Optional[int]) -> EvaluationDataset:
    dataset = EvaluationDataset()
    selected_records = records[:limit] if limit is not None else records
    for index, record in enumerate(selected_records):
        dataset.add_golden(record_to_golden(record, source_path, index))
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DeepEval Golden data from an evaluation set.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Evaluation set path, JSON or JSONL.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to write Golden data.")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME, help="Output file name without extension.")
    parser.add_argument("--format", choices=("jsonl", "json", "csv"), default="jsonl", help="Output format.")
    parser.add_argument("--limit", type=int, default=None, help="Only convert the first N records.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    records = load_records(input_path)
    dataset = build_dataset(records, input_path, args.limit)
    output_path = dataset.save_as(args.format, directory=str(output_dir), file_name=args.output_name)

    print(f"Golden data size: {len(dataset.goldens)}")
    print(f"Golden data saved to: {output_path}")


if __name__ == "__main__":
    main()
