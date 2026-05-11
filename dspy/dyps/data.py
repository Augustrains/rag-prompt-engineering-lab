# -*- coding: utf-8 -*-
"""
Data loading utilities for DSPy DYPS.

Loads golden dataset, provides train/dev/test splits,
and transforms records into DSPy-compatible Example objects.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dspy


# ──────────────────────────── 类型别名 ────────────────────────────

GoldenRecord = Dict[str, Any]


# ──────────────────────────── 工具函数 ────────────────────────────

def load_jsonl(path: Path) -> List[GoldenRecord]:
    """Load records from a JSONL file."""
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


def save_jsonl(path: Path, records: List[GoldenRecord]) -> None:
    """Save records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_input(record: GoldenRecord) -> str:
    """Extract user query from a golden record."""
    return str(record.get("input") or record.get("question") or "").strip()


def get_expected_output(record: GoldenRecord) -> Optional[str]:
    """Extract expected answer from a golden record."""
    return record.get("expected_output")


def get_unique_id(record: GoldenRecord) -> str:
    """Extract unique ID from a golden record."""
    meta = record.get("additional_metadata") or {}
    uid = meta.get("unique_id") or record.get("id")
    return str(uid) if uid else ""


def get_category(record: GoldenRecord) -> Optional[str]:
    """Extract category from a golden record."""
    meta = record.get("additional_metadata") or {}
    cat = meta.get("category") or record.get("category")
    return str(cat).strip() if cat else None


# ──────────────────────────── 核心加载器 ────────────────────────────

class GoldenDataset:
    """
    Loads and manages the golden dataset for DYPS optimization.

    Usage:
        dataset = GoldenDataset(path="data/golden/goldens.jsonl")
        train, dev, test = dataset.split(train_ratio=0.7, dev_ratio=0.15, seed=42)

        # Get a single example as DSPy Example
        example = dataset[0]
    """

    def __init__(
        self,
        path: Path,
        fields: Tuple[str, str, str] = ("input", "expected_output", "retrieval_context"),
    ):
        self.path = Path(path)
        self.fields = fields  # (input_key, expected_key, context_key)
        self._records: List[GoldenRecord] = []
        self._examples: List[dspy.Example] = []
        self._load()

    def _load(self) -> None:
        """Load records and convert to DSPy Examples."""
        self._records = load_jsonl(self.path)
        self._examples = [
            self._record_to_example(rec) for rec in self._records
        ]

    def _record_to_example(self, record: GoldenRecord) -> dspy.Example:
        """Convert a golden record to a DSPy Example."""
        input_key, expected_key, context_key = self.fields
        kwargs = {
            "input": get_input(record),
            "expected_output": get_expected_output(record) or "",
            "unique_id": get_unique_id(record),
            "category": get_category(record) or "",
        }
        if context_key and context_key in record:
            kwargs["retrieval_context"] = record[context_key]
        return dspy.Example(**kwargs).with_inputs("input", "retrieval_context")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dspy.Example:
        return self._examples[index]

    def get_record(self, index: int) -> GoldenRecord:
        """Get the raw golden record at index."""
        return self._records[index]

    def get_records(self) -> List[GoldenRecord]:
        """Get all raw golden records."""
        return self._records

    def split(
        self,
        train_ratio: float = 0.7,
        dev_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
        """
        Split dataset into train / dev / test sets.

        All sets are stratified by category to ensure balanced distribution.
        """
        random.seed(seed)

        # Group by category
        by_category: Dict[str, List[Tuple[int, dspy.Example]]] = {}
        for idx, example in enumerate(self._examples):
            cat = example.category or "未分类"
            by_category.setdefault(cat, []).append((idx, example))

        train, dev, test = [], [], []

        for cat, items in by_category.items():
            random.shuffle(items)
            n = len(items)
            n_train = max(1, int(n * train_ratio))
            n_dev = max(1, int(n * dev_ratio))

            train.extend(ex for _, ex in items[:n_train])
            dev.extend(ex for _, ex in items[n_train:n_train + n_dev])
            test.extend(ex for _, ex in items[n_train + n_dev:])

        return train, dev, test

    def sample(
        self,
        n: int,
        seed: int = 42,
        categories: Optional[List[str]] = None,
    ) -> List[dspy.Example]:
        """
        Randomly sample n examples, optionally filtered by category.

        Args:
            n: Number of examples to sample
            seed: Random seed for reproducibility
            categories: If provided, only sample from these categories
        """
        random.seed(seed)
        pool = [
            ex for ex in self._examples
            if categories is None or ex.category in categories
        ]
        return random.sample(pool, min(n, len(pool)))

    def filter(
        self,
        categories: Optional[List[str]] = None,
        min_category_size: int = 1,
    ) -> List[dspy.Example]:
        """Filter examples by category."""
        filtered = self._examples
        if categories:
            filtered = [ex for ex in filtered if ex.category in categories]
        return filtered

    def stats(self) -> Dict[str, Any]:
        """Return dataset statistics."""
        from collections import Counter
        cats = [ex.category for ex in self._examples]
        return {
            "total": len(self._examples),
            "by_category": dict(Counter(cats)),
        }


# ──────────────────────────── 便捷函数 ────────────────────────────

def load_goldens(
    path: Path = Path("/root/autodl-tmp/rag-prompt-engineering-lab-upload/data/golden/goldens.jsonl"),
) -> GoldenDataset:
    """Load the default golden dataset."""
    return GoldenDataset(path)
