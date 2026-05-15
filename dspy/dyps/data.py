# -*- coding: utf-8 -*-
"""
DSPy DYPS 数据加载模块
======================

负责从 goldens.jsonl 加载黄金测试数据集，并将其转换为 DSPy 兼容的 Example 对象。

核心类 GoldenDataset 提供以下能力：
  1. 加载 JSONL → 内部存储 raw records + DSPy Example 对象
  2. 按类别分层切分 train / dev / test（保证各类别分布均衡）
  3. 按类别过滤和随机采样
  4. 数据集统计（总数、类别分布）

Golden 数据格式 (goldens.jsonl 每行)：
  {
    "input": "用户问题",
    "expected_output": "标准答案",
    "retrieval_context": [...],     // 可能为 null（DSPy 优化时使用学生实际检索结果）
    "additional_metadata": {
      "unique_id": "...",
      "category": "充电相关",
      ...
    }
  }
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dspy


# ═══════════════════════════════════════════════════════════════════════════════
# 类型别名
# ═══════════════════════════════════════════════════════════════════════════════

GoldenRecord = Dict[str, Any]  # 原始 JSONL 中一条记录的反序列化结果


# ═══════════════════════════════════════════════════════════════════════════════
# JSONL 读写工具
# ═══════════════════════════════════════════════════════════════════════════════

def load_jsonl(path: Path) -> List[GoldenRecord]:
    """从 JSONL 文件逐行加载记录（跳过空行）"""
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
    """将记录列表保存为 JSONL 文件（自动创建父目录）"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 字段提取辅助函数（兼容多种字段命名）
# ═══════════════════════════════════════════════════════════════════════════════

def get_input(record: GoldenRecord) -> str:
    """提取用户问题（兼容 input / question 两种字段名）"""
    return str(record.get("input") or record.get("question") or "").strip()


def get_expected_output(record: GoldenRecord) -> Optional[str]:
    """提取标准答案"""
    return record.get("expected_output")


def get_unique_id(record: GoldenRecord) -> str:
    """提取唯一 ID（优先从 additional_metadata.unique_id，其次 id）"""
    meta = record.get("additional_metadata") or {}
    uid = meta.get("unique_id") or record.get("id")
    return str(uid) if uid else ""


def get_category(record: GoldenRecord) -> Optional[str]:
    """提取类别标签（优先从 additional_metadata.category，其次 category）"""
    meta = record.get("additional_metadata") or {}
    cat = meta.get("category") or record.get("category")
    return str(cat).strip() if cat else None


# ═══════════════════════════════════════════════════════════════════════════════
# 核心类：GoldenDataset
# ═══════════════════════════════════════════════════════════════════════════════

class GoldenDataset:
    """
    黄金数据集管理器。

    职责：
      1. 从 JSONL 加载原始记录
      2. 将每条记录转换为 DSPy Example（供 DSPy 框架使用）
      3. 提供分层采样切分、过滤、统计等功能

    使用示例：
        dataset = GoldenDataset(path="data/golden/goldens.jsonl")
        train, dev, test = dataset.split(train_ratio=0.7, dev_ratio=0.15, seed=42)
        example = dataset[0]  # 返回 dspy.Example 对象
    """

    def __init__(
        self,
        path: Path,
        fields: Tuple[str, str, str] = ("input", "expected_output", "retrieval_context"),
    ):
        """
        初始化数据集。

        Args:
            path: golden JSONL 文件路径
            fields: (input字段名, expected_output字段名, retrieval_context字段名) 三元组
                    用于指定从原始记录中提取哪些字段构建 DSPy Example
        """
        self.path = Path(path)
        self.fields = fields
        self._records: List[GoldenRecord] = []       # 原始 JSON 记录
        self._examples: List[dspy.Example] = []      # DSPy Example 对象（供 DSPy 模块使用）
        self._load()

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        """加载 JSONL 并转换为 DSPy Example 列表"""
        self._records = load_jsonl(self.path)
        self._examples = [
            self._record_to_example(rec) for rec in self._records
        ]

    def _record_to_example(self, record: GoldenRecord) -> dspy.Example:
        """
        将一条 Golden 记录转换为 DSPy Example。

        DSPy Example 包含以下字段：
          - input: 用户问题
          - expected_output: 标准答案
          - unique_id: 唯一标识
          - category: 类别标签
          - retrieval_context: 检索上下文（可选，golden 数据中常为 null）

        with_inputs("input", "retrieval_context") 告诉 DSPy：
          这两个字段是模块的输入，其他字段（如 expected_output）是标签/元数据。
        """
        input_key, expected_key, context_key = self.fields
        kwargs = {
            "input": get_input(record),
            "expected_output": get_expected_output(record) or "",
            "unique_id": get_unique_id(record),
            "category": get_category(record) or "",
        }
        # 如果原始记录中包含检索上下文，则一并传入
        if context_key and context_key in record:
            kwargs["retrieval_context"] = record[context_key]
        return dspy.Example(**kwargs).with_inputs("input", "retrieval_context")

    # ── 基础访问方法 ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        """返回数据集总条数"""
        return len(self._records)

    def __getitem__(self, index: int) -> dspy.Example:
        """按索引获取 DSPy Example"""
        return self._examples[index]

    def get_record(self, index: int) -> GoldenRecord:
        """按索引获取原始 Golden 记录（非 DSPy Example）"""
        return self._records[index]

    def get_records(self) -> List[GoldenRecord]:
        """获取所有原始 Golden 记录"""
        return self._records

    # ── 数据集切分 ────────────────────────────────────────────────────────

    def split(
        self,
        train_ratio: float = 0.7,
        dev_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
        """
        按类别分层切分为 train / dev / test 三组。

        分层策略：
          1. 按 category 字段将数据分组
          2. 对每组独立进行 shuffle + 比例切分
          3. 每组至少有 1 条（避免某些类别在某个集合中完全缺失）

        这样做的好处：确保每个类别在 train/dev/test 中的分布与整体一致，
        避免评测结果因类别不均衡而失真。

        Args:
            train_ratio: 训练集比例（默认 0.7）
            dev_ratio: 开发集比例（默认 0.15）
            seed: 随机种子

        Returns:
            (train, dev, test) 三元组，每个元素是 dspy.Example 列表
        """
        random.seed(seed)

        # 第一步：按类别分组，每组保留原始索引和 Example
        by_category: Dict[str, List[Tuple[int, dspy.Example]]] = {}
        for idx, example in enumerate(self._examples):
            cat = example.category or "未分类"
            by_category.setdefault(cat, []).append((idx, example))

        train, dev, test = [], [], []

        # 第二步：对每个类别独立 shuffle 和切分
        for cat, items in by_category.items():
            random.shuffle(items)
            n = len(items)
            n_train = max(1, int(n * train_ratio))
            n_dev = max(1, int(n * dev_ratio))

            train.extend(ex for _, ex in items[:n_train])
            dev.extend(ex for _, ex in items[n_train:n_train + n_dev])
            test.extend(ex for _, ex in items[n_train + n_dev:])

        return train, dev, test

    # ── 采样与过滤 ────────────────────────────────────────────────────────

    def sample(
        self,
        n: int,
        seed: int = 42,
        categories: Optional[List[str]] = None,
    ) -> List[dspy.Example]:
        """
        随机采样 n 条样本。

        Args:
            n: 目标采样数
            seed: 随机种子
            categories: 可选，限定只从这些类别中采样

        Returns:
            采样到的 dspy.Example 列表（实际数量 = min(n, 符合条件的总数)）
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
        """
        按类别过滤数据。

        Args:
            categories: 要保留的类别列表（None 表示不过滤）
            min_category_size: 忽略，预留参数

        Returns:
            过滤后的 dspy.Example 列表
        """
        filtered = self._examples
        if categories:
            filtered = [ex for ex in filtered if ex.category in categories]
        return filtered

    # ── 统计 ──────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """返回数据集统计信息（总数 + 各类别数量分布）"""
        from collections import Counter
        cats = [ex.category for ex in self._examples]
        return {
            "total": len(self._examples),
            "by_category": dict(Counter(cats)),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════════════════════

def load_goldens(
    path: Path = Path("/root/autodl-tmp/rag-prompt-engineering-lab-upload/data/golden/goldens.jsonl"),
) -> GoldenDataset:
    """快速加载默认位置的 Golden 数据集"""
    return GoldenDataset(path)
