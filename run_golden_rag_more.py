"""Run RAG on an expanded stratified golden sample (50 cases).

Sampling strategy:
  1. Stratified by category (top 13 categories + grouped 'other')
  2. 优先包含已知顽疾 case（基于之前评测失败记录）
  3. 无答案 case 保证至少 2 条
  4. 关键词匹配标记高价值边缘 case
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from src.utils import HybridRAGConfig, HybridRAGPipeline


DATA_DIR = Path("/root/autodl-tmp/rag-prompt-engineering-lab-upload/data")
GOLDEN_PATH = DATA_DIR / "golden" / "goldens.jsonl"

# 抽样配额（总计约50条，±2条可接受）
CATEGORY_QUOTAS = {
    "充电相关": 7,
    "驾驶操控与底盘": 5,
    "辅助驾驶与智能功能": 5,
    "车辆通用功能": 4,
    "安全与儿童保护": 4,
    "车门锁与后备箱": 4,
    "触控屏与系统设置": 4,
    "警报与故障处理": 4,
    "媒体与连接": 3,
    "空调与温度控制": 3,
    "维护与保养": 2,
    "车辆参数与规格": 2,
    "无法回答/闲聊": 2,
}

OTHER_CATEGORIES = [
    "导航与地图", "灯光与照明", "HomeLink与外部设备",
    "哨兵与驻车模式", "后视镜", "USB与数据记录",
    "软件与更新", "行人警示系统",
]
OTHER_QUOTA = 3

# 顽疾关键词：之前版本中反复失败的 case 相关关键词
CHRONIC_KEYWORDS = [
    "行程", "重命名",
    "钥匙", "新钥匙",
    "防滑链", "雪地链",
    "碰撞", "多碰撞", "制动",
    "赛道", "赛道模式",
    "儿童座椅", "座椅",
    "里程", "里程表",
    "手套箱", "手套",
    "更新", "OTA",
    "摄像头", "校准",
    "备份", "恢复",
]

NO_ANSWER_KEYWORDS = ["无法回答", "闲聊", "无相关信息"]


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
    meta = record.get("additional_metadata") or {}
    for key in ("category", "类别"):
        val = record.get(key) or meta.get(key)
        if val:
            return str(val).strip()
    return None


def get_unique_id(record: Dict[str, Any]) -> str:
    meta = record.get("additional_metadata") or {}
    return str(
        meta.get("unique_id")
        or meta.get("id")
        or record.get("id")
        or ""
    )


def get_input(record: Dict[str, Any]) -> str:
    return str(record.get("input") or record.get("question") or "").strip()


def is_chronic_case(record: Dict[str, Any]) -> bool:
    text = get_input(record).lower()
    for kw in CHRONIC_KEYWORDS:
        if kw.lower() in text:
            return True
    return False


def is_no_answer_case(record: Dict[str, Any]) -> bool:
    text = get_input(record)
    for kw in NO_ANSWER_KEYWORDS:
        if kw in text:
            return True
    return False


def select_stratified(
    records: List[Dict[str, Any]],
    quotas: Dict[str, int],
    other_quota: int = 3,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    random.seed(seed)

    grouped = defaultdict(list)
    other_records = []
    chronic_by_cat = defaultdict(list)

    for record in records:
        cat = get_category(record)
        if cat in quotas:
            grouped[cat].append(record)
        elif cat in OTHER_CATEGORIES:
            other_records.append(record)
        else:
            grouped.setdefault("未分类", []).append(record)
        if is_chronic_case(record):
            chronic_by_cat[cat].append(record)

    selected: List[Dict[str, Any]] = []
    selected_ids: Set[str] = set()

    # 规则1：顽疾 case 优先，每类别最多预占 2 个
    for cat, quota in quotas.items():
        chronic_pool = [
            r for r in chronic_by_cat.get(cat, [])
            if get_unique_id(r) not in selected_ids
        ]
        chronic_taken = min(2, quota, len(chronic_pool))
        for r in chronic_pool[:chronic_taken]:
            selected.append(r)
            selected_ids.add(get_unique_id(r))

    # 规则2：按配额填充，顽疾 case 不足时用随机样本补足
    for cat, quota in quotas.items():
        current_in_cat = [r for r in selected if get_category(r) == cat]
        needed = quota - len(current_in_cat)
        if needed <= 0:
            continue
        pool = [
            r for r in grouped.get(cat, [])
            if get_unique_id(r) not in selected_ids
        ]
        if len(pool) < needed:
            raise ValueError(
                f"Category '{cat}' needs {needed} records, "
                f"but only {len(pool)} available"
            )
        chosen = random.sample(pool, needed)
        for r in chosen:
            selected.append(r)
            selected_ids.add(get_unique_id(r))

    # 规则3：Other categories
    other_pool = [
        r for r in other_records
        if get_unique_id(r) not in selected_ids
    ]
    other_chosen = random.sample(
        other_pool, min(other_quota, len(other_pool))
    )
    for r in other_chosen:
        selected.append(r)
        selected_ids.add(get_unique_id(r))

    # 规则4：无答案 case 保证至少 2 条
    no_answer_selected = [r for r in selected if is_no_answer_case(r)]
    if len(no_answer_selected) < 2:
        no_answer_pool = [
            r for r in records
            if is_no_answer_case(r) and get_unique_id(r) not in selected_ids
        ]
        needed = 2 - len(no_answer_selected)
        extra = random.sample(
            no_answer_pool, min(needed, len(no_answer_pool))
        )
        for r in extra:
            selected.append(r)
            selected_ids.add(get_unique_id(r))

    return selected, selected_ids


def print_selection_stats(selected: List[Dict[str, Any]]) -> None:
    by_cat = defaultdict(int)
    chronic_count = sum(1 for r in selected if is_chronic_case(r))
    no_answer_count = sum(1 for r in selected if is_no_answer_case(r))

    for r in selected:
        by_cat[get_category(r) or "未分类"] += 1

    print(f"\n=== 抽样统计 ===")
    print(f"总条数: {len(selected)}")
    print(f"顽疾 case 数: {chronic_count}")
    print(f"无答案 case 数: {no_answer_count}")
    print("\n各类别分布:")
    for cat, cnt in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt}")


def build_output_record(
    golden: Dict[str, Any], rag_result: Dict[str, Any]
) -> Dict[str, Any]:
    answer = rag_result["answer"]
    return {
        "id": get_unique_id(golden),
        "category": get_category(golden),
        "input": get_input(golden),
        "expected_output": golden.get("expected_output"),
        "actual_output": answer["answer"],
        "retrieval_context": rag_result["retrieval_context"],
        "is_chronic": is_chronic_case(golden),
        "is_no_answer": is_no_answer_case(golden),
    }


def save_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAG on expanded stratified golden sample (50 cases)."
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Version label (v5=v2-prompt, v6=v4-prompt). Output goes to data/<version>/",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print sampling stats without running RAG",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    goldens = load_jsonl(GOLDEN_PATH)
    print(f"Loaded {len(goldens)} golden records")

    selected, selected_ids = select_stratified(
        goldens, CATEGORY_QUOTAS, OTHER_QUOTA, seed=args.seed
    )
    print_selection_stats(selected)

    if args.dry_run:
        print("\n[dry-run] 跳过 RAG 执行")
        return

    output_path = DATA_DIR / args.version / "goldens_final_eval.jsonl"
    print(f"\nRunning RAG with v{args.version} ...")

    pipeline = HybridRAGPipeline(
        HybridRAGConfig(bm25_topk=5, milvus_topk=10, rerank_topk=5)
    )

    results = []
    for golden in tqdm(selected, desc=f"RAG v{args.version}"):
        query = get_input(golden)
        rag_result = pipeline.answer(query)
        results.append(build_output_record(golden, rag_result))

    save_jsonl(output_path, results)
    print(f"\nSaved to: {output_path} ({len(results)} records)")

    by_cat = defaultdict(int)
    for r in results:
        by_cat[r["category"]] += 1
    print("\n结果类别分布:")
    for cat, cnt in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt} 条")


if __name__ == "__main__":
    main()
