"""基于 DeepEval LLM 指标的异步评测模块。"""

import argparse
import asyncio
import json
import re
from datetime import date
from pathlib import Path

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.llms.openai_model import GPTModel
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.test_case import LLMTestCase
from openai import OpenAI, AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


CONFIG_PATH = Path("/root/autodl-tmp/RAG/config.ini")
DATA_DIR = Path("/root/autodl-tmp/rag-prompt-engineering-lab-upload/data")
CHANGELOG_PATH = Path("/root/autodl-tmp/rag-prompt-engineering-lab-upload/data/prompts/changelog.md")

# 建议先设为 1，降低评测模型返回非法 JSON 的概率
MAX_CONCURRENCY = 1

# 每个 metric 最多尝试 3 次，3 次失败后终止程序
MAX_RETRIES = 3


class DeepSeekJudge(DeepEvalBaseLLM):
    def __init__(self, model, api_key, base_url, temperature=0):
        # 注意：这些属性必须放在 super().__init__ 前面
        # 因为 DeepEvalBaseLLM.__init__ 会调用 self.load_model()
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.name = model
        super().__init__(model)

    def generate(self, prompt, schema=None):
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        completion = client.chat.completions.create(
            model=self.name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict evaluation assistant. You must output valid JSON only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )

        raw = completion.choices[0].message.content or ""

        if schema:
            json_output = trim_and_load_json(raw)
            return schema.model_validate(json_output), 0.0

        return raw, 0.0

    async def a_generate(self, prompt, schema=None):
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        completion = await client.chat.completions.create(
            model=self.name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict evaluation assistant. You must output valid JSON only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )

        raw = completion.choices[0].message.content or ""

        if schema:
            json_output = trim_and_load_json(raw)
            return schema.model_validate(json_output), 0.0

        return raw, 0.0

    def load_model(self, async_mode=False):
        if not async_mode:
            return OpenAI(api_key=self.api_key, base_url=self.base_url)

        return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_model_name(self):
        return self.name


def load_jsonl(path: Path):
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


def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_env(config_path: Path) -> dict:
    env = {}
    pattern = re.compile(r'^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*?)(?:\s+#.*)?$')

    with config_path.open(encoding="utf-8") as f:
        for line in f:
            match = pattern.match(line)

            if not match:
                continue

            key, val = match.groups()
            env[key] = val.strip().strip('"').strip("'")

    return env


def build_judge_model(config_path: Path) -> DeepSeekJudge:
    env = load_env(config_path)

    required_keys = [
        "DOUBAO_MODEL_NAME",
        "DOUBAO_API_KEY",
        "DOUBAO_BASE_URL",
    ]

    missing_keys = [key for key in required_keys if not env.get(key)]

    if missing_keys:
        raise ValueError(f"config.ini 缺少必要配置: {missing_keys}")

    return DeepSeekJudge(
        model=env["DOUBAO_MODEL_NAME"],
        api_key=env["DOUBAO_API_KEY"],
        base_url=env["DOUBAO_BASE_URL"],
        temperature=0,
    )


def build_metrics(model: GPTModel):
    return [
        AnswerRelevancyMetric(
            threshold=0.5,
            include_reason=True,
            async_mode=True,
            model=model,
        ),
        FaithfulnessMetric(
            threshold=0.5,
            include_reason=True,
            async_mode=True,
            model=model,
        ),
        ContextualRecallMetric(
            threshold=0.6,
            include_reason=True,
            async_mode=True,
            model=model,
        ),
    ]


def to_test_case(record: dict) -> LLMTestCase:
    return LLMTestCase(
        input=record["input"],
        actual_output=record["actual_output"],
        expected_output=record.get("expected_output"),
        retrieval_context=record["retrieval_context"],
    )


async def evaluate_one_record(
    record: dict,
    model: GPTModel,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        test_case = to_test_case(record)
        metrics = build_metrics(model)

        async def run_metric(metric):
            metric_name = metric.__class__.__name__
            last_error = None

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    score = await metric.a_measure(test_case)

                    return metric_name, {
                        "score": score,
                        "success": metric.is_successful(),
                        "reason": metric.reason,
                        "attempts": attempt,
                        "error": None,
                    }

                except Exception as e:
                    last_error = e

                    print(
                        f"[WARN] id={record.get('id')} "
                        f"metric={metric_name} "
                        f"attempt={attempt}/{MAX_RETRIES} failed: {e}"
                    )

                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(2 * attempt)

            raise RuntimeError(
                f"Metric failed after {MAX_RETRIES} attempts. "
                f"id={record.get('id')}, "
                f"metric={metric_name}, "
                f"error={last_error}"
            )

        metric_items = await asyncio.gather(
            *(run_metric(metric) for metric in metrics)
        )

        metric_results = dict(metric_items)
        scores = [item["score"] for item in metric_results.values()]

        return {
            "id": record.get("id"),
            "category": record.get("category"),
            "input": record["input"],
            "metrics": metric_results,
            "overall_score": sum(scores) / len(scores),
            "success": all(item["success"] for item in metric_results.values()),
        }


async def evaluate_records(
    records: list[dict],
    model: GPTModel,
    max_concurrency: int = 1,
) -> list[dict]:
    semaphore = asyncio.Semaphore(max_concurrency)

    tasks = [
        evaluate_one_record(record, model, semaphore)
        for record in records
    ]

    return await tqdm_asyncio.gather(*tasks, desc="DeepEval")


def print_summary(results: list[dict]) -> None:
    avg_score = sum(r["overall_score"] for r in results) / len(results)
    success_count = sum(1 for r in results if r["success"])

    print()
    print(f"评测完成，共 {len(results)} 条")
    print(f"平均分: {avg_score:.3f}")
    print(f"成功数: {success_count}/{len(results)}")

    metric_names = list(results[0]["metrics"].keys()) if results else []

    for metric_name in metric_names:
        scores = [
            result["metrics"][metric_name]["score"]
            for result in results
        ]

        avg_metric_score = sum(scores) / len(scores)
        print(f"{metric_name} 平均分: {avg_metric_score:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG results with DeepEval.")

    parser.add_argument(
        "--version",
        required=True,
        help="Version label to evaluate, e.g. v1, v2. Reads from data/<version>/",
    )

    parser.add_argument(
        "--prompt-file",
        default="",
        help="Prompt file name used for this version (e.g. rag_qa_prompt.txt). Optional.",
    )

    parser.add_argument(
        "--description",
        default="",
        help="Short description of this version's prompt changes. Optional.",
    )

    return parser.parse_args()


def compute_summary(results: list[dict]) -> dict:
    metric_names = list(results[0]["metrics"].keys()) if results else []
    summary = {"overall": sum(r["overall_score"] for r in results) / len(results)}

    for name in metric_names:
        scores = [r["metrics"][name]["score"] for r in results]
        summary[name] = sum(scores) / len(scores)

    return summary


def update_changelog(version: str, summary: dict, prompt_file: str, description: str) -> None:
    if not CHANGELOG_PATH.exists():
        return

    content = CHANGELOG_PATH.read_text(encoding="utf-8")
    today = date.today().isoformat()
    prompt_cell = prompt_file or f"{version}.txt"
    data_cell = f"data/{version}/"

    row = (
        f"| {version} | {prompt_cell} | {data_cell} | "
        f"{summary['overall']:.4f} | "
        f"{summary.get('AnswerRelevancyMetric', '-'):.4f} | "
        f"{summary.get('FaithfulnessMetric', '-'):.4f} | "
        f"{summary.get('ContextualRecallMetric', '-'):.4f} | "
        f"{description or '-'} |"
    )

    lines = content.split("\n")
    new_lines = []
    inserted = False

    for line in lines:
        new_lines.append(line)
        if line.startswith("|------") and not inserted:
            new_lines.append(row)
            inserted = True

    if not inserted:
        new_lines.append(row)

    CHANGELOG_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"已更新 changelog: {CHANGELOG_PATH}")


def main() -> None:
    args = parse_args()

    input_path = DATA_DIR / args.version / "goldens_final_eval.jsonl"
    output_path = DATA_DIR / args.version / "goldens_final_eval_deepeval_scores.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    records = load_jsonl(input_path)

    if not records:
        raise ValueError("没有读取到评测数据。")

    print(f"读取评测数据: {input_path}")
    print(f"评测样本数量: {len(records)}")
    print(f"最大并发数: {MAX_CONCURRENCY}")
    print(f"失败最大重试次数: {MAX_RETRIES}")

    model = build_judge_model(CONFIG_PATH)

    results = asyncio.run(
        evaluate_records(
            records=records,
            model=model,
            max_concurrency=MAX_CONCURRENCY,
        )
    )

    save_jsonl(output_path, results)

    print(f"结果已保存到: {output_path}")

    print_summary(results)

    summary = compute_summary(results)
    update_changelog(args.version, summary, args.prompt_file, args.description)


if __name__ == "__main__":
    main()