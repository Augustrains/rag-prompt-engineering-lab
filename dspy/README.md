# DSPy DYPS RAG Prompt Optimization

## 1. 项目概述

本项目实现了一套基于 **DSPy DYPS（Dynamic Prompt Selection）** 的 RAG 提示词自动化优化框架。通过让强大的教师模型（Teacher）指导本地学生模型（Student）进行迭代优化，最终找到最适合本地 RAG 场景的提示词参数。

**核心问题：** 本地 Qwen3-8B 模型配合 RAG pipeline 回答效果不佳，需要优化生成提示词（Prompt），但人工调优成本高、效率低。

**解决思路：** 引入一个强大的 API 模型（deepseek-v4-flash）作为教师，让学生模型与其对比评测，教师模型给出反馈并驱动提示词参数的迭代优化。

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      run_dyps.py（入口）                      │
│           编排整个 DYPS 流程：数据加载 → 模型初始化 → 优化 → 评测  │
└──────────┬────────────────┬──────────────────┬──────────────┘
           │                │                  │
           ▼                ▼                  ▼
┌──────────────────┐  ┌────────────┐  ┌─────────────────────┐
│  dyps/data.py    │  │ dyps/models│  │  dyps/evaluator.py │
│  GoldenDataset   │  │ StudentRAG  │  │   DeepEvalJudge    │
│  数据加载与切分   │  │ TeacherLM   │  │   Evaluator        │
│  (train/dev/test)│  │DSPyLMWrapper│  │  DeepEval 三维评测  │
└──────────────────┘  └──────┬─────┘  └─────────────────────┘
                             │
                             │ prompt_params 注入
                             ▼
              ┌──────────────────────────────┐
              │    dyps/teleprompter.py       │
              │       DYPSOptimizer           │
              │  BootstrapDemonstrator         │
              │  TeacherScorer + HintGenerator │
              │  核心优化循环：冷启动 → 评分 → 变异  │
              └──────────────┬───────────────┘
                             │ params: {temperature, max_tokens, use_cot}
                             ▼
              ┌──────────────────────────────────────┐
              │         RAG 项目 (外部依赖)            │
              │  HybridRAGPipeline.answer()          │
              │    → request_chat(query, context,    │
              │        temperature, max_tokens,      │
              │        system_prompt)                │
              │    → vLLM API (Qwen3-8B)             │
              └──────────────────────────────────────┘
```

### 2.1 参数注入链路（核心改进）

优化器生成的参数通过 4 层链路最终影响本地 LLM 的推理行为：

```
DYPSOptimizer.optimize_async()
  → prompt_params = {"temperature": 0.36, "max_tokens": 1469, "use_cot": True}
    │
    ▼ [teleprompter.py] student(input=..., prompt_params=params)
StudentRAG.forward(input, prompt_params)
  → use_cot → 构造 CoT system_prompt
    │
    ▼ [models.py] pipeline.answer(input, temperature, max_tokens, system_prompt)
HybridRAGPipeline.answer(query, temperature, max_tokens, system_prompt)
    │
    ▼ [RAG/utils.py] chat_fn(query, context, temperature, max_tokens, system_prompt)
request_chat(query, context, temperature, max_tokens, system_prompt)
    │
    ▼ [RAG/llm_local_client.py] vLLM API 调用（参数真实生效）
Qwen3-8B 本地模型推理
```

| 参数 | 作用 | 映射方式 |
|------|------|----------|
| `temperature` | LLM 生成温度 | 直接传递给 vLLM API |
| `max_tokens` | 最大输出长度 | 直接传递给 vLLM API |
| `use_cot` | 是否使用 Chain-of-Thought | 切换 system_prompt 为逐步推理指令 |
| `demo_selection` | 示例选择策略（预留） | 尚未映射 |

---

## 3. 教师-学生架构（Teacher-Student Model）

| 角色 | 模型 | 用途 |
|------|------|------|
| **Student（学生）** | 本地 Qwen3-8B + HybridRAGPipeline（BM25 + Milvus + BGE Reranker） | 被优化的对象，输出答案 |
| **Teacher（教师）** | deepseek-v4-flash（API 调用） | 生成参考答案、评分、生成改进 hints |
| **DSPy LM** | deepseek-v4-flash（API，包装为 DSPy LM） | 供 DSPy 内部操作使用（bootstrap demos 等） |
| **Judge** | deepseek-v4-flash（API，包装为 DeepEval 兼容接口） | 运行 DeepEval 三维评测指标 |

---

## 4. 模块说明

### 4.1 `dyps/data.py` — 数据加载与切分

负责从 `goldens.jsonl` 加载黄金数据集，并切分为训练/开发/测试集。

- **`GoldenDataset`**：核心类，加载 JSONL 并转换为 DSPy `Example` 对象
- **`split()`**：按类别分层采样，切分 train / dev / test（默认 70% / 15% / 15%）
- **`sample()` / `filter()`**：支持按类别筛选和随机采样
- **`stats()`**：返回数据集统计（总数、按类别分布）

```python
dataset = GoldenDataset(path="data/golden/goldens.jsonl")
train, dev, test = dataset.split(train_ratio=0.7, dev_ratio=0.15, seed=42)
```

### 4.2 `dyps/models.py` — 三类模型封装

**`StudentRAG`**（`dspy.Module`）
- 包装本地 `HybridRAGPipeline`（BM25 + Milvus + BGE Reranker + Qwen3-8B vLLM）
- `forward(input, prompt_params=None)` → 返回 `dspy.Prediction(answer, retrieval_context)`
- **`prompt_params`** 接受 `{"temperature", "max_tokens", "use_cot"}` 字典
- `use_cot=True` 时自动切换为 Chain-of-Thought 系统提示词
- 这是 DYPS 要优化的模型

**`TeacherLM`**（`dspy.Module`）
- 调用 API 模型，基于检索上下文生成参考答案
- `forward(input, context)` → 参考答案
- `grade(input, expected, actual)` → 三维评分

**`DSPyLMWrapper`**
- 将任意 OpenAI 兼容 API 包装为 DSPy `LM` 对象
- `configure()` → 设置 `dspy.configure(lm=...)`

### 4.3 `dyps/signatures.py` — DSPy 签名定义

DSPy 签名定义了模块的输入输出接口：

| 签名 | 用途 |
|------|------|
| `RAGSignature` | 主 RAG 问答签名：`input` + `retrieval_context` → `answer` |
| `TeacherGradingSignature` | 教师评分签名：输出 `answer_relevancy`、`faithfulness`、`contextual_recall`、`overall_score` |
| `HintGenerationSignature` | 教师生成改进 hints：`hint` + `improved_answer` + `reasoning` |
| `CompositeRAGSignature` | 扩展签名，含 `expected_output` 和 `score` 字段 |

### 4.4 `dyps/evaluator.py` — 评测模块

**`DeepEvalJudge`**（`DeepEvalBaseLLM`）
- 将 API 模型包装为 DeepEval 可用的 Judge 接口
- 实现 `generate()`（同步）和 `a_generate()`（异步），返回 JSON

**`Evaluator`**
- 异步运行 DeepEval 三个指标：`AnswerRelevancyMetric`、`FaithfulnessMetric`、`ContextualRecallMetric`
- 综合分权重：`answer_relevancy × 0.3 + faithfulness × 0.4 + contextual_recall × 0.3`
- 支持并发控制（`max_concurrency`）和重试机制（`MAX_RETRIES=3`）

**`EvalResult`**：单条评测结果，包含各维度分数、总体分、`success`、`reason`

### 4.5 `dyps/teleprompter.py` — DYPS 优化核心

**`TeacherScorer`**
- 调用 `Evaluator` 对学生答案评分
- 无 Judge 时回退到基于关键词重叠的规则评分

**`BootstrapDemonstrator`**
- 对每个样本同时调用 Student 和 Teacher，收集 DemoRecord
- 标注 `is_positive`（overall ≥ 0.7）或 `is_negative`（overall < 0.4）
- 用于构建高质量的 bootstrap 示例池

**`HintGenerator`**
- 调用 Teacher API，分析学生答案问题，生成 JSON 格式的改进建议
- 输出 `issue`、`hint`、`improved_answer`、`reasoning`

**`DYPSOptimizer`**（核心类）
1. **冷启动（Cold Start）**：用 Teacher 生成初始 bootstrap demos
2. **优化循环（`optimize_async`）**：
   - 生成新参数（基于历史最佳参数做探索-利用变异）
   - 将参数通过 `prompt_params` 注入 StudentRAG，实际影响 vLLM 推理
   - 在 dev 集上评测当前参数
   - 若教师胜率超过阈值则早停
3. 返回 `OptimizationResult`（最佳参数、最佳分数、全程历史）

### 4.6 `dyps/config.py` — 配置中心

所有配置集中管理：
- 数据路径：`GOLDEN_PATH`、`OUTPUT_DIR`
- 教师/学生/DSPy LM 配置
- 评测阈值：`answer_relevancy ≥ 0.5`、`faithfulness ≥ 0.5`、`contextual_recall ≥ 0.6`
- DYPS 超参数：`num_trials=50`、`max_demos=8`、`cold_start_samples=10`、`teacher_win_threshold=0.7`

### 4.7 `run_dyps.py` — 主入口脚本

编排完整流程：

```
setup_environment()
  → 加载 config.ini 环境变量

load_dataset()
  → GoldenDataset + split → train / dev / test

setup_models()
  → StudentRAG + TeacherLM + DeepEvalJudge + DSPyLMWrapper

DYPSOptimizer.cold_start()
  → BootstrapDemonstrator 生成初始 demos

DYPSOptimizer.optimize_async()
  → 迭代优化 trials

evaluate_on_test()
  → 在 test 集上评测最佳参数

save_summary()
  → 输出 run_summary.json
```

支持丰富的 CLI 参数：`--num-trials`、`--cold-start`、`--train-ratio`、`--dev-ratio`、`--dry-run`、`--skip-test-eval` 等。

---

## 5. 依赖关系图

```
run_dyps.py
├── dyps/data.py → GoldenDataset
├── dyps/models.py
│   ├── StudentRAG ──→ HybridRAGPipeline (from /root/autodl-tmp/RAG)
│   │                   └── request_chat() → vLLM API (Qwen3-8B)
│   │                       └── 参数注入: temperature, max_tokens, system_prompt
│   ├── TeacherLM ────→ DeepSeek API
│   └── DSPyLMWrapper → dspy.LM + DeepSeek API
├── dyps/evaluator.py
│   ├── DeepEvalJudge → DeepSeek API + DeepEval metrics
│   └── Evaluator ────→ DeepEval metrics
└── dyps/teleprompter.py
    ├── TeacherScorer ──→ DeepEvalJudge
    ├── BootstrapDemonstrator → StudentRAG + TeacherLM
    ├── HintGenerator ──→ TeacherLM
    └── DYPSOptimizer
        └── prompt_params 注入 → StudentRAG → ... → vLLM
```

---

## 6. 运行方式

```bash
# 基础运行（使用 rag conda 环境）
/root/autodl-tmp/conda/envs/rag/bin/python run_dyps.py

# 自定义参数
/root/autodl-tmp/conda/envs/rag/bin/python run_dyps.py \
    --num-trials 50 \
    --cold-start 10 \
    --train-ratio 0.7 \
    --dev-ratio 0.15 \
    --output-dir ./data/dyps

# 仅验证配置
/root/autodl-tmp/conda/envs/rag/bin/python run_dyps.py --dry-run

# 跳过测试集评测（节省 token）
/root/autodl-tmp/conda/envs/rag/bin/python run_dyps.py --skip-test-eval
```

---

## 7. 输出文件

运行后在 `OUTPUT_DIR`（默认 `./data/dyps`）生成：

| 文件 | 内容 |
|------|------|
| `test_eval_results.jsonl` | 测试集每条记录的三维评测结果 |
| `run_summary.json` | 完整运行摘要（最佳参数、最佳分、trial 历史、测试汇总） |

---

## 8. 技术栈

- **DSPy 3.2.1** — 提示词编程框架
- **DeepEval 4.0.0** — LLM 评测框架（AnswerRelevancy / Faithfulness / ContextualRecall）
- **本地 RAG Pipeline** — HybridRAGPipeline（BM25 + Milvus + BGE Reranker + Qwen3-8B vLLM）
- **教师模型** — deepseek-v4-flash（通过 DeepSeek API 调用）
- **Python 3.12** + conda 环境（`rag`）

---

## 9. 修改记录

### 2026-05-11 — 参数注入链路打通 + Bug 修复

**核心改进：实现了 prompt_params → vLLM 的完整参数注入链路**

修改了 4 个文件，打通了从优化器到本地 LLM 的 4 层参数传递：

| 文件 | 改动 |
|------|------|
| `/root/autodl-tmp/RAG/src/client/llm_local_client.py` | `request_chat()` 新增 `temperature`、`max_tokens`、`system_prompt` 参数，替换硬编码 |
| `/root/autodl-tmp/RAG/src/utils.py` | `HybridRAGPipeline.answer()` 新增参数透传 `→ chat_fn()` |
| `dyps/models.py` | `StudentRAG.forward()` 接收 `prompt_params`，`use_cot` 映射到 CoT 系统提示词 |
| `dyps/teleprompter.py` | `optimize_async()` 调用 student 时传入 `prompt_params=params` |

**Bug 修复（3 个）：**

| Bug | 修复 |
|-----|------|
| `StudentRAG.forward()` 不接受 `retrieval_context` 参数导致冷启动崩溃 | 添加 `**kwargs` |
| `run_optimization()` 返回类型为 `None`，主流程拿到 None 后 `best_score` 报 AttributeError | 改为 `return result` |
| `save_summary()` 中 `PosixPath` 无法 JSON 序列化 | `vars(args)` 转换时 Path → str |

**验证结果：**
- 参数注入验证：同一问题 3 组不同参数 → 3 个不同答案 ✅
- 端到端全流程：`run_dyps.py --num-trials 2` exit 0 ✅
- 评测分数变化：`best_score=0.6775`，`best_params={'temperature': 0.36, 'max_tokens': 1469, 'use_cot': True}`
