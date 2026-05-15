# Tesla RAG 提示词工程实验记录

## 项目概述

本项目针对特斯拉 Model 3 用户手册知识库构建 RAG 问答系统，使用 DeepEval 对不同提示词版本进行系统性评测与优化。

---

## 目录结构

```
rag-prompt-engineering-lab-upload/
├── data/
│   ├── prompts/              # 提示词版本文件
│   │   ├── rag_qa_prompt.txt   # v2 原始提示词
│   │   ├── v1.txt ~ v6.txt     # 各版本提示词
│   │   └── changelog.md         # 版本变更记录
│   ├── golden/               # Golden 测试数据集
│   └── v1/ ~ v6/             # 各版本的评测结果
│       ├── goldens_final_eval.jsonl          # RAG 预测结果
│       └── goldens_final_eval_deepeval_scores.jsonl  # DeepEval 打分
├── src/                      # 源代码
│   └── utils.py              # HybridRAGPipeline
├── build_golden_data.py      # 构建 Golden 数据
├── run_golden_rag.py         # 运行 RAG 推理 (20条)
├── run_golden_rag_more.py    # 运行 RAG 推理 (52条扩展)
├── evaluate_golden_deepeval.py  # DeepEval 评测
├── dspy/                     # DSPy DYPS 自动提示词优化
│   ├── run_dyps.py           # 主入口脚本
│   ├── dyps/                 # 核心模块
│   │   ├── config.py         # 配置中心
│   │   ├── data.py           # 数据加载与切分
│   │   ├── models.py         # 教师/学生/DSPy LM 三种模型封装
│   │   ├── signatures.py     # DSPy 签名定义
│   │   ├── evaluator.py      # DeepEval 评测模块
│   │   └── teleprompter.py   # DYPS 优化器核心
│   └── test/                 # 测试脚本
```

---

## 评测指标说明

| 指标 | 说明 | 阈值 |
|------|------|------|
| **FaithfulnessMetric** | 答案是否忠实于 retrieval context，无幻觉 | 0.5 |
| **AnswerRelevancyMetric** | 答案是否切题地回应用户问题 | 0.5 |
| **ContextualRecallMetric** | 答案是否覆盖了预期答案中的关键内容 | 0.6 |

> **注意**：当前 AnswerRelevancyMetric 对 "无答案" 场景存在评分盲点，当模型正确输出"无答案"时，该指标可能给出 0 分。这是指标设计问题，不反映提示词质量。

---

## 提示词版本记录

| 版本 | 评测数据 | Overall | AnswerRelevancy | Faithfulness | ContextualRecall | 说明 |
|------|---------|---------|-----------------|-------------|------------------|------|
| v1 | 20条 | 0.9667 | 0.9750 | 0.9250 | 1.0000 | 基础版本 |
| v2 | 20条 | 0.9861 | 0.9583 | 1.0000 | 1.0000 | 信息→参考资料，强化无答案判定条件 |
| v3 | 20条 | 0.9647 | 0.9167 | 0.9775 | 1.0000 | 增加否定式约束，导致 AnswerRelevancy 下降 |
| v4 | 20条 | 0.9809 | 0.9417 | 1.0000 | 1.0000 | 否定式约束改为正向约束，修复赛道模式 case |
| **v5** | **52条** | **0.9174** | 0.8822 | 0.9436 | 0.9263 | 同 v2 提示词（无判断标准），扩大评测范围后整体下降 ~7% |
| **v6** | **52条** | **0.9416** | 0.9048 | **0.9840** | **0.9359** | v2 + **判断标准**约束，Faithfulness +4%，整体 +2.4% |

> v5 和 v6 在相同 52 条测试集上对比，v6 在 Overall、Faithfulness、ContextualRecall 三项均优于 v5。

---

## 提示词内容（v6 最新版）

```markdown
### 参考资料
{context}

### 任务
你是特斯拉电动汽车Model 3车型的用户手册问答系统，你具备上方【参考资料】中的知识。
请回答问题"{query}"，答案需要精准，语句通顺，并严格按照以下格式输出

{{答案}}【{{引用编号1}}, {{引用编号2}}, ...】
如果【参考资料】中没有与用户问题相关的内容，请说 "无答案" ，不允许在答案中添加编造成分。

判断标准：答案中的每一句话必须能直接回答用户的问题。如果一句话单独拿出来无法回应用户的提问，就不应包含在答案中。答案的每个信息点必须能在【参考资料】中找到直接对应的原文依据。
```

**关键差异（v5 vs v6）：** 仅在 v6 中增加了"判断标准"这一正向约束指令，使 Faithfulness 从 0.9436 提升至 0.9840。

---

## v6 失败 Case 分析（52条，4/52 失败 = 7.7%）

### Case 1 & 2：AnswerRelevancy=0（整体 0.3333）

| | 内容 |
|---|---|
| **问题** | "我想听里面女主持人唱的歌" / "放几首这会儿适合听的歌吧" |
| **预期答案** | 无答案 |
| **实际答案** | 无答案（正确） |
| **根因** | DeepEval AnswerRelevancy 指标设计缺陷，无法识别"无答案"为合法回复 |

**结论：指标问题，非提示词问题，无需修改提示词。**

---

### Case 3：AnswerRelevancy=0（整体 0.6667）

| | 内容 |
|---|---|
| **问题** | 充电接口闩锁被冻结时，如何融化冰？ |
| **预期答案** | 触摸屏打开后部除霜 + **充电接口入口加热器** |
| **实际答案** | 触摸屏打开后部除霜（缺失加热器部分） |
| **根因** | RAG 检索到的 context 不完整，未覆盖"充电接口入口加热器"内容 |

**结论：RAG 检索问题，非提示词问题。需优化 chunking 或扩大 topk。**

---

### Case 4：ContextualRecall=0（整体 0.6667）

| | 内容 |
|---|---|
| **问题** | 说出三个和车内灯光有关的部件 |
| **预期答案** | 顶灯、**危险警告灯**、氛围灯带 |
| **实际答案** | 脚部空间灯、顶灯、氛围灯 |
| **根因** | "危险警告灯"内容在检索 context 中不存在，RAG 未检索到 |

**结论：RAG 检索缺失问题，非提示词问题。需确认该 chunk 是否可被检索。**

---

## 核心结论

```
v6 失败分类：
  Case 1-2: 指标设计缺陷（指标层面）→ 无需修改提示词
  Case 3:   RAG 检索不完整（RAG层面）→ 优化 chunking / retrieval
  Case 4:   RAG 检索缺失（RAG层面）→ 优化 chunking / retrieval

→ 当前 v6 提示词本身已是最优解
→ 进一步优化方向：RAG pipeline 而非提示词迭代
```

---

## 评测数据构建方法

1. 从 676 条人工标注的特斯拉 QA 数据中，按类别分层抽样构建测试集
2. 各版本使用相同的 52 条测试集，保证横向可比性
3. 评测流程：RAG 推理 → DeepEval 打分 → 汇总分析

---

## 后续优化建议

| 优先级 | 方向 | 具体行动 |
|--------|------|---------|
| P0 | 评测指标优化 | 修复 DeepEval 对"无答案"场景的 AnswerRelevancy 计算方式 |
| P1 | RAG 检索优化 | 调查 Case 3/4 的 retrieval context，确认 chunk 覆盖度 |
| P2 | RAG chunking 优化 | 确认"危险警告灯"、"充电接口入口加热器"是否在知识库中且可检索 |
| P3 | 提示词维护 | 如有新场景需求，再在 v6 基础上做小幅调整 |
| P4 | DSPy 自动优化 | 已初步验证（见下方 DSPy 章节），后续可扩大 cold-start 样本量 |

---

## DSPy DYPS 自动提示词优化

### 背景

手动调优 v1~v6 迭代了 6 个版本，每个版本需要：
1. 构思提示词修改方案
2. 运行 RAG 推理（52 条 × 每次数分钟）
3. DeepEval 评测
4. 人工分析失败 case

这个流程耗时且依赖经验。**DYPS（Dynamic Prompt Selection）** 借助教师-学生架构，让强 API 模型自动指导学生模型优化提示词参数。

### 架构

```
教师 (deepseek-v4-flash API)
  │
  │ 评分 / 生成 hints / 参考答案
  ▼
学生 (本地 Qwen3-8B + BM25 + Milvus + BGE Reranker)
  │
  │ 动态注入: system_prompt, temperature, max_tokens
  ▼
vLLM 推理 → DeepEval 三维评测 → 教师反馈 → 下一轮参数
```

### 运行配置

| 参数 | 值 |
|------|-----|
| 教师模型 | deepseek-v4-flash (API) |
| 学生模型 | Qwen3-8B (本地 vLLM) |
| 评测指标 | AnswerRelevancy + Faithfulness + ContextualRecall |
| Golden 数据 | 676 条（train 70% / dev 15% / test 15%，按类别分层） |
| 教师胜率阈值 | 0.7（低于 1-0.7=0.3 时早停） |

### 评测结果


**测试集各维度得分：**

| 维度 | 分数 | 权重 |
|------|------|------|
| AnswerRelevancy（答案相关性） | 0.9239 | 0.3 |
| Faithfulness（忠诚度/无幻觉） | 0.9777 | 0.4 |
| ContextualRecall（上下文召回） | 0.9483 | 0.3 |


### 与手动调优 v6 横向对比

| | DYPS 自动优化 | v6 手动提示词 | 差异 |
|------|-------------|-------------|------|
| **Overall** | **0.9528** | 0.9416 | DYPS +1.1% |
| **Faithfulness** | 0.9777 | **0.9840** | v6 +0.6% |
| **AnswerRelevancy** | 0.9239 | 0.9048 | DYPS +1.9% |
| **ContextualRecall** | 0.9483 | 0.9359 | DYPS +1.2% |



### 运行方式

```bash
# 进入 DSPy 目录
cd dspy/

# 基础运行（默认 30 trials）
/root/autodl-tmp/conda/envs/rag/bin/python run_dyps.py

# 快速验证（仅 5 trials，跳过测试集评测）
/root/autodl-tmp/conda/envs/rag/bin/python run_dyps.py --num-trials 5 --skip-test-eval

# 仅检查配置不执行
/root/autodl-tmp/conda/envs/rag/bin/python run_dyps.py --dry-run

# 自定义初始提示词
/root/autodl-tmp/conda/envs/rag/bin/python run_dyps.py \
    --initial-prompt "你是特斯拉Model 3的专家问答系统..." \
    --num-trials 20
```

### 输出文件

运行后在 `data/dyps/` 生成：

| 文件 | 内容 |
|------|------|
| `run_summary.json` | 完整运行摘要（参数、最佳分、trial 历史、测试汇总） |
| `test_eval_results.jsonl` | 测试集每条记录的三维评测详情 |
| `dyps_YYYYMMDD_HHMMSS.log` | 完整运行日志 |
