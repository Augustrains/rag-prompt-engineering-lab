# -*- coding: utf-8 -*-
"""
DSPy 签名（Signatures）定义
===========================

签名是 DSPy 框架的核心抽象：它只定义模块的输入/输出接口，
而不规定具体如何实现。DSPy 编译器会根据签名自动选择策略。

本项目定义了四种签名：

  1. RAGSignature — 主 RAG 问答任务
     input + retrieval_context → answer

  2. TeacherGradingSignature — 教师模型评分
     input + expected_output + student_answer → 三维分数 + 综合分 + 推理

  3. HintGenerationSignature — 教师生成改进提示
     input + context + student_answer + expected_output → hint + improved_answer

  4. CompositeRAGSignature — 扩展 RAG 签名（含评测元数据）
     input + context + expected_output → answer + score

设计原则：
  - InputField / OutputField 的 desc 参数不仅用于文档，还影响 DSPy 编译器
    如何构造 prompt（DSPy 会将 desc 嵌入到给 LLM 的指令中）
  - prefix 参数控制输出字段的前缀标记（如 "答案:"）
"""

import dspy


# ═══════════════════════════════════════════════════════════════════════════════
# 主 RAG 签名
# ═══════════════════════════════════════════════════════════════════════════════

class RAGSignature(dspy.Signature):
    """
    标准 RAG 问答签名。

    定义了一个典型的检索增强生成流程：
      用户提问 → 检索相关文档 → 基于文档生成答案

    输入：
      - input: 用户的自然语言问题
      - retrieval_context: RAG pipeline 检索到的上下文文档列表

    输出：
      - answer: 基于检索上下文生成的答案（准确、完整）
    """
    input = dspy.InputField(desc="用户的问题或查询")
    retrieval_context = dspy.InputField(
        desc="从RAG检索到的上下文信息列表"
    )
    answer = dspy.OutputField(
        desc="基于检索上下文生成的准确、完整的答案",
        prefix="答案:"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 教师评分签名
# ═══════════════════════════════════════════════════════════════════════════════

class TeacherGradingSignature(dspy.Signature):
    """
    教师模型评分签名。

    定义了教师如何给学生答案打分的接口。
    包含推理过程（reasoning）以便理解和调试评分逻辑。

    输入：
      - input: 用户问题
      - expected_output: 标准答案（golden 数据中的期望输出）
      - student_answer: 学生模型的生成结果

    输出：
      - reasoning: 评分推理过程（帮助理解评分依据）
      - answer_relevancy: 答案相关性分数（0-1，是否切题）
      - faithfulness: 忠诚度分数（0-1，是否忠实于上下文、无幻觉）
      - contextual_recall: 上下文召回分数（0-1，是否覆盖关键信息）
      - overall_score: 综合评分（加权平均）
    """
    input = dspy.InputField(desc="用户的问题")
    expected_output = dspy.InputField(desc="标准答案")
    student_answer = dspy.InputField(desc="学生模型生成的答案")

    reasoning = dspy.OutputField(desc="评分推理过程")
    answer_relevancy = dspy.OutputField(desc="答案相关性分数 (0-1)")
    faithfulness = dspy.OutputField(desc="忠诚度分数 (0-1)")
    contextual_recall = dspy.OutputField(desc="上下文召回分数 (0-1)")
    overall_score = dspy.OutputField(desc="综合评分 (0-1)")


# ═══════════════════════════════════════════════════════════════════════════════
# 教师提示生成签名
# ═══════════════════════════════════════════════════════════════════════════════

class HintGenerationSignature(dspy.Signature):
    """
    教师生成改进提示的签名。

    当学生答案质量不佳时，教师分析问题并提供改进建议。
    这是 DYPS 优化循环中"提示词驱动探索"策略的依据。

    输入：
      - input: 用户问题
      - retrieval_context: 检索到的上下文（学生能看到的信息）
      - student_answer: 学生的当前答案（有问题的答案）
      - expected_output: 标准答案（目标）

    输出：
      - hint: 如何改进提示词或答案的具体建议
      - improved_answer: 更好的答案示例（给学生做参考）
      - reasoning: 为什么这个 hint 有用
    """
    input = dspy.InputField(desc="用户的问题")
    retrieval_context = dspy.InputField(desc="检索到的上下文")
    student_answer = dspy.InputField(desc="学生当前的答案")
    expected_output = dspy.InputField(desc="期望的标准答案")

    hint = dspy.OutputField(desc="改进提示词或答案的提示")
    improved_answer = dspy.OutputField(desc="改进后的答案示例")
    reasoning = dspy.OutputField(desc="提示的推理说明")


# ═══════════════════════════════════════════════════════════════════════════════
# 组合签名（含元数据）
# ═══════════════════════════════════════════════════════════════════════════════

class CompositeRAGSignature(dspy.Signature):
    """
    扩展 RAG 签名。

    相比 RAGSignature，额外包含：
      - expected_output（用于评测对比）
      - score（质量评分，用于优化追踪）

    主要用于 DYPS 优化循环中，需要同时记录预期答案和评分的场景。
    """
    input = dspy.InputField(desc="用户的问题")
    retrieval_context = dspy.InputField(desc="RAG检索到的上下文")
    expected_output = dspy.InputField(desc="期望的标准答案（用于评测）")

    answer = dspy.OutputField(desc="生成的答案")
    score = dspy.OutputField(desc="质量评分 (0-1)")


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def create_rag_module(signature: type = RAGSignature):
    """
    基于签名创建一个简单的 RAG 模块。

    使用 DSPy 内置的 ChainOfThought 包装给定的签名，
    ChainOfThought 会让 LLM 在输出答案前先进行推理（类似 CoT prompting）。

    注意：本项目实际使用的是 StudentRAG（自定义 module），
    此函数主要用于测试或作为简单基线。
    """
    return dspy.ChainOfThought(signature)
