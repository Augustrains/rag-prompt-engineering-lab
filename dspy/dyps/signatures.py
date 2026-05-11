# -*- coding: utf-8 -*-
"""
DSPy Signatures for RAG task.

Signatures define the input/output interface for modules.
We define:
1. RAG signature - the main task signature
2. Teacher grading signature - for teacher model to grade student answers
3. Hint generation signature - for teacher to provide hints
"""

import dspy


# ──────────────────────────── 主 RAG 签名 ────────────────────────────

class RAGSignature(dspy.Signature):
    """
    Signature for RAG question answering task.

    Input:
        - input: The user's question / query
        - retrieval_context: The context retrieved from the RAG pipeline

    Output:
        - answer: The generated answer based on the retrieval context
    """
    input = dspy.InputField(desc="用户的问题或查询")
    retrieval_context = dspy.InputField(
        desc="从RAG检索到的上下文信息列表"
    )
    answer = dspy.OutputField(
        desc="基于检索上下文生成的准确、完整的答案",
        prefix="答案:"
    )


# ──────────────────────────── 教师评分签名 ────────────────────────────

class TeacherGradingSignature(dspy.Signature):
    """
    Signature for teacher model to grade student answers.

    Input:
        - input: The user's question
        - expected_output: The expected gold standard answer
        - student_answer: The student's generated answer

    Output:
        - reasoning: Explanation of the grading
        - answer_relevancy: Score (0-1) for answer relevancy
        - faithfulness: Score (0-1) for faithfulness
        - contextual_recall: Score (0-1) for contextual recall
        - overall_score: Weighted overall score
    """
    input = dspy.InputField(desc="用户的问题")
    expected_output = dspy.InputField(desc="标准答案")
    student_answer = dspy.InputField(desc="学生模型生成的答案")

    reasoning = dspy.OutputField(desc="评分推理过程")
    answer_relevancy = dspy.OutputField(desc="答案相关性分数 (0-1)")
    faithfulness = dspy.OutputField(desc="忠诚度分数 (0-1)")
    contextual_recall = dspy.OutputField(desc="上下文召回分数 (0-1)")
    overall_score = dspy.OutputField(desc="综合评分 (0-1)")


# ──────────────────────────── 教师提示签名 ────────────────────────────

class HintGenerationSignature(dspy.Signature):
    """
    Signature for teacher to generate hints to improve prompts.

    Input:
        - input: The user's question
        - retrieval_context: Retrieved context
        - student_answer: Student's current answer
        - expected_output: Expected answer

    Output:
        - hint: A hint on how to improve the answer
        - improved_answer: A better answer
        - reasoning: Why the hint is useful
    """
    input = dspy.InputField(desc="用户的问题")
    retrieval_context = dspy.InputField(desc="检索到的上下文")
    student_answer = dspy.InputField(desc="学生当前的答案")
    expected_output = dspy.InputField(desc="期望的标准答案")

    hint = dspy.OutputField(desc="改进提示词或答案的提示")
    improved_answer = dspy.OutputField(desc="改进后的答案示例")
    reasoning = dspy.OutputField(desc="提示的推理说明")


# ──────────────────────────── 组合签名 ────────────────────────────

class CompositeRAGSignature(dspy.Signature):
    """
    Extended RAG signature that includes metadata for optimization.

    Used during DYPS optimization to track additional fields.
    """
    input = dspy.InputField(desc="用户的问题")
    retrieval_context = dspy.InputField(desc="RAG检索到的上下文")
    expected_output = dspy.InputField(desc="期望的标准答案（用于评测）")

    answer = dspy.OutputField(desc="生成的答案")
    score = dspy.OutputField(desc="质量评分 (0-1)")


# ──────────────────────────── 工具函数 ────────────────────────────

def create_rag_module(signature: type = RAGSignature):
    """
    Create a basic RAG module with a given signature.

    This is a simple module that uses ChainOfThought for reasoning.
    """
    return dspy.ChainOfThought(signature)