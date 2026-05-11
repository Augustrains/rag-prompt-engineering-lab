# 3. 跑通官方 RAG 示例（照抄即可，5分钟跑通）
import dspy
import os
import dspy

os.environ["LITELLM_FALLBACK_LOCALLY"] = "true"

lm = dspy.LM(
    model="openai/deepseek-v4-flash",
    api_key=os.environ["DOUBAO_API_KEY"],
    base_url=os.environ.get("DOUBAO_BASE_URL"),
)
dspy.configure(lm=lm)


# 定义签名（相当于你的 rag_qa_prompt 规则）
class GenerateAnswer(dspy.Signature):
    context = dspy.InputField(desc="提供给模型的上下文信息")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="简洁准确的答案")

# 用 ChainOfThought 模块
cot = dspy.ChainOfThought(GenerateAnswer)
result = cot(context="特斯拉Model 3续航500公里", question="Model 3续航多少？")
print(result.answer)

