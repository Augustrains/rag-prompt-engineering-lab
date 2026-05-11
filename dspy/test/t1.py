import os
import dspy

os.environ["LITELLM_FALLBACK_LOCALLY"] = "true"

lm = dspy.LM(
    os.environ["DOUBAO_MODEL_NAME"],
    api_key=os.environ["DOUBAO_API_KEY"],
    base_url=os.environ.get("DOUBAO_BASE_URL"),
)
dspy.configure(lm=lm)