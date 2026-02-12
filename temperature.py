from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
     temperature=1,
     max_new_tokens = 50
)
model = ChatHuggingFace(llm = llm)
result = model.invoke("how are you")

print(result.content)
