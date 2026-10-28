import os
os.environ['LANGCHAIN_PROJECT'] = 'Sequential LLM App'

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

load_dotenv()

# Step 1: Load the local TinyLlama model
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    model_kwargs={
        "local_files_only": True,
    },
    pipeline_kwargs={
        "max_new_tokens": 256,
        "do_sample": False,
        "return_full_text": False,
    },
)

# Wrap the local pipeline as a chat model
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

# Step 2: Generate a report
prompt1 = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful research assistant. "
        "Write a concise report with an introduction, "
        "key causes, impacts, and possible solutions. "
        "Do not invent statistics."
    ),
    ("human", "Write a report about {topic}."),
])

# Step 3: Summarize the report
prompt2 = ChatPromptTemplate.from_messages([
    (
        "system",
        "Summarize the supplied report into exactly "
        "five concise bullet points. Do not add questions "
        "or unrelated information."
    ),
    ("human", "Report:\n{text}"),
])

# Step 4: Build the sequential chain
report_chain = prompt1 | model | parser
summary_chain = prompt2 | model | parser

chain = report_chain | (
    lambda report: summary_chain.invoke({"text": report})
)
config = {
    'tags': ['llm app',' report generation', 'summarization'],
    'metadata':{'model1':'for genration'}
}
# Step 5: Execute
result = chain.invoke({
    "topic": "Unemployment in India",
},config =config)

print("\n===== FIVE-POINT SUMMARY =====\n")
print(result)
