
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline

load_dotenv()

model = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    model_kwargs={
        "local_files_only": True,
    },
pipeline_kwargs={
    "max_new_tokens": 32,
    "do_sample": False,
    "return_full_text": False,
},
)

prompt = PromptTemplate.from_template(
    "You are a helpful assistant.\n"
    "Question: {question}\n"
    "Answer:"
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({
    "question": "What is the capital of Peru?"
})

print("Model output:", repr(result))
print("Output length:", len(result))
