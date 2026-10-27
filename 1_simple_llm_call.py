
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline

load_dotenv()

# Load your downloaded model locally.
# No OpenAI API call is made.
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    model_kwargs={
        "local_files_only": True,
    },
    pipeline_kwargs={
        "max_new_tokens": 128,
        "do_sample": False,
        "return_full_text": False,
    },
)

# Define the prompt
prompt = PromptTemplate.from_template(
    "Answer the following question clearly and briefly: {question}"
)

# Define the output parser
parser = StrOutputParser()

# Chain: prompt → local LLM → parser
chain = prompt | llm | parser

# Execute the chain
result = chain.invoke(
    {"question": "What is the capital of Peru?"}
)

print("Model output:", result)
