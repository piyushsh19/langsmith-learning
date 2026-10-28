
import operator
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langsmith import traceable

from langchain_huggingface import (
    HuggingFacePipeline,
    ChatHuggingFace,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

from langgraph.graph import StateGraph, START, END


# ==================================================
# 1. Setup
# ==================================================

load_dotenv()

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


# ==================================================
# 2. Load the local LLM
# ==================================================

llm_pipeline = HuggingFacePipeline.from_model_id(
    model_id=MODEL_ID,
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

# Wrap the local model as a chat model
model = ChatHuggingFace(llm=llm_pipeline)


# ==================================================
# 3. Structured output schema
# ==================================================

class EvaluationSchema(BaseModel):
    feedback: str = Field(
        description="Detailed and constructive feedback for the essay"
    )
    score: int = Field(
        description="Score out of 10",
        ge=0,
        le=10,
    )


# Validate model output against the Pydantic schema
parser = PydanticOutputParser(
    pydantic_object=EvaluationSchema
)


# ==================================================
# 4. Sample essay
# ==================================================

essay2 = """
India and AI Time

Now world change very fast because new tech call Artificial
Intel… something (AI). India also want become big in this AI
thing. If work hard, India can go top. But if no careful,
India go back.

India have many good. We have smart student, many engine-ear,
and good IT peoples. Big company like TCS, Infosys, Wipro
already use AI. Government also do program “AI for All”.
It want AI in farm, doctor place, school and transport.

In farm, AI help farmer know when to put seed, when rain come,
how stop bug. In health, AI help doctor see sick early.
In school, AI help student learn good. Government office use
AI to find bad people and work fast.

But problem come also. First is many villager no have phone
or internet. So AI not help them. Second, many people lose job
because AI and machine do work. Poor people get more bad.

One more big problem is privacy. AI need big big data.
Who take care? India still make data rule. If no strong rule,
AI do bad.

India must all people together – govern, school, company and
normal people. We teach AI and make sure AI not bad. Also talk
to other country and learn from them.

If India use AI good way, we become strong, help poor and
make better life. But if only rich use AI, and poor no get,
then big bad thing happen.

So, in short, AI time in India have many hope and many danger.
We must go right road. AI must help all people, not only some.
Then India grow big and world say "good job India".
"""


# ==================================================
# 5. LangGraph state
# ==================================================

class UPSCState(TypedDict, total=False):
    essay: str

    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str

    overall_feedback: str

    # Parallel nodes append their scores
    individual_scores: Annotated[list[int], operator.add]

    avg_score: float


# ==================================================
# 6. Helper: evaluate one essay dimension
# ==================================================

def evaluate_dimension(
    essay: str,
    dimension: str,
    instructions: str,
) -> EvaluationSchema:

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an experienced UPSC essay evaluator. "
            "Evaluate the essay fairly and constructively. "
            "Give a score from 0 to 10 based on the requested "
            "dimension. Explain strengths and weaknesses. "
            "Return only valid JSON matching this schema:\n"
            "{format_instructions}"
        ),
        (
            "human",
            "Evaluation dimension: {dimension}\n\n"
            "Evaluation criteria: {instructions}\n\n"
            "Essay:\n{essay}"
        ),
    ])

    chain = prompt | model | parser

    return chain.invoke({
        "dimension": dimension,
        "instructions": instructions,
        "essay": essay,
        "format_instructions": parser.get_format_instructions(),
    })


# ==================================================
# 7. Parallel evaluation nodes
# ==================================================

@traceable(
    name="evaluate_language_fn",
    tags=["dimension:language"],
    metadata={"dimension": "language"},
)
def evaluate_language(state: UPSCState):
    result = evaluate_dimension(
        essay=state["essay"],
        dimension="Language quality",
        instructions=(
            "Assess grammar, spelling, vocabulary, sentence "
            "construction, word choice, and formal writing style."
        ),
    )

    return {
        "language_feedback": result.feedback,
        "individual_scores": [result.score],
    }


@traceable(
    name="evaluate_analysis_fn",
    tags=["dimension:analysis"],
    metadata={"dimension": "analysis"},
)
def evaluate_analysis(state: UPSCState):
    result = evaluate_dimension(
        essay=state["essay"],
        dimension="Depth of analysis",
        instructions=(
            "Assess depth of reasoning, supporting examples, "
            "critical thinking, coverage of the topic, "
            "balanced arguments, and practical solutions."
        ),
    )

    return {
        "analysis_feedback": result.feedback,
        "individual_scores": [result.score],
    }


@traceable(
    name="evaluate_thought_fn",
    tags=["dimension:clarity"],
    metadata={"dimension": "clarity_of_thought"},
)
def evaluate_thought(state: UPSCState):
    result = evaluate_dimension(
        essay=state["essay"],
        dimension="Clarity of thought",
        instructions=(
            "Assess logical flow, organization of ideas, "
            "coherence between paragraphs, clarity of arguments, "
            "and quality of the conclusion."
        ),
    )

    return {
        "clarity_feedback": result.feedback,
        "individual_scores": [result.score],
    }


# ==================================================
# 8. Final evaluation node
# ==================================================

@traceable(
    name="final_evaluation_fn",
    tags=["aggregate"],
)
def final_evaluation(state: UPSCState):

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a UPSC essay mentor. Summarize the supplied "
            "evaluation feedback into constructive overall feedback. "
            "Highlight strengths, the most important weaknesses, "
            "and actionable improvements. Do not invent scores."
        ),
        (
            "human",
            "Language feedback:\n{language_feedback}\n\n"
            "Analysis feedback:\n{analysis_feedback}\n\n"
            "Clarity feedback:\n{clarity_feedback}\n\n"
            "Individual scores: {scores}\n"
            "Average score: {avg_score}/10"
        ),
    ])

    overall_chain = prompt | model | StrOutputParser()

    scores = state.get("individual_scores", [])
    avg_score = (
        sum(scores) / len(scores)
        if scores
        else 0.0
    )

    overall = overall_chain.invoke({
        "language_feedback": state.get("language_feedback", ""),
        "analysis_feedback": state.get("analysis_feedback", ""),
        "clarity_feedback": state.get("clarity_feedback", ""),
        "scores": scores,
        "avg_score": round(avg_score, 2),
    })

    return {
        "overall_feedback": overall,
        "avg_score": avg_score,
    }


# ==================================================
# 9. Build LangGraph workflow
# ==================================================

graph = StateGraph(UPSCState)

graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation", final_evaluation)

# Fan-out: run the three evaluations in parallel
graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")

# Join: wait for all three evaluation nodes
graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_thought", "final_evaluation")

graph.add_edge("final_evaluation", END)

workflow = graph.compile()


# ==================================================
# 10. Run the workflow
# ==================================================

if __name__ == "__main__":

    print(f"Loading local model: {MODEL_ID}")
    print("Evaluating essay...\n")

    result = workflow.invoke(
        {"essay": essay2},
        config={
            "run_name": "evaluate_upsc_essay_local",
            "tags": [
                "essay",
                "langgraph",
                "evaluation",
                "local-llm",
            ],
            "metadata": {
                "essay_length": len(essay2),
                "model": MODEL_ID,
                "dimensions": [
                    "language",
                    "analysis",
                    "clarity",
                ],
            },
        },
    )

    print("\n========== EVALUATION RESULTS ==========\n")

    print("LANGUAGE FEEDBACK:")
    print(result.get("language_feedback", ""))

    print("\nANALYSIS FEEDBACK:")
    print(result.get("analysis_feedback", ""))

    print("\nCLARITY OF THOUGHT FEEDBACK:")
    print(result.get("clarity_feedback", ""))

    print("\nOVERALL FEEDBACK:")
    print(result.get("overall_feedback", ""))

    print("\nINDIVIDUAL SCORES:")
    print(result.get("individual_scores", []))

    print("\nAVERAGE SCORE:")
    print(f"{result.get('avg_score', 0.0):.2f}/10")
