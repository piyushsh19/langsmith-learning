
import os
import json
import requests

from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_huggingface import HuggingFacePipeline
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

load_dotenv()

# --------------------------------------------------
# Step 1: Initialize tools
# --------------------------------------------------

search_tool = DuckDuckGoSearchRun()


@tool
def get_weather_data(city: str) -> str:
    """Fetch the current weather data for a given city."""

    api_key = os.getenv("WEATHERSTACK_API_KEY")

    if not api_key:
        return "Weather service unavailable: WEATHERSTACK_API_KEY is not configured."

    try:
        response = requests.get(
            "https://api.weatherstack.com/current",
            params={
                "access_key": api_key,
                "query": city,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            return json.dumps(data["error"])

        current = data.get("current", {})
        location = data.get("location", {})

        return json.dumps({
            "city": location.get("name", city),
            "country": location.get("country"),
            "temperature_c": current.get("temperature"),
            "weather_description": current.get("weather_descriptions"),
            "humidity": current.get("humidity"),
            "feelslike_c": current.get("feelslike"),
        })

    except requests.RequestException as exc:
        return f"Weather request failed: {exc}"
    except ValueError:
        return "Weather service returned invalid JSON."


tools = [search_tool, get_weather_data]


# --------------------------------------------------
# Step 2: Load your local LLM
# --------------------------------------------------

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-3B-Instruct",
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


# --------------------------------------------------
# Step 3: Pull the ReAct prompt
# --------------------------------------------------

prompt = hub.pull("hwchase17/react")


# --------------------------------------------------
# Step 4: Create the ReAct agent
# --------------------------------------------------

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,
)


# --------------------------------------------------
# Step 5: Run the agent
# --------------------------------------------------

if __name__ == "__main__":
    question = "What is the current temperature in Gurgaon?"

    response = agent_executor.invoke({
        "input": question
    })

    print("\n===== AGENT ANSWER =====\n")
    print(response["output"])
