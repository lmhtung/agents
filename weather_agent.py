import os
# from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from langchain.agents import create_agent
import requests
import arrow
import uuid
# from langgraph.checkpoint.sqlite import SqliteSaver

thread_id = str(uuid.uuid4())

os.environ["OPENAI_API_KEY"] = "sk-test"
os.environ["OPENWEATHER_API_KEY"] = "bf6dd17a-0336-11f1-b866-0242ac120004-bf6dd224-0336-11f1-b866-0242ac120004"

# Define system prompt
SYSTEM_PROMPT = """You are an expert weather forecaster.

You have access to three tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location. """

# Define Context schema
@dataclass
class Context():
    user_role: str

# Define tools
@tool
def geocode(city: str):
    """Convert city name to latitude and longitude using OpenStreetMap."""
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={
            "q": city,
            "format": "json",
            "limit": 1
        },
        headers={"User-Agent": "weather-agent"}
    )
    data = resp.json()
    return float(data[0]["lat"]), float(data[0]["lon"])

def get_weather(lat: float, lng: float):
    start = arrow.now().floor("hour")
    end = arrow.now().shift(hours=6)

    params = [
        "airTemperature",
        "cloudCover",
        "humidity",
        "windSpeed",
        "precipitation",
    ]

    resp = requests.get(
        "https://api.stormglass.io/v2/weather/point",
        params={
            "lat": lat,
            "lng": lng,
            "params": ",".join(params),
            "start": start.to("UTC").timestamp(),
            "end": end.to("UTC").timestamp(),
        },
        headers={
            "Authorization": os.environ["OPENWEATHER_API_KEY"]
        },
        timeout=10
    )

    resp.raise_for_status()
    return resp.json()

def pick_value(d: dict):
    for src in ["noaa", "icon", "meteo"]:
        if src in d:
            return d[src]
    return None

@tool
def get_weather_for_location(lat: float, lng: float) -> dict:
    """Get weather using Stormglass API"""
    data = get_weather(lat, lng)

    hour = data["hours"][0]

    return {
        "temperature": pick_value(hour["airTemperature"]),
        "wind_speed": pick_value(hour["windSpeed"]),
        "humidity": pick_value(hour["humidity"]),
        "cloud": pick_value(hour["cloudCover"]),
    }


# Define Reponse Format
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    punny_response : str
    weather_conditions : str | None = None
    wind_speed : str | None = None
    humidity : str | None = None



# Configure model
llm = ChatOpenAI(
    model="qwen3-8b-fp8",
    base_url="http://100.67.127.53:8000/v1",
    temperature=0.5,
    timeout = 30,
)

# Define memory
# checkpointer = SqliteSaver("weather_agent_memory.db")
checkpointer = InMemorySaver()

agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    tools=[geocode, get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)
# Define configure
config = {"configurable": {"thread_id": thread_id}}

# Main
print("Chat demo")
while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Thoát chat!")
        break

    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
        context=Context(user_role= "Student"),
    )

    # In kết quả có structured output
    print("Assistant: " , end="")
    print(response['structured_response'].punny_response)
