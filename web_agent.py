from agno.agent import Agent
from agno.models.groq import Groq
from agno.models.ollama import Ollama
from agno.storage.sqlite import SqliteStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.playground import Playground
from dotenv import load_dotenv
from os import getenv
from agno.playground import serve_playground_app

load_dotenv(override=True)
api_key = getenv("GROQ_API_KEY")
storage_file = "tmp/agents.db"

def create_web_agent(tools: str, client: str, model_id: list) -> Agent:
    """Create and configure a web agent with Groq model and DuckDuckGo tools"""
    return Agent(
        name="Web Agent",
        model=client(id=model_id),
        tools=[DuckDuckGoTools()],
        instructions=[
            "Always include sources",
            "If you're not sure about something, say so",
            "Be concise and clear in your responses"
        ],
        storage=SqliteStorage(table_name="web_agent", db_file=storage_file),
        add_datetime_to_instructions=True,
        add_history_to_messages=True,
        num_history_responses=3,  # Reduced from 5 to prevent context overflow
        markdown=True,
    )

def create_playground_app(agent: Agent):
    """Create a Playground app with the given agent"""
    app = Playground(agents=[agent]).get_app()
    return app

def main():
    """Main function to run the application"""
    tools = [DuckDuckGoTools()]
    models = {"ollama": "gemma2:2b", "groq": "llama-3.3-70b-versatile"}
    web_agent = create_web_agent(tools, Groq, models["groq"])
        
    app = create_playground_app(web_agent)
    return app

# Create the app instance at module level
app = main()

if __name__ == "__main__":
    serve_playground_app("main:app", reload=True, port=7172)