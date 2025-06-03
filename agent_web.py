from agno.agent import Agent
from agno.models.groq import Groq
from agno.models.ollama import Ollama
from agno.storage.sqlite import SqliteStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.playground import Playground
from dotenv import load_dotenv
from os import getenv
from agno.playground import serve_playground_app
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(override=True)

class WebAgentApp:
    """A class to manage the web agent application"""
    
    def __init__(self, storage_file: str = "tmp/agents.db"):
        self.storage_file = storage_file
        self.api_key = getenv("GROQ_API_KEY")
    
    def create_web_agent(self, tools: list, client: type, model_id: str) -> Agent:
        """Create and configure a web agent with specified model and tools"""
        try:
            agent = Agent(
                name="Web Agent",
                model=client(id=model_id),
                tools=tools,
                instructions=[
                    "Always include sources",
                    "If you're not sure about something, say so",
                    "Be concise and clear in your responses",
                    "If you encounter an error, explain what might have caused it"
                ],
                storage=SqliteStorage(table_name="web_agent", db_file=self.storage_file),
                add_datetime_to_instructions=True,
                add_history_to_messages=True,
                num_history_responses=3,  # Reduced to prevent context overflow
                markdown=True,
            )
            return agent
        except Exception as e:
            logger.error(f"Failed to create agent: {str(e)}")
            raise
    
    def create_playground_app(self, agent: Agent):
        """Create a Playground app with the given agent"""
        try:
            return Playground(agents=[agent]).get_app()
        except Exception as e:
            logger.error(f"Failed to create playground app: {str(e)}")
            raise

def main():
    """Main function to run the application"""
    try:
        web_agent_app = WebAgentApp()
        tools = [DuckDuckGoTools()]
        
        # Try Groq first, fall back to Ollama if it fails
        try:
            agent = web_agent_app.create_web_agent(tools, Groq, "llama-3.3-70b-versatile")
            logger.info("Using Groq model")
        except Exception as e:
            logger.warning(f"Groq failed, falling back to Ollama: {str(e)}")
            agent = web_agent_app.create_web_agent(tools, Ollama, "gemma2:2b")
            logger.info("Using Ollama model")
        
        app = web_agent_app.create_playground_app(agent)
        return app
    except Exception as e:
        logger.error(f"Application failed to initialize: {str(e)}")
        raise

# Create the app instance at module level
app = main()

if __name__ == "__main__":
    try:
        serve_playground_app("agent_web:app", reload=True, port=7172)
    except Exception as e:
        logger.error(f"Failed to serve application: {str(e)}")