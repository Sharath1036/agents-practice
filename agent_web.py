from agno.agent import Agent
from agno.models.groq import Groq
from agno.models.ollama import Ollama
from agno.storage.sqlite import SqliteStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.playground import Playground, serve_playground_app
from dotenv import load_dotenv
from os import getenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

class WebAgentService:
    """Class-based web agent application"""

    def __init__(self, storage_file: str = "tmp/agents.db"):
        self.storage_file = storage_file
        self.api_key = getenv("GROQ_API_KEY")
        self.agent = None
        self.app = None

    def _get_tools(self) -> list:
        return [DuckDuckGoTools()]

    def _create_agent(self, tools: list, client: type, model_id: str) -> Agent:
        """Internal method to create an agent"""
        return Agent(
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
            num_history_responses=3,
            markdown=True,
        )

    def initialize_agent(self):
        """Attempt to initialize the agent with Groq, fallback to Ollama"""
        tools = self._get_tools()
        try:
            self.agent = self._create_agent(tools, Groq, "llama-3.3-70b-versatile")
            logger.info("Using Groq model")
        except Exception as e:
            logger.warning(f"Groq failed, falling back to Ollama: {str(e)}")
            self.agent = self._create_agent(tools, Ollama, "gemma2:2b")
            logger.info("Using Ollama model")

    def create_app(self):
        """Create the web app interface using Playground"""
        if not self.agent:
            raise ValueError("Agent has not been initialized.")
        try:
            self.app = Playground(agents=[self.agent]).get_app()
        except Exception as e:
            logger.error(f"Failed to create playground app: {str(e)}")
            raise

    def run(self, host_string: str = "agent_web:app", port: int = 7172, reload: bool = True):
        """Run the playground web server"""
        if not self.app:
            raise ValueError("App has not been created.")
        try:
            serve_playground_app(host_string, reload=reload, port=port)
        except Exception as e:
            logger.error(f"Failed to serve application: {str(e)}")
            raise


# Instantiate and initialize everything
web_service = WebAgentService()
web_service.initialize_agent()
web_service.create_app()

# Expose the FastAPI app
app = web_service.app

if __name__ == "__main__":
    web_service.run()
