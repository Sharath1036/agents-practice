from os import getenv
from dotenv import load_dotenv
from agno.agent import Agent
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.qdrant import Qdrant
from agno.vectordb.mongodb import MongoDb
from agno.embedder.ollama import OllamaEmbedder

class PDFKnowledgeAgent:
    def __init__(self, urls: list[str], database: str):
        load_dotenv(override=True)

        self.collection_name = "vector-embeddings"
        self.qdrant_url = getenv("QDRANT_URL")
        self.qdrant_api_key = getenv("QDRANT_API_KEY")
        self.mongo_connection_string = getenv("MONGO_CONNECTION_STRING")
        self.embedder = OllamaEmbedder(id="openhermes", host='http://localhost:11434/', timeout=1000.0)
        self.vector_db = self._init_vector_db(database)
        self.knowledge_base = self._init_knowledge_base(urls)
        self.agent = self._init_agent()

    def _init_vector_db(self, database: str) -> Qdrant:
        if database == 'Qdrant':
            return Qdrant(
                collection=self.collection_name,
                url=self.qdrant_url,
                api_key=self.qdrant_api_key
            )
        else:    
            return MongoDb(
                collection_name=self.collection_name,
                db_url=self.mongo_connection_string,
                database="agno",
            )

    def _init_knowledge_base(self, urls: list[str]) -> PDFUrlKnowledgeBase:
        return PDFUrlKnowledgeBase(
            urls=urls,
            vector_db=self.vector_db,
            embedder=self.embedder
        )

    def _init_agent(self) -> Agent:
        return Agent(
            knowledge=self.knowledge_base,
            show_tool_calls=True
        )

    def embed_sample(self, text: str):
        embeddings = self.embedder.get_embedding(text)
        print(f"Embeddings (first 5 values): {embeddings[:5]}")
        print(f"Embedding Dimension: {len(embeddings)}")

    def load_documents(self, recreate: bool = False):
        self.knowledge_base.load(recreate=recreate)

    def query(self, prompt: str, markdown: bool = True):
        self.agent.print_response(prompt, markdown=markdown)


if __name__ == "__main__":
    urls = [
        "https://www.scollingsworthenglish.com/uploads/3/8/4/2/38422447/garth_stein_-_the_art_of_racing_in_the_rain.pdf"
    ]

    database = 'MongoDb' 
    runner = PDFKnowledgeAgent(urls=urls, database=database)

    runner.embed_sample("The quick brown fox jumps over the lazy dog.")
    runner.load_documents(recreate=False)
    runner.query("How did Eve die?", markdown=True)

