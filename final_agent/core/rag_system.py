import uuid
from db.vector_db_manager import VectorDbManager
from db.parent_store_manager import ParentStoreManager
from document_chunker import DocumentChuncker
from rag_agent.tools import ToolFactory
from rag_agent.graph import create_agent_graph
from utils import load_config
from langchain_openai import ChatOpenAI

cfg =  load_config()

class RAGSystem:
    
    def __init__(self, collection_name=cfg["rag"]["child_collection"]):
        self.collection_name = collection_name
        self.vector_db = VectorDbManager()
        self.parent_store = ParentStoreManager()
        self.chunker = DocumentChuncker()
        self.agent_graph = None
        self.thread_id = str(uuid.uuid4())
        self.recursion_limit = 50
        
    def initialize(self):
        self.vector_db.create_collection(self.collection_name)
        collection = self.vector_db.get_collection(self.collection_name)
        
        llm = ChatOpenAI(
            model=cfg["models"]["llm_model"],
            base_url="http://100.67.127.53:8000/v1",
            api_key="sk-test",
            temperature=cfg["models"]["llm_temperature"],
            timeout= cfg["models"]["llm_timeout"],
        )
        tools = ToolFactory(collection).create_tools()
        self.agent_graph = create_agent_graph(llm, tools)
        
    def get_config(self):
        return {"configurable": {"thread_id": self.thread_id}, "recursion_limit": self.recursion_limit}
    
    def reset_thread(self):
        try:
            self.agent_graph.checkpointer.delete_thread(self.thread_id)
        except Exception as e:
            print(f"Warning: Could not delete thread {self.thread_id}: {e}")
        self.thread_id = str(uuid.uuid4())