from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from ..core.config import get_settings

settings = get_settings()

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
"""

class RAGService:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            base_url=settings.SCW_GENERATIVE_APIs_ENDPOINT,
            api_key=settings.SCW_SECRET_KEY,
            model="llama-3.1-8b-instruct",
        )
        self.prompt = hub.pull("rlm/rag-prompt")
        
    def get_rag_chain(self):
        retriever = self.vector_store.as_retriever()
        return (
            {"context": retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    async def generate_response(self, query: str, stream: bool = False):
        chain = self.get_rag_chain()
        if stream:
            return chain.stream(query)
        return chain.invoke(query)