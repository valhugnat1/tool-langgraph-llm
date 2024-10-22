
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from ..core.config import get_settings
from ..core.logging import setup_lunary
from langchain_core.messages import HumanMessage, AIMessage

settings = get_settings()
MODEL_LLM="llama-3.1-8b-instruct"

class MODELService:

    def __init__(self, vector_store):
        self.retriever = vector_store.as_retriever()
        self.llm = ChatOpenAI(
            base_url=settings.SCW_GENERATIVE_APIs_ENDPOINT,
            api_key=settings.SCW_SECRET_KEY,
            model=MODEL_LLM,
            callbacks=[setup_lunary()],
        )

    def rag_setup_query (self): 
        prompt = hub.pull("rlm/rag-prompt")
        return (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    async def stream_response(self, messages, enable_rag=True):

        query = str(messages[-1].content)

        if enable_rag :
            rag_chain = self.rag_setup_query()
            return rag_chain.stream(query)

        else: 
            return self.llm.stream(query)

    
    def generate_response(self, messages, enable_rag=True):

        query = str(messages[-1].content)

        if enable_rag :
            rag_chain = self.rag_setup_query()
            return rag_chain.invoke(query)
        
        else: 
            print (query)
            return self.llm.invoke(query).content

