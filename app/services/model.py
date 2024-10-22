from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from ..core.config import get_settings
from ..core.logging import setup_lunary
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

settings = get_settings()
MODEL_LLM = "llama-3.1-8b-instruct"


class MODELService:
    def __init__(self, vector_store):
        self.retriever = vector_store.as_retriever()
        self.llm = ChatOpenAI(
            base_url=settings.SCW_GENERATIVE_APIs_ENDPOINT,
            api_key=settings.SCW_SECRET_KEY,
            model=MODEL_LLM,
            callbacks=[setup_lunary()],
        )

    def rag_setup_query(self):
        prompt = hub.pull("rlm/rag-prompt")
        return (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def object_to_langchain_conv(self, objects, sys_prompt=""):

        langchain_conv = [SystemMessage(content=sys_prompt)] if sys_prompt else []

        langchain_conv += [
            HumanMessage(content=obj.content) if obj.role == "user" else AIMessage(content=obj.content)
            for obj in objects
        ]
        return langchain_conv

    async def stream_response(self, messages, enable_rag=True):
        query = str(messages[-1].content)
        conv = self.object_to_langchain_conv(messages)

        if enable_rag:
            rag_chain = self.rag_setup_query()
            return rag_chain.stream(query)

        else:
            return self.llm.stream(conv)
        
    def generate_response(self, messages, enable_rag=True):
        conv = self.object_to_langchain_conv(messages)
        query = str(messages[-1].content)

        if enable_rag:
            rag_chain = self.rag_setup_query()
            return rag_chain.invoke(query)

        else:
            return self.llm.invoke(conv).content
