from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from ..core.config import get_settings
from ..core.logging import setup_lunary
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate

settings = get_settings()
MODEL_BASE_LLM = "llama-3.1-8b-instruct"


class MODELService:
    def __init__(self, vector_store, request):
        self.retriever = vector_store.as_retriever()
        self.engine_ask = request.model
        self.query = str(request.messages[-1].content)
        self.conv = self.object_to_langchain_conv(request.messages)
        self.qa_history = self.conv[:-1]

        self.llm = ChatOpenAI(
            base_url=settings.SCW_GENERATIVE_APIs_ENDPOINT,
            api_key=settings.SCW_SECRET_KEY,
            model=MODEL_BASE_LLM,
            callbacks=[setup_lunary()],
        )

        template = """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        Context: {context} 

        {history} 
        Human: {question} 
        AI:
        """
        self.rag_prompt_custom = PromptTemplate.from_template(template)

        self.rag_template = (
            {
                "context": self.retriever,
                "history": self.str_qa_history,
                "question": RunnablePassthrough(),
            }
            | self.rag_prompt_custom
            | self.llm
            | StrOutputParser()
        )

    def object_to_langchain_conv(self, objects, sys_prompt=""):
        langchain_conv = [SystemMessage(content=sys_prompt)] if sys_prompt else []

        langchain_conv += [
            HumanMessage(content=obj.content)
            if obj.role == "user"
            else AIMessage(content=obj.content)
            for obj in objects
        ]
        return langchain_conv

    def str_qa_history(self, query):
        """Convert a list of HumanMessage and AIMessage into a formatted string."""
        return "\n".join(
            [
                f"Human: {msg.content}"
                if isinstance(msg, HumanMessage)
                else f"AI: {msg.content}"
                for msg in self.qa_history
            ]
        )


    def generate_response(self, rag_enable=True):
        
        if rag_enable:
            return self.rag_template.invoke(self.query)

        else:
            return self.llm.invoke(self.conv).content


    async def stream_response(self, rag_enable=True):

        if rag_enable:
            return self.rag_template.stream(self.query)

        else:
            return self.llm.stream(self.conv)