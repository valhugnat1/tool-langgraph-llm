from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from ..core.config import get_settings
from ..core.logging import setup_lunary
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate
from app.services.vector_store import VectorStoreDB

# Fetches application settings and sets the LLM model to use.
settings = get_settings()
MODEL_BASE_LLM = "llama-3.1-8b-instruct"


class MODELService:
    """Service class responsible for handling interaction with a question-answering model."""

    def __init__(self, messages):
        # Set up the retriever from the vector store and extract the current query from the request.
        self.retriever = VectorStoreDB().get_retriever()
        self.query = str(messages[-1].content)

        # Converts the incoming request into a format suitable for LangChain.
        self.conv = self.object_to_langchain_conv(messages)
        self.qa_history = self.conv[
            :-1
        ]  # Exclude the last message as it is the current query.

        # Initializes the ChatOpenAI model with the necessary configuration.
        self.llm = ChatOpenAI(
            base_url=settings.SCW_GENERATIVE_APIs_ENDPOINT,
            api_key=settings.SCW_SECRET_KEY,
            model=MODEL_BASE_LLM,
            callbacks=[setup_lunary()],
        )

        # Custom prompt template for the retrieval-augmented generation (RAG) process.
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

        # Defines a pipeline that combines the retriever, history, query, prompt template and LLM to generate responses.
        self.rag_template = (
            {
                "context": RunnablePassthrough.assign(
                    context=(lambda x: self.format_docs(x["context"]))
                ),
                "history": self.str_qa_history,
                "question": RunnablePassthrough(),
            }
            | self.rag_prompt_custom
            | self.llm
            | StrOutputParser()
        )

        self.rag_runnable_template = RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough()}
        ).assign(answer=self.rag_template)

    def format_docs(slef, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def object_to_langchain_conv(self, objects, sys_prompt=""):
        """Converts a list of messages (objects) to a LangChain-compatible conversation format."""
        langchain_conv = [SystemMessage(content=sys_prompt)] if sys_prompt else []

        langchain_conv += [
            HumanMessage(content=obj.content)
            if obj.role == "user"
            else AIMessage(content=obj.content)
            for obj in objects
        ]
        return langchain_conv

    def str_qa_history(self, query):
        """Convert a list of HumanMessage and AIMessage into a formatted string for use in the prompt."""
        return "\n".join(
            [
                f"Human: {msg.content}"
                if isinstance(msg, HumanMessage)
                else f"AI: {msg.content}"
                for msg in self.qa_history
            ]
        )

    def generate_response(self, rag_enable=False):
        """Generates a response (no stream) based on the user's query, either with or without RAG."""
        if rag_enable:
            return self.rag_runnable_template.invoke(self.query)
        else:
            return self.llm.invoke(self.conv).content

    async def stream_response(self, rag_enable=False):
        """Generates a streaming response based on the user's query, either with or without RAG."""
        if rag_enable:
            return self.rag_runnable_template.stream(self.query)
        else:
            return self.llm.stream(self.conv)
