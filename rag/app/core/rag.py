from langchain_community.chat_models import ChatOpenAI
from ..core.logging import setup_logging
from langchain import hub
import os
import time
import json
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.core.config import get_settings
from app.services.vector_store import get_vector_store


settings = get_settings()

CHROMA_PATH = "chroma"  # old vector DB
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
"""



def context_rag_fct(query_text):
    embedding_function = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=10)
    if len(results) == 0 or results[0][1] < 0.7:
        print("Unable to find matching results.")

    context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    return prompt, results



def query_rag(query_text):
    prompt, results = context_rag_fct(query_text)

    model = ChatOpenAI(
        base_url=settings.SCW_GENERATIVE_APIs_ENDPOINT,
        model_name="llama-3.1-8b-instruct",
        api_key=settings.SCW_SECRET_KEY,
        callbacks=[setup_logging()],
    )

    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response, response_text



async def _resp_async_generator(request: str):
    global vector_store

    llm = ChatOpenAI(
        base_url=settings.SCW_GENERATIVE_APIs_ENDPOINT,
        api_key=settings.SCW_SECRET_KEY,
        model="llama-3.1-8b-instruct",
    )

    prompt = hub.pull("rlm/rag-prompt")
    retriever = get_vector_store().as_retriever()

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    i = 0
    for response_chunk in rag_chain.stream(str(request.messages[-1])):

        chunk = {
            "id": i,
            "object": "chat.completion.chunk",
            "created": time.time(),
            "model": request.model, 
            "choices": [{"delta": {"content": response_chunk}}],
        }
        i += 1
        yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"

