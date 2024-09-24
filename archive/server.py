import os
from dotenv import load_dotenv
from starlette.responses import StreamingResponse
from fastapi import FastAPI, HTTPException, Request
from langchain.document_loaders.pdf import PyPDFDirectoryLoader # Importing PDF loader from Langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter # Importing text splitter from Langchain
from langchain.embeddings import OpenAIEmbeddings # Importing OpenAI embeddings from Langchain
from langchain.schema import Document # Importing Document schema from Langchain
from langchain.vectorstores.chroma import Chroma # Importing Chroma vector store from Langchain
from dotenv import load_dotenv # Importing dotenv to get API key from .env file
from langchain_openai import ChatOpenAI
import os # Importing os module for operating system functionalities
import shutil # Importing shutil module for high-level file operations
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate

from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio
import json
import time

from typing import Optional, List

from pydantic import BaseModel, Field

load_dotenv()  # This loads the environment variables from the .env file

# Now you can access the variables like this:
openai_api_key = os.getenv("OPENAI_API_KEY")
antropic_api_key = os.getenv("ANTHROPIC_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("ANTHROPIC_API_KEY")

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from craft_ai_sdk.io import Input, Output, InputSource, OutputDestination
from craft_ai_sdk import CraftAiSdk, INPUT_OUTPUT_TYPES, CREATION_PARAMETER_VALUE
from dotenv import load_dotenv
import craft_ai_sdk, os, json, time, requests

print (craft_ai_sdk.__version__)

load_dotenv()
stage = "integration" # integration , dave, kenny, preprod, prod

# Access the environment variables

craft_ai_environment_url = os.getenv("CRAFT_AI_ENVIRONMENT_URL_"+stage.upper())
craft_ai_sdk_token = os.getenv("CRAFT_AI_SDK_TOKEN_"+stage.upper())


if stage != "prod" :
    sdk = CraftAiSdk(environment_url=craft_ai_environment_url, sdk_token=craft_ai_sdk_token,control_url="https://"+stage+".craft.ai")
else : 
    sdk = CraftAiSdk(environment_url=craft_ai_environment_url, sdk_token=craft_ai_sdk_token)

import re
import uuid
from langchain_core.tools import StructuredTool

import random
import string

def generate_random_characters(num_chars=10) -> str:
    characters = string.ascii_letters + string.digits  # Letters (uppercase and lowercase) + digits
    return ''.join(random.choice(characters) for _ in range(num_chars))

def divide(a: int, b :int):
    """Divide a and b"""
    return a/b


def multi(a: int, b :int):
    """Multiple a and b"""
    return a*b


def get_pipeline_info(pipeline_name:str):
    
    return sdk.get_pipeline(pipeline_name)


def get_user_name_info(user_id:str):
    return sdk.get_user(user_id)["name"]



def python_script_to_file(script:str):
    file_name = generate_random_characters()+".py"
    with open("steps/"+file_name, "w") as text_file:
        text_file.write(script)

    return file_name


""" def python_script_to_requirement(script:str):
    file_name = "requirements.py"
    with open(file_name, "w") as text_file:
        text_file.write(script)

    return file_name 
    """

def create_pipeline(pipeline_name, file_path, function_name):

    return sdk.create_pipeline(
        pipeline_name=pipeline_name,
        function_path=file_path,
        function_name=function_name,   
        container_config={
            "local_folder": "steps", 
            "requirements_path": CREATION_PARAMETER_VALUE.NULL
            
        },

    )
def run_pipeline(pipeline_name):


    return sdk.run_pipeline(pipeline_name=pipeline_name)

def get_execution_logs (execution_id):

    logs = sdk.get_pipeline_execution_logs(
        execution_id=execution_id
    )

    return '\n'.join(log["message"] for log in logs)

tool_registry = {
    str(uuid.uuid4()): StructuredTool.from_function(divide,
        name="devide_number",
        description=f"Take 2 int give as parameter and divide number1/number2 as return",
    ),
    str(uuid.uuid4()): StructuredTool.from_function(multi,
        name="multiple_number",
        description=f"Take 2 int give as parameter and multiple number1*number2 as return",
    ),
    str(uuid.uuid4()): StructuredTool.from_function(get_pipeline_info,
        name="get_pipeline_info",
        description=f"Take pipeline name as input to give in return object with all pipeline info",
    ),
    str(uuid.uuid4()): StructuredTool.from_function(get_user_name_info,
        name="get_user_name_info",
        description=f"Take user as input to give in return user name",
    ),
    str(uuid.uuid4()): StructuredTool.from_function(python_script_to_file,
        name="python_script_to_file",
        description=f"Take script python as input to put it in file to prepare pipeline creation after",
    ),
    str(uuid.uuid4()): StructuredTool.from_function(create_pipeline,
        name="create_pipeline",
        description=f"Take pipeline_name, file_path, function_name as input and create pipeline with it",
    ),
    str(uuid.uuid4()): StructuredTool.from_function(run_pipeline,
        name="run_pipeline",
        description=f"Take pipeline_name as input and run the given pipeline",
    ),
    str(uuid.uuid4()): StructuredTool.from_function(get_execution_logs,
        name="get_execution_logs",
        description=f"Take execution_id as input and give back execution logs",
    ),
}


from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_openai import OpenAIEmbeddings

tool_documents = [
    Document(
        page_content=tool.description,
        id=id,
        metadata={"tool_name": tool.name},
    )
    for id, tool in tool_registry.items()
]

vector_store = InMemoryVectorStore(embedding=OpenAIEmbeddings())
document_ids = vector_store.add_documents(tool_documents)


from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode


# llm = ChatAnthropic(
#     model="claude-3-haiku-20240307", temperature=0
# ).bind_tools(tools)

tools = list(tool_registry.values())
llm = ChatOpenAI()


from typing import Literal
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import StateGraph, MessagesState
from typing import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langgraph.prebuilt import ToolNode, tools_condition


def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "__end__"

class State(TypedDict):
    messages: Annotated[list, add_messages]
    selected_tools: list[str]


def agent(state: State):
    selected_tools = [tool_registry[id] for id in state["selected_tools"]]
    llm_with_tools = llm.bind_tools(selected_tools)
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def select_tools(state: State):
    last_user_message = state["messages"][-1]
    query = last_user_message.content
    tool_documents = vector_store.similarity_search(query)
    return {"selected_tools": [document.id for document in tool_documents]}


workflow = StateGraph(MessagesState)
# Define the two nodes we will cycle between
workflow.add_node("agent", agent)
tool_node = ToolNode(tools=tools)
workflow.add_node("tools", tool_node)

workflow.add_conditional_edges(
    "agent",
    tools_condition,
)
workflow.add_node("select_tools", select_tools)

workflow.add_edge("__start__", "select_tools")
workflow.add_edge("select_tools", "agent")
workflow.add_edge("tools", "agent")
# workflow.add_edge("agent", "tools")

llm_app = workflow.compile()





app = FastAPI(title="OpenAI-compatible API")



# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, replace "*" with specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)




# data models
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "mock-gpt-model"
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False



async def _resp_async_vanilla(text_resp: str, model_name):
    # let's pretend every word is a token and return it over time

    model = ChatOpenAI(model=model_name)
    i=0
    
    for token in query_process(text_resp, model_name, True):
        print(token.content)
        chunk = {
            "id": i,
            "object": "chat.completion.chunk",
            "created": time.time(),
            "model": model_name,
            "choices": [{"delta": {"content": str(token.content) + " "}}],
        }
        i+=1
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"



def query_process(query_text, model_name, stream=False):
  
  last_text_return = ""

  if stream : 
     return ""
  else: 
    for event in llm_app.stream({"messages": [("user", query_text)]}):
        for value in event.values():
            print(value)
            if 'messages' in value :
                last_text_return = value["messages"][-1].content


  return last_text_return
 




@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):

    print(request, request.stream)

    if request.stream:
        return StreamingResponse(
            _resp_async_vanilla(str(request.messages), str(request.model)), media_type="application/x-ndjson"
        )

    else : 

        if request.messages:
            response_text = (
                "As a mock AI Assitant, I can only echo your last message: "
                + request.messages[-1].content
            )
    

            # Let's call our function we have defined
            response_text = query_process(str(request.messages), request.model)

        else:
            response_text = "As a mock AI Assitant, I can only echo your last message, but there wasn't one!"


    return {
        "id": "1337",
        "object": "chat.completion",
        "created": time.time(),
        "model": request.model,
        "choices": [{"message": Message(role="assistant", content=str(response_text))}],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)

