import os
import json
import time
import random
import string
import uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional, List
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.tools import StructuredTool
from langchain_core.documents import Document
from craft_ai_sdk import CraftAiSdk, CREATION_PARAMETER_VALUE
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.prebuilt import ToolNode, tools_condition

# Load environment variables and init craft sdk
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
antropic_api_key = os.getenv("ANTHROPIC_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

stage = "integration"
craft_ai_environment_url = os.getenv(f"CRAFT_AI_ENVIRONMENT_URL_{stage.upper()}")
craft_ai_sdk_token = os.getenv(f"CRAFT_AI_SDK_TOKEN_{stage.upper()}")

sdk = CraftAiSdk(
    environment_url=craft_ai_environment_url,
    sdk_token=craft_ai_sdk_token,
    control_url=f"https://{stage}.craft.ai"
)

stage = "preprod"
craft_ai_environment_url_preprod = os.getenv(f"CRAFT_AI_ENVIRONMENT_URL_{stage.upper()}")
craft_ai_sdk_token_preprod = os.getenv(f"CRAFT_AI_SDK_TOKEN_{stage.upper()}")

sdk_preprod = CraftAiSdk(
    environment_url=craft_ai_environment_url_preprod,
    sdk_token=craft_ai_sdk_token_preprod,
    control_url=f"https://{stage}.craft.ai"
)


# Tool functions
def generate_random_characters(num_chars=10) -> str:
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(num_chars))

def divide(a: int, b: int):
    return a / b

def multi(a: int, b: int):
    return a * b


def square_root(a: int):
    """square_root a"""

    res = sdk.trigger_endpoint(
        "product-square-root-depl",
        os.getenv("ENDPOINT_TOKEN_SQUARE_ROOT"),
        # Endpoint inputs: Commented inputs are optional, uncomment them to use them.
        inputs={
            "a": a,
        },
    )

    return res["outputs"]["b"]


def get_pipeline_info(pipeline_name: str):
    return sdk.get_pipeline(pipeline_name)

def get_user_name_info(user_id: str):
    return sdk.get_user(user_id)["name"]

def python_script_to_file(script: str):
    file_name = generate_random_characters() + ".py"
    with open("steps/" + file_name, "w") as text_file:
        text_file.write(script)
    return file_name

def create_pipeline(pipeline_name, file_path, function_name):
    return sdk.create_pipeline(
        pipeline_name=pipeline_name,
        function_path=file_path,
        function_name=function_name,
        container_config={
            "local_folder": "steps",
            "requirements_path": CREATION_PARAMETER_VALUE.NULL
        }
    )

def run_pipeline(pipeline_name):
    return sdk.run_pipeline(pipeline_name=pipeline_name)

def get_execution_logs(execution_id):
    logs = sdk.get_pipeline_execution_logs(execution_id=execution_id)
    return '\n'.join(log["message"] for log in logs)


def prediction_RTE(region):
    """prediction_RTE region"""

    response =  sdk_preprod.trigger_endpoint(
        "rte-predict-xgb-endpt",
        os.getenv("ENDPOINT_TOKEN_RTE_XGB"),
        inputs={
            "focal_location": region,  
        },
    )

    predictions_byte = response['outputs']['predictions']
    decoded_predictions = predictions_byte.decode('utf-8')

    # predictions_list = json.loads(decoded_predictions)


    return decoded_predictions



# Tool registry
tool_registry = {
    str(uuid.uuid4()): StructuredTool.from_function(divide, name="divide_number", description="Divide two numbers."),
    str(uuid.uuid4()): StructuredTool.from_function(multi, name="multiply_number", description="Multiply two numbers."),
    str(uuid.uuid4()): StructuredTool.from_function(get_pipeline_info, name="get_pipeline_info", description="Get pipeline information."),
    str(uuid.uuid4()): StructuredTool.from_function(get_user_name_info, name="get_user_name_info", description="Get user name information."),
    str(uuid.uuid4()): StructuredTool.from_function(python_script_to_file, name="python_script_to_file", description="Save Python script to file."),
    str(uuid.uuid4()): StructuredTool.from_function(create_pipeline, name="create_pipeline", description="Create a pipeline."),
    str(uuid.uuid4()): StructuredTool.from_function(run_pipeline, name="run_pipeline", description="Run a pipeline."),
    str(uuid.uuid4()): StructuredTool.from_function(get_execution_logs, name="get_execution_logs", description="Get execution logs."),
    str(uuid.uuid4()): StructuredTool.from_function(square_root, name="square_root", description=f"Take 1 int give as parameter and do a square root of it as return",),
    str(uuid.uuid4()): StructuredTool.from_function(prediction_RTE, name="prediction_RTE", description=f"Take region as input and make electrical consumption prediction for the next mounth"),
}

# Vector store for tool selection
tool_documents = [
    Document(page_content=tool.description, id=id, metadata={"tool_name": tool.name})
    for id, tool in tool_registry.items()
]
vector_store = InMemoryVectorStore(embedding=OpenAIEmbeddings())
vector_store.add_documents(tool_documents)

# Creation of the graph 

tools = list(tool_registry.values())
llm = ChatOpenAI()

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

llm_app = workflow.compile()




# FastAPI app setup
app = FastAPI(title="OpenAI-compatible API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models for fast API
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "mock-gpt-model"
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False

# Fonction for generate answer using graph 

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

# Chat completion route
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):

    if request.stream:
        raise ValueError("Streaming isn't available on this API")

    else : 

        if request.messages:
            response_text = query_process(str(request.messages), request.model)
        else:
            response_text = "User message not received"


    return {
        "id": "1337",
        "object": "chat.completion",
        "created": time.time(),
        "model": request.model,
        "choices": [{"message": Message(role="assistant", content=str(response_text))}],
    }
# Run app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
