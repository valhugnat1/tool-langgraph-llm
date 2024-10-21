import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models.chat import Message, ChatCompletionRequest

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_chat_completion():
    request_data = ChatCompletionRequest(
        messages=[
            Message(role="user", content="Test message")
        ]
    )
    
    response = client.post("/chat/completions", json=request_data.dict())
    assert response.status_code == 200
    assert "choices" in response.json()

def test_chat_completion_streaming():
    request_data = ChatCompletionRequest(
        messages=[
            Message(role="user", content="Test message")
        ],
        stream=True
    )
    
    response = client.post("/chat/completions", json=request_data.dict())
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"