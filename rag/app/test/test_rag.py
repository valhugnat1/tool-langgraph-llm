import pytest
from unittest.mock import Mock, patch
from app.services.rag import RAGService
from app.services.vector_store import get_vector_store

@pytest.fixture
def mock_vector_store():
    return Mock()

@pytest.fixture
def rag_service(mock_vector_store):
    return RAGService(mock_vector_store)

@pytest.mark.asyncio
async def test_generate_response(rag_service):
    test_query = "Test question"
    expected_response = "Test response"
    
    # Mock the chain invocation
    rag_service.get_rag_chain = Mock()
    rag_service.get_rag_chain.return_value.invoke.return_value = expected_response
    
    response = await rag_service.generate_response(test_query)
    assert response == expected_response

@pytest.mark.asyncio
async def test_generate_response_streaming(rag_service):
    test_query = "Test question"
    expected_chunks = ["chunk1", "chunk2", "chunk3"]
    
    # Mock the chain streaming
    rag_service.get_rag_chain = Mock()
    rag_service.get_rag_chain.return_value.stream.return_value = expected_chunks
    
    response_stream = rag_service.generate_response(test_query, stream=True)
    chunks = [chunk async for chunk in response_stream]
    assert chunks == expected_chunks

@pytest.mark.integrationtest
def test_vector_store_connection():
    """Integration test for vector store connection"""
    try:
        vector_store = get_vector_store()
        # Try a simple similarity search
        results = vector_store.similarity_search("test", k=1)
        assert isinstance(results, list)
    except Exception as e:
        pytest.fail(f"Vector store connection failed: {str(e)}")