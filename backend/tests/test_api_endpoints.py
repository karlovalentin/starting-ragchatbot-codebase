"""API endpoint tests for the RAG system FastAPI application."""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from config import Config
from rag_system import RAGSystem
from models import Course, Lesson


@pytest.fixture
def test_app(mock_config, mock_rag_system):
    """Create a test FastAPI app without static file mounting."""
    from fastapi import HTTPException
    from pydantic import BaseModel
    from typing import List, Optional, Dict, Any
    
    # Create test app
    app = FastAPI(title="Test Course Materials RAG System", root_path="")
    
    # Add middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Pydantic models for request/response
    class QueryRequest(BaseModel):
        """Request model for course queries"""
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        """Response model for course queries"""
        answer: str
        sources: List[Dict[str, Optional[str]]]
        session_id: str

    class CourseStats(BaseModel):
        """Response model for course statistics"""
        total_courses: int
        course_titles: List[str]
    
    # API Endpoints
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources"""
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            
            answer, sources = mock_rag_system.query(request.query, session_id)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics"""
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def read_root():
        """Basic root endpoint for testing"""
        return {"message": "RAG System API"}
    
    return app


@pytest.fixture
def test_client(test_app):
    """Create a test client for the FastAPI app."""
    return TestClient(test_app)


@pytest.mark.api
class TestAPIEndpoints:
    """Test FastAPI endpoint functionality."""
    
    def test_root_endpoint(self, test_client):
        """Test the root endpoint returns expected response."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        assert response.json() == {"message": "RAG System API"}
    
    def test_query_endpoint_success(self, test_client, mock_rag_system):
        """Test successful query processing."""
        # Setup mock RAG system response
        mock_rag_system.session_manager.create_session.return_value = "test-session-123"
        mock_rag_system.query.return_value = (
            "This is a test response about testing fundamentals.",
            [
                {"text": "Testing content chunk", "url": "https://example.com/lesson1"},
                {"text": "More testing info", "url": "https://example.com/lesson2"}
            ]
        )
        
        # Make request
        query_data = {
            "query": "What is testing?",
            "session_id": None
        }
        
        response = test_client.post("/api/query", json=query_data)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "This is a test response about testing fundamentals."
        assert data["session_id"] == "test-session-123"
        assert len(data["sources"]) == 2
        assert data["sources"][0]["text"] == "Testing content chunk"
        assert data["sources"][0]["url"] == "https://example.com/lesson1"
        
        # Verify RAG system was called correctly
        mock_rag_system.session_manager.create_session.assert_called_once()
        mock_rag_system.query.assert_called_once_with("What is testing?", "test-session-123")
    
    def test_query_endpoint_with_existing_session(self, test_client, mock_rag_system):
        """Test query with existing session ID."""
        mock_rag_system.query.return_value = (
            "Response with existing session.",
            [{"text": "Content", "url": "https://example.com/lesson"}]
        )
        
        query_data = {
            "query": "Follow up question",
            "session_id": "existing-session-456"
        }
        
        response = test_client.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["session_id"] == "existing-session-456"
        
        # Session should not be created when provided
        mock_rag_system.session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with("Follow up question", "existing-session-456")
    
    def test_query_endpoint_missing_query(self, test_client):
        """Test query endpoint with missing query field."""
        response = test_client.post("/api/query", json={})
        
        assert response.status_code == 422  # Validation error
        error_data = response.json()
        assert "detail" in error_data
    
    def test_query_endpoint_rag_system_error(self, test_client, mock_rag_system):
        """Test query endpoint when RAG system raises an exception."""
        mock_rag_system.session_manager.create_session.return_value = "test-session"
        mock_rag_system.query.side_effect = Exception("RAG system error")
        
        query_data = {
            "query": "Test query",
            "session_id": None
        }
        
        response = test_client.post("/api/query", json=query_data)
        
        assert response.status_code == 500
        error_data = response.json()
        assert error_data["detail"] == "RAG system error"
    
    def test_courses_endpoint_success(self, test_client, mock_rag_system):
        """Test successful course analytics retrieval."""
        # Setup mock analytics response
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": ["Test Course 1", "Test Course 2", "Test Course 3"]
        }
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Test Course 1" in data["course_titles"]
        
        mock_rag_system.get_course_analytics.assert_called_once()
    
    def test_courses_endpoint_empty_analytics(self, test_client, mock_rag_system):
        """Test course analytics with no courses."""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 0
        assert data["course_titles"] == []
    
    def test_courses_endpoint_rag_system_error(self, test_client, mock_rag_system):
        """Test course analytics endpoint when RAG system raises an exception."""
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 500
        error_data = response.json()
        assert error_data["detail"] == "Analytics error"


@pytest.mark.api
class TestRequestValidation:
    """Test API request validation and error handling."""
    
    def test_query_invalid_json(self, test_client):
        """Test query endpoint with invalid JSON."""
        response = test_client.post(
            "/api/query",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_query_empty_string(self, test_client, mock_rag_system):
        """Test query endpoint with empty query string."""
        mock_rag_system.session_manager.create_session.return_value = "test-session"
        mock_rag_system.query.return_value = ("Empty query response", [])
        
        query_data = {"query": ""}
        response = test_client.post("/api/query", json=query_data)
        
        # Should succeed even with empty string
        assert response.status_code == 200
        
        mock_rag_system.query.assert_called_once_with("", "test-session")
    
    def test_query_very_long_string(self, test_client, mock_rag_system):
        """Test query endpoint with very long query string."""
        mock_rag_system.session_manager.create_session.return_value = "test-session"
        mock_rag_system.query.return_value = ("Long query response", [])
        
        long_query = "test " * 1000  # 5000 character query
        query_data = {"query": long_query}
        
        response = test_client.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        mock_rag_system.query.assert_called_once_with(long_query, "test-session")


@pytest.mark.api
@pytest.mark.integration
class TestEndToEndAPIFlow:
    """Test complete API workflows."""
    
    def test_complete_query_flow(self, test_client, mock_rag_system):
        """Test a complete query workflow with session management."""
        # First query - creates new session
        mock_rag_system.session_manager.create_session.return_value = "session-123"
        mock_rag_system.query.return_value = (
            "First response",
            [{"text": "Source 1", "url": "https://example.com/1"}]
        )
        
        response1 = test_client.post("/api/query", json={
            "query": "First question"
        })
        
        assert response1.status_code == 200
        data1 = response1.json()
        session_id = data1["session_id"]
        
        # Follow-up query with same session
        mock_rag_system.query.return_value = (
            "Follow-up response",
            [{"text": "Source 2", "url": "https://example.com/2"}]
        )
        
        response2 = test_client.post("/api/query", json={
            "query": "Follow-up question",
            "session_id": session_id
        })
        
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["session_id"] == session_id
        
        # Verify calls
        assert mock_rag_system.query.call_count == 2
        mock_rag_system.session_manager.create_session.assert_called_once()
    
    def test_analytics_then_query(self, test_client, mock_rag_system):
        """Test getting course analytics before making a query."""
        # Get analytics first
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["Course A", "Course B"]
        }
        
        analytics_response = test_client.get("/api/courses")
        assert analytics_response.status_code == 200
        
        # Then make a query
        mock_rag_system.session_manager.create_session.return_value = "session-456"
        mock_rag_system.query.return_value = (
            "Query response",
            [{"text": "Content", "url": "https://example.com/lesson"}]
        )
        
        query_response = test_client.post("/api/query", json={
            "query": "Test query"
        })
        
        assert query_response.status_code == 200
        
        # Both endpoints should work independently
        mock_rag_system.get_course_analytics.assert_called_once()
        mock_rag_system.query.assert_called_once()