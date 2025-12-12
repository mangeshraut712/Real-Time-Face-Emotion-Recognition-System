"""Integration tests for Flask API."""

import pytest

from src.web.app import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestAPI:
    """Test suite for Flask API endpoints."""

    def test_index_route(self, client):
        """Test index route returns HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert b"<!doctype html>" in response.data.lower()

    def test_status_endpoint(self, client):
        """Test status endpoint."""
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.get_json()
        assert "running" in data
        assert isinstance(data["running"], bool)

    def test_emotions_endpoint_no_stream(self, client):
        """Test emotions endpoint without active stream."""
        response = client.get("/api/emotions")
        assert response.status_code == 200
        data = response.get_json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_start_stream_endpoint(self, client):
        """Test start stream endpoint."""
        response = client.post(
            "/api/start", json={"camera_index": 0}, content_type="application/json"
        )
        # May fail if no camera, but should return valid response
        assert response.status_code in [200, 500]
        data = response.get_json()
        assert "status" in data or "error" in data

    def test_stop_stream_endpoint(self, client):
        """Test stop stream endpoint."""
        response = client.post("/api/stop")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "stopped"

    def test_404_fallback_to_index(self, client):
        """Test 404 routes fallback to index for SPA."""
        response = client.get("/some/random/path")
        assert response.status_code == 200
        assert b"<!doctype html>" in response.data.lower()
