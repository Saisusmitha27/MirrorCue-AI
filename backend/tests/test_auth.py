import pytest
import uuid
from datetime import datetime
from unittest.mock import MagicMock
from backend.models.user import User

def test_register_success(client, mock_db):
    # Mock database to return None (meaning email is free)
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_result
    
    # We mock DB refresh by assigning fields to the object
    async def mock_refresh(instance):
        instance.id = uuid.uuid4()
        instance.created_at = datetime.utcnow()
        
    mock_db.refresh.side_effect = mock_refresh
    
    payload = {
        "email": "register_test@example.com",
        "full_name": "New Tester",
        "password": "mypassword123"
    }
    
    response = client.post("/auth/register", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == payload["email"]
    assert data["full_name"] == payload["full_name"]
    assert "id" in data
    assert data["is_active"] is True

def test_register_conflict(client, mock_db):
    # Mock database to return an existing user model (email conflict)
    existing_user = User(
        id=uuid.uuid4(),
        email="conflict_test@example.com",
        full_name="Existing User",
        hashed_password="somehashvalue",
        is_active=True,
        created_at=datetime.utcnow()
    )
    
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = existing_user
    mock_db.execute.return_value = mock_result
    
    payload = {
        "email": "conflict_test@example.com",
        "full_name": "Existing User Clone",
        "password": "differentpwd"
    }
    
    response = client.post("/auth/register", json=payload)
    assert response.status_code == 409
    assert response.json()["detail"] == "Email already exists"
