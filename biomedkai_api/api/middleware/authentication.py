from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Optional
import jwt
import os
from datetime import datetime, timezone

# Security scheme
security = HTTPBearer()

# Configuration - these should come from environment variables
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """
    Validate JWT token and return current user information
    """
    try:
        # Decode JWT token
        payload = jwt.decode(
            credentials.credentials,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        
        # Extract user information from token
        user_id: str = payload.get("sub")
        username: str = payload.get("username")
        email: str = payload.get("email")
        role: str = payload.get("role", "user")
        expires_at: float = payload.get("exp")
        
        if user_id is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check if token is expired
        if expires_at and datetime.fromtimestamp(expires_at, tz=timezone.utc) < datetime.now(timezone.utc):
            raise HTTPException(
                status_code=401,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Return user information
        return {
            "user_id": user_id,
            "username": username,
            "email": email,
            "role": role,
            "token_expires": expires_at
        }
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


def verify_role(required_role: str):
    """
    Dependency to verify user has required role
    """
    async def role_checker(current_user: Dict = Depends(get_current_user)) -> Dict:
        user_role = current_user.get("role", "user")
        
        # Define role hierarchy
        role_hierarchy = {
            "admin": 3,
            "doctor": 2,
            "nurse": 1,
            "user": 0
        }
        
        required_level = role_hierarchy.get(required_role, 0)
        user_level = role_hierarchy.get(user_role, 0)
        
        if user_level < required_level:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required role: {required_role}"
            )
        
        return current_user
    
    return role_checker


# Optional: Admin-only dependency
async def get_admin_user(current_user: Dict = Depends(verify_role("admin"))) -> Dict:
    """Dependency that requires admin role"""
    return current_user


# Optional: Doctor or higher dependency
async def get_medical_user(current_user: Dict = Depends(verify_role("nurse"))) -> Dict:
    """Dependency that requires medical professional role"""
    return current_user

async def verify_websocket_token(token: str) -> Dict:
    """
    Validate JWT token for WebSocket connections
    """
    try:
        # Decode JWT token
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        
        # Extract user information from token
        user_id: str = payload.get("sub")
        username: str = payload.get("username")
        email: str = payload.get("email")
        role: str = payload.get("role", "user")
        expires_at: float = payload.get("exp")
        
        if user_id is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials"
            )
        
        # Check if token is expired
        if expires_at and datetime.fromtimestamp(expires_at, tz=timezone.utc) < datetime.now(timezone.utc):
            raise HTTPException(
                status_code=401,
                detail="Token has expired"
            )
        
        # Return user information
        return {
            "user_id": user_id,
            "username": username,
            "email": email,
            "role": role,
            "token_expires": expires_at
        }
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials"
        )
    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Authentication failed"
        )
