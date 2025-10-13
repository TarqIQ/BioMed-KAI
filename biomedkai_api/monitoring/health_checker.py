from fastapi import APIRouter
from fastapi.responses import JSONResponse
from http import HTTPStatus
import time
import psutil
import structlog
from typing import Dict

logger = structlog.get_logger()

health_router = APIRouter(tags=["health"])


@health_router.get("/")
async def health_check():
    """Basic health check endpoint"""
    return JSONResponse(
        status_code=HTTPStatus.OK,
        content={
            "status": "healthy",
            "timestamp": time.time(),
            "service": "Medical AI Agent System"
        }
    )


@health_router.get("/detailed")
async def detailed_health_check():
    """Detailed health check with system metrics"""
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Check external dependencies
        dependencies_status = await check_dependencies()
        
        health_data = {
            "status": "healthy" if all(dependencies_status.values()) else "degraded",
            "timestamp": time.time(),
            "system_metrics": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            },
            "dependencies": dependencies_status,
            "service": "Medical AI Agent System"
        }
        
        status_code = HTTPStatus.OK if health_data["status"] == "healthy" else HTTPStatus.SERVICE_UNAVAILABLE
        
        return JSONResponse(
            status_code=status_code,
            content=health_data
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={
                "status": "unhealthy",
                "timestamp": time.time(),
                "error": str(e),
                "service": "Medical AI Agent System"
            }
        )


@health_router.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe endpoint"""
    try:
        # Check if all critical components are ready
        # Import here to avoid circular imports
        try:
            from .main import orchestrator, memory_system, tool_registry
        except ImportError:
            orchestrator = None
            memory_system = None
            tool_registry = None
        
        if not orchestrator or not memory_system or not tool_registry:
            return JSONResponse(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                content={
                    "status": "not ready",
                    "timestamp": time.time(),
                    "reason": "Core components not initialized"
                }
            )
        
        return JSONResponse(
            status_code=HTTPStatus.OK,
            content={
                "status": "ready",
                "timestamp": time.time(),
                "service": "Medical AI Agent System"
            }
        )
        
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        return JSONResponse(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            content={
                "status": "not ready",
                "timestamp": time.time(),
                "error": str(e)
            }
        )


@health_router.get("/live")
async def liveness_check():
    """Kubernetes liveness probe endpoint"""
    return JSONResponse(
        status_code=HTTPStatus.OK,
        content={
            "status": "alive",
            "timestamp": time.time(),
            "service": "Medical AI Agent System"
        }
    )


async def check_dependencies() -> Dict[str, bool]:
    """Check status of external dependencies"""
    dependencies = {
        "redis": False,
        "postgres": False,
        "neo4j": False,
        "model": False
    }
    
    try:
        # Import here to avoid circular imports
        try:
            from .main import orchestrator, memory_system, tool_registry
        except ImportError:
            orchestrator = None
            memory_system = None
            tool_registry = None
        
        # Check memory system (Redis, Postgres, Neo4j)
        if memory_system:
            try:
                # This would need to be implemented in the memory system
                # For now, assume healthy if initialized
                dependencies["redis"] = True
                dependencies["postgres"] = True
                dependencies["neo4j"] = True
            except Exception:
                pass
        
        # Check model availability
        if orchestrator and hasattr(orchestrator, 'model'):
            dependencies["model"] = True
            
    except Exception as e:
        logger.error("Dependency check failed", error=str(e))
    
    return dependencies