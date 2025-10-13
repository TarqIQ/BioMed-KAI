import asyncio
import pathlib
import uvicorn
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import structlog
from typing import Optional, Dict, Any

from .config.settings import settings, agent_config, load_agent_config
from .api.websocket_server import websocket_router
from .api.rest_api import api_router
from .core.orchestrator import MedicalAgentOrchestrator
from .models.llama_model import LlamaModelWrapper
from .memory.hybrid_memory import HybridMemorySystem
from .tools import create_tool_registry
from .monitoring.metrics_collector import setup_metrics
from .monitoring.health_checker import health_router
import os


def create_app() -> FastAPI:
    # ↓ move all heavy imports *inside* this function
    import structlog
    from contextlib import asynccontextmanager
    from .config.settings import settings, agent_config, load_agent_config
    from .api.websocket_server import websocket_router
    from .api.rest_api import api_router
    from .monitoring.metrics_collector import setup_metrics
    from .monitoring.health_checker import health_router
    from .core.orchestrator import MedicalAgentOrchestrator
    from .models.llama_model import LlamaModelWrapper
    from .memory.hybrid_memory import HybridMemorySystem
    from .tools import create_tool_registry

    import asyncio, os, pathlib

    # Configure structured logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logger = structlog.get_logger()

    async def _with_timeout(coro, seconds, what: str):
        try:
            return await asyncio.wait_for(coro, timeout=seconds)
        except asyncio.TimeoutError:
            logger.warning("Startup step timed out", step=what, timeout_s=seconds)
            return None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan manager"""
        global orchestrator, memory_system, tool_registry

        logger.info("Starting Medical AI Agent System", app_name=settings.app_name, environment=settings.app_env)

        # 0) toggles for local bring-up
        SKIP_MODEL  = os.getenv("BIOMEDKAI_SKIP_MODEL",  "0") == "1"
        SKIP_MEMORY = os.getenv("BIOMEDKAI_SKIP_MEMORY", "0") == "1"

        try:
            # 1) Model
            model = None
            if not SKIP_MODEL:
                model_path = pathlib.Path(settings.model_path)
                if not model_path.exists():
                    logger.error("Model path does not exist", model_path=str(model_path))
                    raise FileNotFoundError(model_path)
                logger.info("Loading LLM model", model_path=str(model_path))
                mw = LlamaModelWrapper(str(model_path))
                params = {
                    "context_size": settings.context_size,
                    "gpu_layers": settings.gpu_layers,
                    "threads": settings.threads,
                    "batch_size": settings.batch_size
                }
                model = await _with_timeout(mw.initialize_with_params(**params), 30, "llm.initialize_with_params")   # ← timeout
            else:
                logger.warning("Skipping model initialization (BIOMEDKAI_SKIP_MODEL=1)")
                mw = None

            # 2) Memory
            memory_system = None
            if not SKIP_MEMORY:
                logger.info("Initializing memory system")
                ms = HybridMemorySystem(
                    redis_url=f"redis://{settings.redis_host}:{settings.redis_port}",
                    postgres_url=f"postgresql://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}",
                    neo4j_url=settings.neo4j_uri,
                    neo4j_auth=(settings.neo4j_user, settings.neo4j_password),
                )
                memory_system = await _with_timeout(ms.initialize(), 10, "memory.initialize")  # ← timeout
            else:
                logger.warning("Skipping memory initialization (BIOMEDKAI_SKIP_MEMORY=1)")

            # 3) Tools
            logger.info("Creating tool registry")
            tool_registry = create_tool_registry()
            if model and tool_registry.get("knowledge_graph_search"):
                model.set_context_retreiver(tool_registry["knowledge_graph_search"])

            # 4) Orchestrator
            logger.info("Initializing agent orchestrator")
            orchestrator = MedicalAgentOrchestrator(
                model=model,
                tools=tool_registry,
                memory_system=memory_system,
                config=agent_config,
            )

            logger.info("Medical AI Agent System started successfully")
            yield

        except Exception as e:
            logger.error("Failed to start Medical AI Agent System", error=str(e))
            raise
        finally:
            logger.info("Shutting down Medical AI Agent System")
            if memory_system and hasattr(memory_system, "close"):
                try:
                    await memory_system.close()
                except Exception as e:
                    logger.warning("Error closing memory system", error=str(e))
            if 'mw' in locals() and mw and hasattr(mw, 'cleanup'):
                try:
                    await mw.cleanup()
                except Exception as e:
                    logger.warning("Error cleaning model wrapper", error=str(e))
            logger.info("Shutdown complete")

    app = FastAPI(title="Medical AI Agent System", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Setup monitoring
    setup_metrics(app)

    # Include routers
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(websocket_router, prefix="/ws")
    app.include_router(health_router, prefix="/health")


    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "name": "Medical AI Agent System",
            "version": "1.0.0",
            "status": "operational",
            "endpoints": {
                "api": "/api/v1",
                "websocket": "/ws",
                "health": "/health",
                "docs": "/docs"
            }
        }
    
    return app




# Global instances
orchestrator: Optional[MedicalAgentOrchestrator] = None
memory_system: Optional[HybridMemorySystem] = None
tool_registry: Optional[Dict[str, Any]] = None
model_wrapper: Optional[LlamaModelWrapper] = None







class EnhancedMediaServer:
    """
    Enhanced version of your MediaServer that integrates with the agent system
    """
    
    def __init__(self):
        # Your existing initialization
        self.model_name = "llama-3.1"
        self.connected_clients = set()
        
        # Initialize the agent system
        self._initialize_agent_system()
        
    def _initialize_agent_system(self):
        """Initialize the agent orchestrator"""
        # Initialize model wrapper
        self.model_wrapper = LlamaModelWrapper()
        asyncio.create_task(self.model_wrapper.initialize())
        
        # Initialize memory system
        self.memory_system = HybridMemorySystem(
            redis_url="redis://localhost:6379",
            postgres_url="postgresql://medical_ai:password@localhost:5432/medical_memory",
            neo4j_url="neo4j://localhost:7687",
            neo4j_auth=("neo4j", "password")
        )
        asyncio.create_task(self.memory_system.initialize())
        
        # Create tools
        self.tool_registry = create_tool_registry()
        
        # Load agent config
        agent_config = load_agent_config()
        
        self.model_wrapper.set_context_retreiver(self.tool_registry.get("knowledge_graph_search"))
        # Create orchestrator
        self.orchestrator = MedicalAgentOrchestrator(
            model=self.model_wrapper,
            tools=self.tool_registry,
            memory_system=self.memory_system,
            config=agent_config
        )
        
    async def handle_generate_response(self, data: Dict[str, Any], websocket):
        """Enhanced response generation using agent system"""
        prompt = data.get("prompt", "")
        chat_history = data.get("chat_history", [])
        use_agents = data.get("use_agents", True)
        
        await websocket.send(json.dumps({
            "type": "GENERATE_RESPONSE",
            "stream_start": True
        }))
        
        if use_agents:
            # Use the agent system
            async for chunk in self.orchestrator.process_query(prompt):
                await websocket.send(json.dumps({
                    "type": "GENERATE_RESPONSE",
                    "chunk": chunk,
                    "agent_mode": True
                }))
                await asyncio.sleep(0)
        else:
            # Use direct model (your existing code)
            async for chunk in self.model_wrapper.model.generate2(prompt, chat_history):
                await websocket.send(json.dumps({
                    "type": "GENERATE_RESPONSE",
                    "chunk": chunk
                }))
                await asyncio.sleep(0)
                
        await websocket.send(json.dumps({
            "type": "GENERATE_RESPONSE",
            "stream_end": True
        }))

# Create FastAPI app
# app = FastAPI(
#     title="Medical AI Agent System",
#     description="Advanced medical AI assistant with multi-agent architecture",
#     version="1.0.0",
#     lifespan=lifespan
# )

# Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Configure appropriately for production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Setup monitoring
# setup_metrics(app)

# # Include routers
# app.include_router(api_router, prefix="/api/v1")
# app.include_router(websocket_router, prefix="/ws")
# app.include_router(health_router, prefix="/health")


# @app.get("/")
# async def root():
#     """Root endpoint"""
#     return {
#         "name": "Medical AI Agent System",
#         "version": "1.0.0",
#         "status": "operational",
#         "endpoints": {
#             "api": "/api/v1",
#             "websocket": "/ws",
#             "health": "/health",
#             "docs": "/docs"
#         }
#     }

# app = create_app()

def run():
    """Main entry point"""
    uvicorn.run(
        "biomedkai_api.main:create_app",
        host="0.0.0.0",
        port=settings.app_port,
        reload=settings.app_debug,
        log_level=settings.log_level.lower(),
        factory=True,             
    )


# if __name__ == "__main__":
#     main()
