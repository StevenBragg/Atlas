"""
Atlas Web Backend - FastAPI Microservice

This service provides a REST API and WebSocket interface for the Atlas
self-organizing audio-visual learning system, enabling bidirectional
communication between Atlas and the world.
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Add Atlas to path
atlas_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(atlas_root))

from api.routes import system, data, memory, control, websocket
from services.atlas_manager import AtlasManager

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=os.getenv("LOG_LEVEL", "INFO")
)

# Global Atlas manager instance
atlas_manager: AtlasManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - initialize and cleanup Atlas."""
    global atlas_manager

    logger.info("Starting Atlas Web Backend...")

    # Initialize Atlas manager
    atlas_manager = AtlasManager()
    await atlas_manager.initialize()

    # Store in app state for access in routes
    app.state.atlas_manager = atlas_manager

    logger.info("Atlas Web Backend ready!")

    yield

    # Cleanup
    logger.info("Shutting down Atlas Web Backend...")
    await atlas_manager.shutdown()
    logger.info("Atlas Web Backend shutdown complete.")


# Create FastAPI application
app = FastAPI(
    title="Atlas Web API",
    description="""
    ## Atlas - Autonomously Teaching, Learning And Self-organizing

    This API provides bidirectional communication between Atlas and the world:

    ### World → Atlas (Input)
    - **Visual Data**: Send images and video frames for processing
    - **Audio Data**: Send audio samples for processing
    - **Commands**: Control learning, configuration, and system state

    ### Atlas → World (Output)
    - **Predictions**: Get temporal and cross-modal predictions
    - **Memory Access**: Browse episodic and semantic memories
    - **Metrics**: Monitor learning progress and system health
    - **Real-time Updates**: WebSocket stream of system state
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(system.router, prefix="/api/system", tags=["System"])
app.include_router(data.router, prefix="/api/data", tags=["Data Input"])
app.include_router(memory.router, prefix="/api/memory", tags=["Memory"])
app.include_router(control.router, prefix="/api/control", tags=["Control"])
app.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information."""
    return {
        "name": "Atlas Web API",
        "version": "1.0.0",
        "description": "Bidirectional interface between Atlas and the world",
        "endpoints": {
            "system": "/api/system - System status and metrics",
            "data": "/api/data - Send data to Atlas",
            "memory": "/api/memory - Access Atlas memories",
            "control": "/api/control - Control Atlas behavior",
            "websocket": "/ws/stream - Real-time updates"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "atlas-web-backend"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("ATLAS_WEB_PORT", "8000"))
    host = os.getenv("ATLAS_WEB_HOST", "0.0.0.0")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
