"""
System API Routes

Endpoints for system status, metrics, and architecture information.
These allow the world to observe Atlas's internal state.
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Dict, Any, Optional

router = APIRouter()


class SystemStatus(BaseModel):
    """System status response model."""
    initialized: bool
    atlas_available: bool
    learning_enabled: bool
    stats: Dict[str, Any]
    timestamp: str
    system_state: Optional[Dict[str, Any]] = None
    architecture: Optional[Dict[str, Any]] = None


class MetricsResponse(BaseModel):
    """Metrics response model."""
    frames_processed: int
    audio_chunks_processed: int
    uptime_seconds: float
    timestamp: str
    prediction_error: Optional[float] = None
    reconstruction_error: Optional[float] = None
    cross_modal_correlation: Optional[float] = None
    total_neurons: Optional[int] = None
    active_associations: Optional[int] = None


@router.get("/status", response_model=SystemStatus)
async def get_system_status(request: Request):
    """
    Get current system status.

    Returns the overall state of Atlas including:
    - Initialization status
    - Learning enabled/disabled
    - Processing statistics
    - Current system state
    - Architecture overview
    """
    atlas_manager = request.app.state.atlas_manager
    return await atlas_manager.get_status()


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(request: Request):
    """
    Get learning and performance metrics.

    Returns metrics including:
    - Frames and audio chunks processed
    - Prediction and reconstruction errors
    - Cross-modal correlation
    - Neuron counts and associations
    """
    atlas_manager = request.app.state.atlas_manager
    return await atlas_manager.get_metrics()


@router.get("/architecture")
async def get_architecture(request: Request) -> Dict[str, Any]:
    """
    Get detailed architecture information.

    Returns the neural network architecture including:
    - Visual pathway layer details
    - Audio pathway layer details
    - Multimodal integration layer
    - Cognitive system status
    """
    atlas_manager = request.app.state.atlas_manager
    return await atlas_manager.get_architecture_info()


@router.get("/checkpoints")
async def list_checkpoints(request: Request) -> Dict[str, Any]:
    """
    List available checkpoints.

    Returns all saved checkpoints with:
    - Checkpoint name
    - File path
    - Size in bytes
    - Last modified timestamp
    """
    atlas_manager = request.app.state.atlas_manager
    return await atlas_manager.list_checkpoints()
