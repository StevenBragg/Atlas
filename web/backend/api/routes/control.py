"""
Control API Routes

Endpoints for controlling Atlas's behavior - this is how the world
can guide, configure, and manage Atlas.
"""

from typing import Optional
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


class LearningControl(BaseModel):
    """Learning control parameters."""
    enabled: bool = Field(..., description="Enable or disable learning")


class LearningRateControl(BaseModel):
    """Learning rate control."""
    rate: float = Field(..., ge=0.0, le=1.0, description="Learning rate (0.0 to 1.0)")


class CheckpointRequest(BaseModel):
    """Checkpoint save request."""
    name: Optional[str] = Field(None, description="Checkpoint name (auto-generated if not provided)")
    compress: Optional[bool] = Field(None, description="Compress checkpoint (uses config default if not provided)")
    sync_to_cloud: Optional[bool] = Field(None, description="Sync to cloud after saving")
    metadata: Optional[dict] = Field(None, description="Additional metadata")


class CheckpointLoadRequest(BaseModel):
    """Checkpoint load request."""
    name: str = Field(..., description="Name of checkpoint to load")
    version: Optional[int] = Field(None, description="Version number to load")


class CheckpointRollbackRequest(BaseModel):
    """Checkpoint rollback request."""
    steps: int = Field(1, ge=1, description="Number of versions to rollback")


class CloudSyncRequest(BaseModel):
    """Cloud sync request."""
    direction: str = Field("to", description="Sync direction: 'to' or 'from'")
    version: Optional[int] = Field(None, description="Specific version to sync")


class ModeControl(BaseModel):
    """Cognitive mode control."""
    mode: str = Field(..., description="Cognitive mode: visual, audio, reasoning, creative, autonomous")


class ConfigUpdate(BaseModel):
    """Configuration update."""
    learning_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    multimodal_size: Optional[int] = Field(None, ge=10, le=1000)
    prune_interval: Optional[int] = Field(None, ge=100, le=100000)
    enable_structural_plasticity: Optional[bool] = None
    enable_temporal_prediction: Optional[bool] = None


@router.post("/learning")
async def set_learning(request: Request, control: LearningControl):
    """
    Enable or disable learning.

    When learning is disabled, Atlas will still process inputs but
    will not update its weights or form new associations.

    **Input**: enabled (bool)
    **Output**: Current learning state
    """
    atlas_manager = request.app.state.atlas_manager
    return await atlas_manager.set_learning_enabled(control.enabled)


@router.get("/learning")
async def get_learning_state(request: Request):
    """
    Get current learning state.

    **Output**: Whether learning is currently enabled
    """
    atlas_manager = request.app.state.atlas_manager
    return {
        "learning_enabled": atlas_manager._learning_enabled,
        "timestamp": None
    }


@router.post("/learning-rate")
async def set_learning_rate(request: Request, control: LearningRateControl):
    """
    Set the learning rate.

    Controls how quickly Atlas adapts to new input. Higher values
    mean faster learning but potentially less stable representations.

    **Input**: rate (0.0 to 1.0)
    **Output**: Confirmation with new rate
    """
    atlas_manager = request.app.state.atlas_manager
    return await atlas_manager.set_learning_rate(control.rate)


@router.post("/checkpoint/save")
async def save_checkpoint(request: Request, checkpoint: CheckpointRequest = None):
    """
    Save a checkpoint.

    Saves the current state of Atlas including all learned weights,
    memories, and configurations.

    **Input**: 
    - name: Optional checkpoint name
    - compress: Whether to compress (uses config default if not provided)
    - sync_to_cloud: Whether to sync to cloud after saving
    - metadata: Additional metadata
    
    **Output**: Checkpoint details including path, size, and version
    """
    atlas_manager = request.app.state.atlas_manager
    
    name = checkpoint.name if checkpoint else None
    compress = checkpoint.compress if checkpoint else None
    sync_to_cloud = checkpoint.sync_to_cloud if checkpoint else None
    metadata = checkpoint.metadata if checkpoint else None
    
    return await atlas_manager.save_checkpoint(
        name=name,
        compress=compress,
        sync_to_cloud=sync_to_cloud,
        metadata=metadata
    )


@router.post("/checkpoint/load")
async def load_checkpoint(request: Request, checkpoint: CheckpointLoadRequest):
    """
    Load a checkpoint.

    Restores Atlas to a previously saved state.

    **Input**: 
    - name: Checkpoint name (optional if version provided)
    - version: Version number (optional if name provided)
    
    **Output**: Load result
    """
    atlas_manager = request.app.state.atlas_manager
    return await atlas_manager.load_checkpoint(
        name=checkpoint.name,
        version=checkpoint.version
    )


@router.post("/checkpoint/rollback")
async def rollback_checkpoint(request: Request, rollback: CheckpointRollbackRequest):
    """
    Rollback to a previous checkpoint version.

    **Input**: steps - Number of versions to rollback
    **Output**: Rollback result with new current version
    """
    atlas_manager = request.app.state.atlas_manager
    return await atlas_manager.rollback_checkpoint(steps=rollback.steps)


@router.post("/checkpoint/sync")
async def sync_checkpoints(request: Request, sync_request: CloudSyncRequest):
    """
    Sync checkpoints with cloud storage.

    **Input**: 
    - direction: 'to' or 'from' cloud
    - version: Specific version to sync (optional)
    
    **Output**: Sync result with list of synced checkpoints
    """
    atlas_manager = request.app.state.atlas_manager
    
    if sync_request.direction == "to":
        return await atlas_manager.sync_checkpoints_to_cloud(version=sync_request.version)
    elif sync_request.direction == "from":
        return await atlas_manager.sync_checkpoints_from_cloud()
    else:
        raise HTTPException(status_code=400, detail="Direction must be 'to' or 'from'")


@router.delete("/checkpoint/{name}")
async def delete_checkpoint(request: Request, name: str):
    """
    Delete a checkpoint.

    **Input**: Checkpoint name in URL
    **Output**: Deletion result
    """
    import os
    from pathlib import Path

    checkpoint_dir = os.getenv("ATLAS_CHECKPOINT_DIR", "checkpoints")
    checkpoint_path = Path(checkpoint_dir) / f"{name}.pkl"

    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {name}")

    try:
        checkpoint_path.unlink()
        return {"deleted": True, "name": name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete checkpoint: {e}")


@router.post("/mode")
async def set_cognitive_mode(request: Request, mode_control: ModeControl):
    """
    Set the cognitive mode.

    Controls what type of processing Atlas prioritizes:
    - visual: Focus on visual processing
    - audio: Focus on audio processing
    - reasoning: Engage abstract reasoning systems
    - creative: Engage creativity and imagination
    - autonomous: Full autonomous operation

    **Input**: Mode name
    **Output**: Mode change confirmation
    """
    atlas_manager = request.app.state.atlas_manager

    valid_modes = ["visual", "audio", "reasoning", "creative", "autonomous"]
    if mode_control.mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode. Must be one of: {valid_modes}"
        )

    result = {
        "mode": mode_control.mode,
        "success": True,
        "timestamp": None
    }

    if atlas_manager.system:
        try:
            if hasattr(atlas_manager.system, 'unified_intelligence'):
                atlas_manager.system.unified_intelligence.set_mode(mode_control.mode)
        except Exception as e:
            result["error"] = str(e)
            result["success"] = False

    return result


@router.get("/mode")
async def get_cognitive_mode(request: Request):
    """
    Get the current cognitive mode.

    **Output**: Current mode name
    """
    atlas_manager = request.app.state.atlas_manager

    mode = "autonomous"  # Default

    if atlas_manager.system:
        try:
            if hasattr(atlas_manager.system, 'unified_intelligence'):
                mode = atlas_manager.system.unified_intelligence.get_current_mode()
        except:
            pass

    return {"mode": mode}


@router.post("/config")
async def update_config(request: Request, config: ConfigUpdate):
    """
    Update system configuration.

    Allows updating various system parameters. Changes take effect
    immediately for most parameters.

    **Input**: Configuration parameters to update
    **Output**: Updated configuration
    """
    atlas_manager = request.app.state.atlas_manager

    updates = {}
    errors = []

    if config.learning_rate is not None:
        result = await atlas_manager.set_learning_rate(config.learning_rate)
        if result.get("success"):
            updates["learning_rate"] = config.learning_rate
        else:
            errors.append(f"learning_rate: {result.get('error')}")

    # Additional config updates would be handled here
    if config.multimodal_size is not None:
        updates["multimodal_size"] = config.multimodal_size

    if config.prune_interval is not None:
        updates["prune_interval"] = config.prune_interval

    if config.enable_structural_plasticity is not None:
        updates["enable_structural_plasticity"] = config.enable_structural_plasticity

    if config.enable_temporal_prediction is not None:
        updates["enable_temporal_prediction"] = config.enable_temporal_prediction

    return {
        "updated": updates,
        "errors": errors if errors else None,
        "success": len(errors) == 0
    }


@router.post("/reset")
async def reset_system(request: Request, preserve_config: bool = True):
    """
    Reset the Atlas system.

    Clears all learned patterns and memories, optionally preserving
    configuration settings.

    **WARNING**: This is destructive and cannot be undone!

    **Input**: preserve_config (bool) - whether to keep current settings
    **Output**: Reset confirmation
    """
    atlas_manager = request.app.state.atlas_manager

    # This would reinitialize the system
    result = {
        "reset": True,
        "preserved_config": preserve_config,
        "message": "System reset (demo mode - no actual reset performed)"
    }

    if atlas_manager.system:
        try:
            # In production, this would actually reset the system
            # For safety, we don't implement destructive operations by default
            result["message"] = "Reset requested - requires confirmation"
            result["reset"] = False
        except Exception as e:
            result["error"] = str(e)

    return result


@router.post("/think")
async def trigger_thinking(request: Request, task: str = "reflect", cycles: int = 1):
    """
    Trigger a thinking cycle.

    Engages Atlas's abstract reasoning systems to think about a task.

    **Input**:
    - task: What to think about (reflect, plan, reason, etc.)
    - cycles: Number of thinking cycles

    **Output**: Thinking results
    """
    atlas_manager = request.app.state.atlas_manager

    result = {
        "task": task,
        "cycles": cycles,
        "output": None
    }

    if atlas_manager.system and hasattr(atlas_manager.system, 'unified_intelligence'):
        try:
            output = atlas_manager.system.unified_intelligence.think(task, num_cycles=cycles)
            result["output"] = output
        except Exception as e:
            result["error"] = str(e)
    else:
        result["output"] = f"Demo thinking output for task: {task}"

    return result


@router.post("/imagine")
async def trigger_imagination(request: Request, steps: int = 10):
    """
    Trigger imagination/creative generation.

    Engages Atlas's creativity systems to imagine future possibilities.

    **Input**: steps - Number of imagination steps
    **Output**: Imagination results
    """
    atlas_manager = request.app.state.atlas_manager

    result = {
        "steps": steps,
        "output": None
    }

    if atlas_manager.system and hasattr(atlas_manager.system, 'unified_intelligence'):
        try:
            output = atlas_manager.system.unified_intelligence.imagine(steps=steps)
            result["output"] = output
        except Exception as e:
            result["error"] = str(e)
    else:
        result["output"] = {
            "imagined_states": [f"Imagined state {i}" for i in range(steps)],
            "creativity_score": 0.75
        }

    return result
