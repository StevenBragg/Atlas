"""
Data Input API Routes

Endpoints for sending data TO Atlas - this is how the world
interfaces with Atlas to provide sensory input.
"""

import base64
import io
from typing import Optional, List

import numpy as np
from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

router = APIRouter()


class FrameInput(BaseModel):
    """Frame input via base64 encoded image."""
    image_base64: str = Field(..., description="Base64 encoded image data")
    learn: bool = Field(True, description="Whether to learn from this frame")


class AudioInput(BaseModel):
    """Audio input via base64 encoded audio."""
    audio_base64: str = Field(..., description="Base64 encoded audio data (raw PCM)")
    sample_rate: int = Field(22050, description="Audio sample rate")
    learn: bool = Field(True, description="Whether to learn from this audio")


class AVPairInput(BaseModel):
    """Synchronized audio-visual pair input."""
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio (raw PCM)")
    sample_rate: int = Field(22050, description="Audio sample rate")
    learn: bool = Field(True, description="Whether to learn from this input")


class ProcessingResult(BaseModel):
    """Result of processing input."""
    processed: bool
    timestamp: str
    frame_number: Optional[int] = None
    chunk_number: Optional[int] = None
    predictions: Optional[dict] = None
    cross_modal_prediction: Optional[List[float]] = None
    error: Optional[str] = None


def decode_image(base64_str: str) -> np.ndarray:
    """Decode base64 image to numpy array."""
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        return np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


def decode_audio(base64_str: str) -> np.ndarray:
    """Decode base64 audio to numpy array."""
    try:
        audio_data = base64.b64decode(base64_str)
        # Assume 16-bit PCM audio
        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        audio = audio / 32768.0  # Normalize to [-1, 1]
        return audio
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio data: {e}")


@router.post("/frame", response_model=ProcessingResult)
async def process_frame(request: Request, frame_input: FrameInput):
    """
    Process a single visual frame.

    Send an image to Atlas for visual processing. The image is processed
    through the visual pathway and, if learning is enabled, associations
    are formed and the network adapts.

    **Input**: Base64 encoded image (PNG, JPEG, etc.)
    **Output**: Processing result with predictions
    """
    atlas_manager = request.app.state.atlas_manager

    frame = decode_image(frame_input.image_base64)
    return await atlas_manager.process_frame(frame, learn=frame_input.learn)


@router.post("/frame/upload", response_model=ProcessingResult)
async def upload_frame(
    request: Request,
    file: UploadFile = File(...),
    learn: bool = Form(True)
):
    """
    Process an uploaded image file.

    Alternative to base64 encoding - upload an image file directly.

    **Input**: Image file (PNG, JPEG, etc.)
    **Output**: Processing result with predictions
    """
    atlas_manager = request.app.state.atlas_manager

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
        frame = np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    return await atlas_manager.process_frame(frame, learn=learn)


@router.post("/audio", response_model=ProcessingResult)
async def process_audio(request: Request, audio_input: AudioInput):
    """
    Process an audio chunk.

    Send audio data to Atlas for auditory processing. The audio is processed
    through the auditory pathway and, if learning is enabled, associations
    are formed and the network adapts.

    **Input**: Base64 encoded raw PCM audio (16-bit signed integer)
    **Output**: Processing result
    """
    atlas_manager = request.app.state.atlas_manager

    audio = decode_audio(audio_input.audio_base64)
    return await atlas_manager.process_audio(
        audio,
        sample_rate=audio_input.sample_rate,
        learn=audio_input.learn
    )


@router.post("/audio/upload", response_model=ProcessingResult)
async def upload_audio(
    request: Request,
    file: UploadFile = File(...),
    sample_rate: int = Form(22050),
    learn: bool = Form(True)
):
    """
    Process an uploaded audio file.

    Upload raw PCM audio data directly.

    **Input**: Raw PCM audio file (16-bit signed integer)
    **Output**: Processing result
    """
    atlas_manager = request.app.state.atlas_manager

    contents = await file.read()
    try:
        audio = np.frombuffer(contents, dtype=np.int16).astype(np.float32)
        audio = audio / 32768.0
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")

    return await atlas_manager.process_audio(audio, sample_rate=sample_rate, learn=learn)


@router.post("/av-pair", response_model=ProcessingResult)
async def process_av_pair(request: Request, av_input: AVPairInput):
    """
    Process a synchronized audio-visual pair.

    Send synchronized audio and visual data to Atlas. This is the ideal
    input method as it allows cross-modal learning between vision and audio.

    **Input**: Base64 encoded image and/or audio
    **Output**: Processing result with cross-modal predictions
    """
    atlas_manager = request.app.state.atlas_manager

    frame = None
    audio = None

    if av_input.image_base64:
        frame = decode_image(av_input.image_base64)

    if av_input.audio_base64:
        audio = decode_audio(av_input.audio_base64)

    if frame is None and audio is None:
        raise HTTPException(status_code=400, detail="Must provide image and/or audio")

    return await atlas_manager.process_av_pair(frame, audio, learn=av_input.learn)


@router.get("/predictions")
async def get_predictions(
    request: Request,
    modality: str = "visual",
    num_steps: int = 5
):
    """
    Get predictions from Atlas.

    Retrieve temporal predictions (future state forecasts) and cross-modal
    predictions (predicting audio from visual or vice versa).

    **Parameters**:
    - modality: "visual" or "audio"
    - num_steps: Number of future steps to predict

    **Output**: Temporal and cross-modal predictions
    """
    atlas_manager = request.app.state.atlas_manager

    if modality not in ["visual", "audio"]:
        raise HTTPException(status_code=400, detail="Modality must be 'visual' or 'audio'")

    return await atlas_manager.get_predictions(modality=modality, num_steps=num_steps)
