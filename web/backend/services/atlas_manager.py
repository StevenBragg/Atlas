"""
Atlas Manager Service

This service manages the Atlas system instance and provides methods
for interacting with it through the web API.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import json

import numpy as np
from loguru import logger

# Add Atlas to path
atlas_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(atlas_root))

try:
    from self_organizing_av_system.core.system import SelfOrganizingAVSystem
    from self_organizing_av_system.models.visual.processor import VisualProcessor
    from self_organizing_av_system.models.audio.processor import AudioProcessor
    from self_organizing_av_system.config.configuration import SystemConfig
    ATLAS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Atlas core not available: {e}")
    ATLAS_AVAILABLE = False


class AtlasManager:
    """
    Manages the Atlas system instance and provides an interface for
    the web API to interact with it.
    """

    def __init__(self):
        self.system: Optional[SelfOrganizingAVSystem] = None
        self.config: Optional[SystemConfig] = None
        self.visual_processor: Optional[VisualProcessor] = None
        self.audio_processor: Optional[AudioProcessor] = None

        self._initialized = False
        self._learning_enabled = True
        self._processing_lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "frames_processed": 0,
            "audio_chunks_processed": 0,
            "start_time": None,
            "last_activity": None,
        }

        # Event subscribers for WebSocket
        self._subscribers: List[asyncio.Queue] = []

    async def initialize(self, config_path: Optional[str] = None) -> bool:
        """Initialize the Atlas system."""
        if not ATLAS_AVAILABLE:
            logger.warning("Atlas core not available - running in demo mode")
            self._initialized = True
            self._stats["start_time"] = datetime.now().isoformat()
            return True

        try:
            logger.info("Initializing Atlas system...")

            # Load configuration
            if config_path:
                self.config = SystemConfig.load(config_path)
            else:
                self.config = SystemConfig()

            # Initialize processors
            visual_config = self.config.get("visual", {})
            audio_config = self.config.get("audio", {})

            self.visual_processor = VisualProcessor(
                input_width=visual_config.get("input_width", 64),
                input_height=visual_config.get("input_height", 64),
                use_grayscale=visual_config.get("use_grayscale", True),
                patch_size=visual_config.get("patch_size", 8),
                stride=visual_config.get("stride", 4),
                layer_sizes=visual_config.get("layer_sizes", [200, 100, 50])
            )

            self.audio_processor = AudioProcessor(
                sample_rate=audio_config.get("sample_rate", 22050),
                n_mels=audio_config.get("n_mels", 64),
                layer_sizes=audio_config.get("layer_sizes", [150, 75, 40])
            )

            # Initialize main system
            system_config = self.config.get("system", {})
            self.system = SelfOrganizingAVSystem(
                visual_processor=self.visual_processor,
                audio_processor=self.audio_processor,
                multimodal_size=system_config.get("multimodal_size", 100),
                learning_rate=system_config.get("learning_rate", 0.01)
            )

            # Try to load latest checkpoint
            checkpoint_dir = os.getenv("ATLAS_CHECKPOINT_DIR", "checkpoints")
            if os.path.exists(checkpoint_dir):
                checkpoints = sorted(Path(checkpoint_dir).glob("*.pkl"))
                if checkpoints:
                    latest = checkpoints[-1]
                    logger.info(f"Loading checkpoint: {latest}")
                    try:
                        self.system.load_state(str(latest))
                    except Exception as e:
                        logger.warning(f"Failed to load checkpoint: {e}")

            self._initialized = True
            self._stats["start_time"] = datetime.now().isoformat()
            logger.info("Atlas system initialized successfully!")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Atlas: {e}")
            # Run in demo mode
            self._initialized = True
            self._stats["start_time"] = datetime.now().isoformat()
            return False

    async def shutdown(self):
        """Shutdown the Atlas system and save state."""
        if self.system:
            try:
                checkpoint_dir = os.getenv("ATLAS_CHECKPOINT_DIR", "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = Path(checkpoint_dir) / f"shutdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                self.system.save_state(str(checkpoint_path))
                logger.info(f"Saved shutdown checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")

        self._initialized = False

    def is_initialized(self) -> bool:
        """Check if Atlas is initialized."""
        return self._initialized

    async def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        status = {
            "initialized": self._initialized,
            "atlas_available": ATLAS_AVAILABLE,
            "learning_enabled": self._learning_enabled,
            "stats": self._stats.copy(),
            "timestamp": datetime.now().isoformat()
        }

        if self.system and ATLAS_AVAILABLE:
            try:
                # Get system state
                state = self.system.get_system_state()
                status["system_state"] = {
                    "total_frames": state.get("total_frames_processed", 0),
                    "learning_step": state.get("learning_step", 0),
                }

                # Get architecture info
                arch_info = self.system.get_architecture_info()
                status["architecture"] = {
                    "visual_layers": arch_info.get("visual", {}),
                    "audio_layers": arch_info.get("audio", {}),
                    "multimodal_size": arch_info.get("multimodal_size", 0)
                }
            except Exception as e:
                logger.error(f"Error getting system state: {e}")

        return status

    async def get_metrics(self) -> Dict[str, Any]:
        """Get learning and performance metrics."""
        metrics = {
            "frames_processed": self._stats["frames_processed"],
            "audio_chunks_processed": self._stats["audio_chunks_processed"],
            "uptime_seconds": 0,
            "timestamp": datetime.now().isoformat()
        }

        if self._stats["start_time"]:
            start = datetime.fromisoformat(self._stats["start_time"])
            metrics["uptime_seconds"] = (datetime.now() - start).total_seconds()

        if self.system and ATLAS_AVAILABLE:
            try:
                state = self.system.get_system_state()
                metrics.update({
                    "prediction_error": state.get("prediction_error", 0),
                    "reconstruction_error": state.get("reconstruction_error", 0),
                    "cross_modal_correlation": state.get("cross_modal_correlation", 0),
                    "total_neurons": state.get("total_neurons", 0),
                    "active_associations": state.get("active_associations", 0)
                })
            except Exception as e:
                logger.error(f"Error getting metrics: {e}")

        return metrics

    async def process_frame(self, frame: np.ndarray, learn: bool = True) -> Dict[str, Any]:
        """Process a visual frame through Atlas."""
        async with self._processing_lock:
            self._stats["frames_processed"] += 1
            self._stats["last_activity"] = datetime.now().isoformat()

            result = {
                "processed": True,
                "frame_number": self._stats["frames_processed"],
                "timestamp": datetime.now().isoformat()
            }

            if self.system and ATLAS_AVAILABLE:
                try:
                    # Process frame through visual pathway
                    should_learn = learn and self._learning_enabled
                    self.system.process_av_pair(frame, None, learn=should_learn)

                    # Get predictions
                    predictions = self.system.get_temporal_predictions(num_steps=3)
                    result["predictions"] = {
                        "temporal": [p.tolist() if isinstance(p, np.ndarray) else p for p in predictions]
                    }

                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    result["error"] = str(e)

            # Notify subscribers
            await self._notify_subscribers({
                "type": "frame_processed",
                "data": result
            })

            return result

    async def process_audio(self, audio_data: np.ndarray, sample_rate: int = 22050, learn: bool = True) -> Dict[str, Any]:
        """Process audio data through Atlas."""
        async with self._processing_lock:
            self._stats["audio_chunks_processed"] += 1
            self._stats["last_activity"] = datetime.now().isoformat()

            result = {
                "processed": True,
                "chunk_number": self._stats["audio_chunks_processed"],
                "sample_rate": sample_rate,
                "duration_ms": len(audio_data) / sample_rate * 1000,
                "timestamp": datetime.now().isoformat()
            }

            if self.system and ATLAS_AVAILABLE:
                try:
                    # Process audio through auditory pathway
                    should_learn = learn and self._learning_enabled
                    self.system.process_av_pair(None, audio_data, learn=should_learn)

                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
                    result["error"] = str(e)

            # Notify subscribers
            await self._notify_subscribers({
                "type": "audio_processed",
                "data": result
            })

            return result

    async def process_av_pair(self, frame: Optional[np.ndarray], audio: Optional[np.ndarray], learn: bool = True) -> Dict[str, Any]:
        """Process synchronized audio-visual pair."""
        async with self._processing_lock:
            if frame is not None:
                self._stats["frames_processed"] += 1
            if audio is not None:
                self._stats["audio_chunks_processed"] += 1
            self._stats["last_activity"] = datetime.now().isoformat()

            result = {
                "processed": True,
                "timestamp": datetime.now().isoformat()
            }

            if self.system and ATLAS_AVAILABLE:
                try:
                    should_learn = learn and self._learning_enabled
                    self.system.process_av_pair(frame, audio, learn=should_learn)

                    # Get cross-modal prediction
                    cross_modal = self.system.get_cross_modal_prediction()
                    if cross_modal is not None:
                        result["cross_modal_prediction"] = cross_modal.tolist() if isinstance(cross_modal, np.ndarray) else cross_modal

                except Exception as e:
                    logger.error(f"Error processing AV pair: {e}")
                    result["error"] = str(e)

            await self._notify_subscribers({
                "type": "av_processed",
                "data": result
            })

            return result

    async def get_predictions(self, modality: str = "visual", num_steps: int = 5) -> Dict[str, Any]:
        """Get temporal and cross-modal predictions."""
        predictions = {
            "modality": modality,
            "num_steps": num_steps,
            "timestamp": datetime.now().isoformat()
        }

        if self.system and ATLAS_AVAILABLE:
            try:
                temporal = self.system.get_temporal_predictions(num_steps=num_steps)
                predictions["temporal"] = [
                    p.tolist() if isinstance(p, np.ndarray) else p
                    for p in temporal
                ]

                cross_modal = self.system.get_cross_modal_prediction(modality=modality)
                if cross_modal is not None:
                    predictions["cross_modal"] = cross_modal.tolist() if isinstance(cross_modal, np.ndarray) else cross_modal

            except Exception as e:
                logger.error(f"Error getting predictions: {e}")
                predictions["error"] = str(e)
        else:
            # Demo data
            predictions["temporal"] = [[0.1, 0.2, 0.3] for _ in range(num_steps)]
            predictions["cross_modal"] = [0.5, 0.5, 0.5]

        return predictions

    async def get_memory_contents(self, memory_type: str = "episodic", limit: int = 100) -> Dict[str, Any]:
        """Get contents of Atlas memory systems."""
        contents = {
            "memory_type": memory_type,
            "timestamp": datetime.now().isoformat(),
            "items": []
        }

        if self.system and ATLAS_AVAILABLE:
            try:
                state = self.system.get_system_state()

                if memory_type == "episodic":
                    # Get episodic memories if available
                    if hasattr(self.system, 'unified_intelligence'):
                        ui = self.system.unified_intelligence
                        if hasattr(ui, 'episodic_memory'):
                            memories = ui.episodic_memory.retrieve_recent(limit)
                            contents["items"] = [
                                {
                                    "id": i,
                                    "timestamp": m.get("timestamp", ""),
                                    "summary": m.get("summary", ""),
                                    "importance": m.get("importance", 0)
                                }
                                for i, m in enumerate(memories)
                            ]

                elif memory_type == "semantic":
                    # Get semantic concepts
                    if hasattr(self.system, 'unified_intelligence'):
                        ui = self.system.unified_intelligence
                        if hasattr(ui, 'semantic_memory'):
                            concepts = ui.semantic_memory.get_all_concepts(limit)
                            contents["items"] = [
                                {
                                    "id": i,
                                    "name": c.get("name", ""),
                                    "category": c.get("category", ""),
                                    "connections": c.get("connections", 0)
                                }
                                for i, c in enumerate(concepts)
                            ]

                elif memory_type == "working":
                    # Get working memory contents
                    if hasattr(self.system, 'unified_intelligence'):
                        ui = self.system.unified_intelligence
                        if hasattr(ui, 'working_memory'):
                            items = ui.working_memory.get_active_items()
                            contents["items"] = items

            except Exception as e:
                logger.error(f"Error getting memory contents: {e}")
                contents["error"] = str(e)
        else:
            # Demo data
            contents["items"] = [
                {"id": i, "content": f"Demo {memory_type} memory item {i}"}
                for i in range(min(5, limit))
            ]

        return contents

    async def set_learning_enabled(self, enabled: bool) -> Dict[str, Any]:
        """Enable or disable learning."""
        self._learning_enabled = enabled
        result = {
            "learning_enabled": self._learning_enabled,
            "timestamp": datetime.now().isoformat()
        }

        await self._notify_subscribers({
            "type": "learning_state_changed",
            "data": result
        })

        return result

    async def set_learning_rate(self, rate: float) -> Dict[str, Any]:
        """Set the learning rate."""
        result = {
            "learning_rate": rate,
            "timestamp": datetime.now().isoformat()
        }

        if self.system and ATLAS_AVAILABLE:
            try:
                self.system.learning_rate = rate
                result["success"] = True
            except Exception as e:
                logger.error(f"Error setting learning rate: {e}")
                result["error"] = str(e)
        else:
            result["success"] = True  # Demo mode

        return result

    async def save_checkpoint(self, name: Optional[str] = None, compress: Optional[bool] = None, sync_to_cloud: Optional[bool] = None, metadata: Optional[dict] = None) -> Dict[str, Any]:
        """Save a checkpoint of the current state."""
        checkpoint_dir = os.getenv("ATLAS_CHECKPOINT_DIR", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        if name is None:
            name = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        checkpoint_path = Path(checkpoint_dir) / f"{name}.pkl"
        result = {
            "checkpoint_name": name,
            "path": str(checkpoint_path),
            "timestamp": datetime.now().isoformat()
        }

        if self.system and ATLAS_AVAILABLE:
            try:
                self.system.save_state(str(checkpoint_path))
                result["success"] = True
                result["size_bytes"] = checkpoint_path.stat().st_size
            except Exception as e:
                logger.error(f"Error saving checkpoint: {e}")
                result["error"] = str(e)
                result["success"] = False
        else:
            result["success"] = True  # Demo mode

        return result

    async def load_checkpoint(self, name: str) -> Dict[str, Any]:
        """Load a checkpoint."""
        checkpoint_dir = os.getenv("ATLAS_CHECKPOINT_DIR", "checkpoints")
        checkpoint_path = Path(checkpoint_dir) / f"{name}.pkl"

        result = {
            "checkpoint_name": name,
            "path": str(checkpoint_path),
            "timestamp": datetime.now().isoformat()
        }

        if not checkpoint_path.exists():
            result["error"] = f"Checkpoint not found: {checkpoint_path}"
            result["success"] = False
            return result

        if self.system and ATLAS_AVAILABLE:
            try:
                self.system.load_state(str(checkpoint_path))
                result["success"] = True
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                result["error"] = str(e)
                result["success"] = False
        else:
            result["success"] = True  # Demo mode

        return result

    async def list_checkpoints(self) -> Dict[str, Any]:
        """List available checkpoints."""
        checkpoint_dir = os.getenv("ATLAS_CHECKPOINT_DIR", "checkpoints")

        result = {
            "checkpoint_dir": checkpoint_dir,
            "checkpoints": [],
            "timestamp": datetime.now().isoformat()
        }

        if os.path.exists(checkpoint_dir):
            for f in sorted(Path(checkpoint_dir).glob("*.pkl")):
                result["checkpoints"].append({
                    "name": f.stem,
                    "path": str(f),
                    "size_bytes": f.stat().st_size,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                })

        return result

    async def get_architecture_info(self) -> Dict[str, Any]:
        """Get detailed architecture information."""
        info = {
            "timestamp": datetime.now().isoformat()
        }

        if self.system and ATLAS_AVAILABLE:
            try:
                arch = self.system.get_architecture_info()
                info.update(arch)
            except Exception as e:
                logger.error(f"Error getting architecture info: {e}")
                info["error"] = str(e)
        else:
            # Demo architecture
            info["visual"] = {
                "layers": [
                    {"name": "V1", "neurons": 200, "type": "edge_detectors"},
                    {"name": "V2", "neurons": 100, "type": "texture_patterns"},
                    {"name": "V3", "neurons": 50, "type": "object_features"}
                ]
            }
            info["audio"] = {
                "layers": [
                    {"name": "A1", "neurons": 150, "type": "frequency_bands"},
                    {"name": "A2", "neurons": 75, "type": "spectral_patterns"},
                    {"name": "A3", "neurons": 40, "type": "sound_objects"}
                ]
            }
            info["multimodal"] = {
                "size": 100,
                "associations": 250
            }

        return info

    # WebSocket subscription management
    def subscribe(self, queue: asyncio.Queue):
        """Subscribe to real-time updates."""
        self._subscribers.append(queue)
        logger.debug(f"New subscriber added. Total: {len(self._subscribers)}")

    def unsubscribe(self, queue: asyncio.Queue):
        """Unsubscribe from real-time updates."""
        if queue in self._subscribers:
            self._subscribers.remove(queue)
            logger.debug(f"Subscriber removed. Total: {len(self._subscribers)}")

    async def _notify_subscribers(self, event: Dict[str, Any]):
        """Notify all subscribers of an event."""
        for queue in self._subscribers:
            try:
                await queue.put(event)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
