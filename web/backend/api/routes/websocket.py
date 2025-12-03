"""
WebSocket API Routes

Real-time bidirectional communication between Atlas and the world.
"""

import asyncio
import json
from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self._counter = 0

    async def connect(self, websocket: WebSocket) -> str:
        """Accept a new connection and return its ID."""
        await websocket.accept()
        self._counter += 1
        connection_id = f"conn_{self._counter}"
        self.active_connections[connection_id] = websocket
        logger.info(f"WebSocket connected: {connection_id}")
        return connection_id

    def disconnect(self, connection_id: str):
        """Remove a connection."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            logger.info(f"WebSocket disconnected: {connection_id}")

    async def send_personal(self, connection_id: str, message: Dict[str, Any]):
        """Send a message to a specific connection."""
        if connection_id in self.active_connections:
            await self.active_connections[connection_id].send_json(message)

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connections."""
        for connection_id, websocket in list(self.active_connections.items()):
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {connection_id}: {e}")
                self.disconnect(connection_id)


manager = ConnectionManager()


@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Main WebSocket endpoint for real-time updates.

    This provides a bidirectional stream:

    **Atlas → World (Server → Client):**
    - System status updates
    - Processing events
    - Learning progress
    - Predictions
    - Memory changes

    **World → Atlas (Client → Server):**
    - Commands (start/stop learning, etc.)
    - Data input (frames, audio)
    - Configuration changes
    """
    connection_id = await manager.connect(websocket)
    atlas_manager = websocket.app.state.atlas_manager

    # Create a queue for this connection to receive updates
    update_queue = asyncio.Queue()
    atlas_manager.subscribe(update_queue)

    try:
        # Send initial state
        status = await atlas_manager.get_status()
        await websocket.send_json({
            "type": "connected",
            "connection_id": connection_id,
            "initial_status": status
        })

        # Create tasks for sending and receiving
        async def send_updates():
            """Send updates from Atlas to the client."""
            while True:
                try:
                    event = await update_queue.get()
                    await websocket.send_json(event)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error sending update: {e}")
                    break

        async def receive_commands():
            """Receive commands from the client."""
            while True:
                try:
                    data = await websocket.receive_json()
                    await handle_command(websocket, atlas_manager, data)
                except WebSocketDisconnect:
                    break
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error receiving command: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })

        async def send_heartbeat():
            """Send periodic heartbeat with metrics."""
            while True:
                try:
                    await asyncio.sleep(5)  # Every 5 seconds
                    metrics = await atlas_manager.get_metrics()
                    await websocket.send_json({
                        "type": "heartbeat",
                        "metrics": metrics
                    })
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error sending heartbeat: {e}")
                    break

        # Run all tasks concurrently
        send_task = asyncio.create_task(send_updates())
        receive_task = asyncio.create_task(receive_commands())
        heartbeat_task = asyncio.create_task(send_heartbeat())

        try:
            await asyncio.gather(send_task, receive_task, heartbeat_task)
        except Exception:
            pass
        finally:
            send_task.cancel()
            receive_task.cancel()
            heartbeat_task.cancel()

    except WebSocketDisconnect:
        pass
    finally:
        atlas_manager.unsubscribe(update_queue)
        manager.disconnect(connection_id)


async def handle_command(websocket: WebSocket, atlas_manager, data: Dict[str, Any]):
    """Handle a command from the client."""
    command = data.get("command")
    payload = data.get("payload", {})

    response = {
        "type": "command_response",
        "command": command,
        "timestamp": datetime.now().isoformat()
    }

    try:
        if command == "get_status":
            response["data"] = await atlas_manager.get_status()

        elif command == "get_metrics":
            response["data"] = await atlas_manager.get_metrics()

        elif command == "set_learning":
            enabled = payload.get("enabled", True)
            response["data"] = await atlas_manager.set_learning_enabled(enabled)

        elif command == "set_learning_rate":
            rate = payload.get("rate", 0.01)
            response["data"] = await atlas_manager.set_learning_rate(rate)

        elif command == "get_predictions":
            modality = payload.get("modality", "visual")
            num_steps = payload.get("num_steps", 5)
            response["data"] = await atlas_manager.get_predictions(modality, num_steps)

        elif command == "get_memory":
            memory_type = payload.get("memory_type", "episodic")
            limit = payload.get("limit", 100)
            response["data"] = await atlas_manager.get_memory_contents(memory_type, limit)

        elif command == "save_checkpoint":
            name = payload.get("name")
            response["data"] = await atlas_manager.save_checkpoint(name)

        elif command == "load_checkpoint":
            name = payload.get("name")
            if not name:
                response["error"] = "Checkpoint name required"
            else:
                response["data"] = await atlas_manager.load_checkpoint(name)

        elif command == "get_architecture":
            response["data"] = await atlas_manager.get_architecture_info()

        else:
            response["error"] = f"Unknown command: {command}"

    except Exception as e:
        logger.error(f"Error handling command {command}: {e}")
        response["error"] = str(e)

    await websocket.send_json(response)


@router.websocket("/data-stream")
async def websocket_data_stream(websocket: WebSocket):
    """
    WebSocket endpoint optimized for streaming data TO Atlas.

    Use this for high-frequency data input like video frames or audio.
    The client can send frames/audio as fast as Atlas can process them.
    """
    connection_id = await manager.connect(websocket)
    atlas_manager = websocket.app.state.atlas_manager

    try:
        await websocket.send_json({
            "type": "data_stream_ready",
            "connection_id": connection_id
        })

        while True:
            try:
                # Receive data (can be binary for efficiency)
                message = await websocket.receive()

                if "bytes" in message:
                    # Binary data - assume it's a frame or audio
                    # This would need proper framing protocol in production
                    pass

                elif "text" in message:
                    data = json.loads(message["text"])
                    data_type = data.get("type")

                    if data_type == "frame":
                        import base64
                        import numpy as np
                        from PIL import Image
                        import io

                        image_data = base64.b64decode(data["data"])
                        image = Image.open(io.BytesIO(image_data))
                        frame = np.array(image)

                        result = await atlas_manager.process_frame(
                            frame,
                            learn=data.get("learn", True)
                        )
                        await websocket.send_json({
                            "type": "frame_result",
                            "data": result
                        })

                    elif data_type == "audio":
                        import base64
                        import numpy as np

                        audio_data = base64.b64decode(data["data"])
                        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                        audio = audio / 32768.0

                        result = await atlas_manager.process_audio(
                            audio,
                            sample_rate=data.get("sample_rate", 22050),
                            learn=data.get("learn", True)
                        )
                        await websocket.send_json({
                            "type": "audio_result",
                            "data": result
                        })

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in data stream: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })

    finally:
        manager.disconnect(connection_id)
