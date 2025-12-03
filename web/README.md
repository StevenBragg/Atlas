# Atlas Web Interface

A React frontend with Python microservice backend for interfacing with the Atlas self-organizing audio-visual learning system.

## Architecture

```
web/
├── frontend/          # React + TypeScript + Vite application
│   ├── src/
│   │   ├── components/    # Reusable UI components
│   │   ├── pages/         # Page components
│   │   ├── hooks/         # Custom React hooks
│   │   ├── services/      # API services
│   │   └── types/         # TypeScript types
│   └── public/            # Static assets
│
├── backend/           # Python FastAPI microservice
│   ├── api/
│   │   └── routes/        # API route handlers
│   └── services/          # Business logic
│
├── docker-compose.yml     # Container orchestration
├── Dockerfile             # Production build
└── Dockerfile.backend     # Development backend
```

## Bidirectional Interface

### World → Atlas (Input)

The web interface allows the world to send data to Atlas:

- **Visual Data**: Upload images or stream webcam frames
- **Audio Data**: Upload audio files or stream microphone input
- **Commands**: Control learning, configuration, and system state

API Endpoints:
- `POST /api/data/frame` - Send an image for processing
- `POST /api/data/audio` - Send audio for processing
- `POST /api/data/av-pair` - Send synchronized audio-visual pairs
- `POST /api/control/learning` - Enable/disable learning
- `POST /api/control/mode` - Set cognitive mode

### Atlas → World (Output)

The web interface allows observing Atlas's internal state:

- **System Status**: Real-time health and metrics
- **Predictions**: Temporal and cross-modal predictions
- **Memory Access**: Browse episodic, semantic, and working memory
- **Architecture**: Visualize neural network structure

API Endpoints:
- `GET /api/system/status` - Get system status
- `GET /api/system/metrics` - Get learning metrics
- `GET /api/memory/episodic` - Access episodic memories
- `GET /api/memory/semantic` - Access semantic knowledge
- `GET /api/memory/associations` - View cross-modal associations
- `WS /ws/stream` - Real-time WebSocket updates

## Quick Start

### Development Mode

1. Start the backend:
```bash
cd web/backend
pip install -r requirements.txt
python main.py
```

2. Start the frontend:
```bash
cd web/frontend
npm install
npm run dev
```

3. Open http://localhost:3000

### Docker Mode

Production build:
```bash
cd web
docker-compose up atlas-web
```

Development with hot-reload:
```bash
cd web
docker-compose --profile dev up
```

## Pages

### Dashboard
Real-time overview of Atlas's learning progress, metrics, and neural network activity.

### Data Input
Interface for feeding visual and audio data to Atlas:
- Image upload
- Webcam capture
- Audio upload
- Learning mode toggle

### Memory
Browse Atlas's memory systems:
- Episodic memories (experiences)
- Semantic memory (concepts)
- Working memory (active attention)
- Cross-modal associations

### Control
Configure and control Atlas:
- Enable/disable learning
- Adjust learning rate
- Set cognitive mode
- Save/load checkpoints
- Trigger thinking and imagination

### Architecture
Visualize the neural network structure:
- Visual pathway (V1 → V2 → V3)
- Audio pathway (A1 → A2 → A3)
- Multimodal integration layer

## API Reference

See the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## WebSocket Events

Connect to `ws://localhost:8000/ws/stream` for real-time updates:

### Server → Client
- `connected` - Initial connection with system status
- `heartbeat` - Periodic metrics update (every 5s)
- `frame_processed` - Frame processing complete
- `audio_processed` - Audio processing complete
- `learning_state_changed` - Learning enabled/disabled

### Client → Server
- `get_status` - Request current status
- `get_metrics` - Request current metrics
- `set_learning` - Enable/disable learning
- `set_learning_rate` - Update learning rate
- `get_predictions` - Request predictions
- `save_checkpoint` - Save current state
- `load_checkpoint` - Load saved state

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ATLAS_WEB_PORT` | 8000 | Backend server port |
| `ATLAS_WEB_HOST` | 0.0.0.0 | Backend server host |
| `ATLAS_CHECKPOINT_DIR` | checkpoints | Checkpoint storage directory |
| `LOG_LEVEL` | INFO | Logging level |
| `ATLAS_ENABLE_UNIFIED_INTELLIGENCE` | true | Enable cognitive systems |

## Technology Stack

**Frontend:**
- React 18
- TypeScript
- Vite
- TailwindCSS
- Recharts (charts)
- Lucide React (icons)

**Backend:**
- Python 3.11
- FastAPI
- WebSockets
- Pydantic
- Uvicorn

**Infrastructure:**
- Docker
- Docker Compose
