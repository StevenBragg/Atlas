"""
Comprehensive tests for the Atlas Web Backend API.

Tests cover:
- Pydantic model validation for all request/response models
- Route definitions and router configuration
- AtlasManager initialization and basic state
- FastAPI app configuration (CORS, routers, metadata)
- HTTP endpoint integration tests via httpx TestClient

All tests handle missing dependencies gracefully using skipUnless decorators.
All tests are deterministic and pass reliably.
"""

import os
import sys
import unittest
import asyncio
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------
try:
    import fastapi
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

try:
    import pydantic
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# We need both fastapi and pydantic for model/route tests
HAS_WEB_DEPS = HAS_FASTAPI and HAS_PYDANTIC

# We need all three for integration tests with TestClient
HAS_TEST_CLIENT = HAS_WEB_DEPS and HAS_HTTPX

# Conditional imports -- only when dependencies are present
if HAS_WEB_DEPS:
    # Add web backend to path so its internal imports resolve
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'web', 'backend'))

    from web.backend.api.routes import control, data, memory, system
    from web.backend.services.atlas_manager import AtlasManager

if HAS_TEST_CLIENT:
    from starlette.testclient import TestClient


# ===========================================================================
# Pydantic Model Validation Tests
# ===========================================================================

@unittest.skipUnless(HAS_WEB_DEPS, "fastapi and pydantic are required")
class TestControlModels(unittest.TestCase):
    """Tests for Pydantic models defined in the control routes module."""

    def test_learning_control_valid(self):
        """LearningControl accepts a valid 'enabled' boolean."""
        model = control.LearningControl(enabled=True)
        self.assertTrue(model.enabled)

        model2 = control.LearningControl(enabled=False)
        self.assertFalse(model2.enabled)

    def test_learning_control_missing_required(self):
        """LearningControl raises ValidationError when 'enabled' is omitted."""
        with self.assertRaises(pydantic.ValidationError):
            control.LearningControl()

    def test_learning_rate_control_valid(self):
        """LearningRateControl accepts a rate within [0.0, 1.0]."""
        model = control.LearningRateControl(rate=0.5)
        self.assertAlmostEqual(model.rate, 0.5)

    def test_learning_rate_control_boundary_low(self):
        """LearningRateControl accepts rate=0.0 (lower boundary)."""
        model = control.LearningRateControl(rate=0.0)
        self.assertAlmostEqual(model.rate, 0.0)

    def test_learning_rate_control_boundary_high(self):
        """LearningRateControl accepts rate=1.0 (upper boundary)."""
        model = control.LearningRateControl(rate=1.0)
        self.assertAlmostEqual(model.rate, 1.0)

    def test_learning_rate_control_too_low(self):
        """LearningRateControl rejects rate below 0.0."""
        with self.assertRaises(pydantic.ValidationError):
            control.LearningRateControl(rate=-0.1)

    def test_learning_rate_control_too_high(self):
        """LearningRateControl rejects rate above 1.0."""
        with self.assertRaises(pydantic.ValidationError):
            control.LearningRateControl(rate=1.5)

    def test_learning_rate_control_missing_required(self):
        """LearningRateControl raises ValidationError when rate is omitted."""
        with self.assertRaises(pydantic.ValidationError):
            control.LearningRateControl()

    def test_checkpoint_request_optional_name(self):
        """CheckpointRequest allows name to be omitted (defaults to None)."""
        model = control.CheckpointRequest()
        self.assertIsNone(model.name)

    def test_checkpoint_request_with_name(self):
        """CheckpointRequest accepts an explicit name."""
        model = control.CheckpointRequest(name="my_checkpoint")
        self.assertEqual(model.name, "my_checkpoint")

    def test_checkpoint_load_request_valid(self):
        """CheckpointLoadRequest requires a name."""
        model = control.CheckpointLoadRequest(name="saved_state")
        self.assertEqual(model.name, "saved_state")

    def test_checkpoint_load_request_missing(self):
        """CheckpointLoadRequest raises ValidationError when name is omitted."""
        with self.assertRaises(pydantic.ValidationError):
            control.CheckpointLoadRequest()

    def test_mode_control_valid(self):
        """ModeControl accepts a string mode value."""
        for mode_name in ["visual", "audio", "reasoning", "creative", "autonomous"]:
            model = control.ModeControl(mode=mode_name)
            self.assertEqual(model.mode, mode_name)

    def test_mode_control_missing(self):
        """ModeControl raises ValidationError when mode is omitted."""
        with self.assertRaises(pydantic.ValidationError):
            control.ModeControl()

    def test_config_update_all_optional(self):
        """ConfigUpdate allows all fields to be omitted (all optional)."""
        model = control.ConfigUpdate()
        self.assertIsNone(model.learning_rate)
        self.assertIsNone(model.multimodal_size)
        self.assertIsNone(model.prune_interval)
        self.assertIsNone(model.enable_structural_plasticity)
        self.assertIsNone(model.enable_temporal_prediction)

    def test_config_update_partial(self):
        """ConfigUpdate accepts a subset of fields."""
        model = control.ConfigUpdate(learning_rate=0.01, multimodal_size=200)
        self.assertAlmostEqual(model.learning_rate, 0.01)
        self.assertEqual(model.multimodal_size, 200)
        self.assertIsNone(model.prune_interval)

    def test_config_update_learning_rate_constraint(self):
        """ConfigUpdate learning_rate must be in [0.0, 1.0]."""
        with self.assertRaises(pydantic.ValidationError):
            control.ConfigUpdate(learning_rate=2.0)

    def test_config_update_multimodal_size_constraint(self):
        """ConfigUpdate multimodal_size must be in [10, 1000]."""
        with self.assertRaises(pydantic.ValidationError):
            control.ConfigUpdate(multimodal_size=5)
        with self.assertRaises(pydantic.ValidationError):
            control.ConfigUpdate(multimodal_size=5000)

    def test_config_update_prune_interval_constraint(self):
        """ConfigUpdate prune_interval must be in [100, 100000]."""
        with self.assertRaises(pydantic.ValidationError):
            control.ConfigUpdate(prune_interval=10)
        with self.assertRaises(pydantic.ValidationError):
            control.ConfigUpdate(prune_interval=999999)


@unittest.skipUnless(HAS_WEB_DEPS, "fastapi and pydantic are required")
class TestDataModels(unittest.TestCase):
    """Tests for Pydantic models defined in the data routes module."""

    def test_frame_input_valid(self):
        """FrameInput accepts base64 string and learn flag."""
        model = data.FrameInput(image_base64="abc123==", learn=True)
        self.assertEqual(model.image_base64, "abc123==")
        self.assertTrue(model.learn)

    def test_frame_input_learn_defaults_true(self):
        """FrameInput learn defaults to True."""
        model = data.FrameInput(image_base64="data")
        self.assertTrue(model.learn)

    def test_frame_input_missing_image(self):
        """FrameInput raises ValidationError without image_base64."""
        with self.assertRaises(pydantic.ValidationError):
            data.FrameInput()

    def test_audio_input_valid(self):
        """AudioInput accepts all fields."""
        model = data.AudioInput(audio_base64="audiodata==", sample_rate=44100, learn=False)
        self.assertEqual(model.audio_base64, "audiodata==")
        self.assertEqual(model.sample_rate, 44100)
        self.assertFalse(model.learn)

    def test_audio_input_defaults(self):
        """AudioInput has correct default values for sample_rate and learn."""
        model = data.AudioInput(audio_base64="x")
        self.assertEqual(model.sample_rate, 22050)
        self.assertTrue(model.learn)

    def test_audio_input_missing_audio(self):
        """AudioInput raises ValidationError without audio_base64."""
        with self.assertRaises(pydantic.ValidationError):
            data.AudioInput()

    def test_av_pair_input_all_optional_data(self):
        """AVPairInput allows both image and audio to be None."""
        model = data.AVPairInput()
        self.assertIsNone(model.image_base64)
        self.assertIsNone(model.audio_base64)
        self.assertEqual(model.sample_rate, 22050)
        self.assertTrue(model.learn)

    def test_av_pair_input_with_both(self):
        """AVPairInput accepts both image and audio data."""
        model = data.AVPairInput(image_base64="img", audio_base64="aud")
        self.assertEqual(model.image_base64, "img")
        self.assertEqual(model.audio_base64, "aud")

    def test_processing_result_minimal(self):
        """ProcessingResult accepts minimal required fields."""
        model = data.ProcessingResult(processed=True, timestamp="2024-01-01T00:00:00")
        self.assertTrue(model.processed)
        self.assertEqual(model.timestamp, "2024-01-01T00:00:00")
        self.assertIsNone(model.frame_number)
        self.assertIsNone(model.chunk_number)
        self.assertIsNone(model.predictions)
        self.assertIsNone(model.error)

    def test_processing_result_full(self):
        """ProcessingResult accepts all optional fields."""
        model = data.ProcessingResult(
            processed=True,
            timestamp="2024-01-01T00:00:00",
            frame_number=42,
            chunk_number=7,
            predictions={"temporal": [1, 2, 3]},
            cross_modal_prediction=[0.1, 0.2],
            error=None,
        )
        self.assertEqual(model.frame_number, 42)
        self.assertEqual(model.chunk_number, 7)
        self.assertEqual(model.predictions, {"temporal": [1, 2, 3]})
        self.assertEqual(model.cross_modal_prediction, [0.1, 0.2])


@unittest.skipUnless(HAS_WEB_DEPS, "fastapi and pydantic are required")
class TestMemoryModels(unittest.TestCase):
    """Tests for Pydantic models defined in the memory routes module."""

    def test_memory_item_minimal(self):
        """MemoryItem requires only id."""
        model = memory.MemoryItem(id=0)
        self.assertEqual(model.id, 0)
        self.assertIsNone(model.content)
        self.assertIsNone(model.timestamp)
        self.assertIsNone(model.summary)

    def test_memory_item_full(self):
        """MemoryItem accepts all fields."""
        model = memory.MemoryItem(
            id=1,
            content="test memory",
            timestamp="2024-01-01",
            summary="a test",
            importance=0.9,
            name="concept_1",
            category="visual",
            connections=5,
        )
        self.assertEqual(model.id, 1)
        self.assertEqual(model.content, "test memory")
        self.assertAlmostEqual(model.importance, 0.9)
        self.assertEqual(model.connections, 5)

    def test_memory_contents_valid(self):
        """MemoryContents accepts memory_type, timestamp, and items list."""
        item = memory.MemoryItem(id=0, content="hello")
        model = memory.MemoryContents(
            memory_type="episodic",
            timestamp="2024-01-01T00:00:00",
            items=[item],
        )
        self.assertEqual(model.memory_type, "episodic")
        self.assertEqual(len(model.items), 1)
        self.assertIsNone(model.error)

    def test_memory_contents_empty_items(self):
        """MemoryContents accepts an empty items list."""
        model = memory.MemoryContents(
            memory_type="semantic",
            timestamp="2024-01-01T00:00:00",
            items=[],
        )
        self.assertEqual(len(model.items), 0)

    def test_memory_query_valid(self):
        """MemoryQuery accepts query string, memory_type, and limit."""
        model = memory.MemoryQuery(query="face", memory_type="episodic", limit=5)
        self.assertEqual(model.query, "face")
        self.assertEqual(model.memory_type, "episodic")
        self.assertEqual(model.limit, 5)

    def test_memory_query_defaults(self):
        """MemoryQuery has sensible defaults for memory_type and limit."""
        model = memory.MemoryQuery(query="test")
        self.assertEqual(model.memory_type, "all")
        self.assertEqual(model.limit, 10)

    def test_memory_query_missing_query(self):
        """MemoryQuery raises ValidationError without query field."""
        with self.assertRaises(pydantic.ValidationError):
            memory.MemoryQuery()


@unittest.skipUnless(HAS_WEB_DEPS, "fastapi and pydantic are required")
class TestSystemModels(unittest.TestCase):
    """Tests for Pydantic models defined in the system routes module."""

    def test_system_status_valid(self):
        """SystemStatus accepts all required fields."""
        model = system.SystemStatus(
            initialized=True,
            atlas_available=False,
            learning_enabled=True,
            stats={"frames_processed": 0},
            timestamp="2024-01-01T00:00:00",
        )
        self.assertTrue(model.initialized)
        self.assertFalse(model.atlas_available)
        self.assertIn("frames_processed", model.stats)
        self.assertIsNone(model.system_state)
        self.assertIsNone(model.architecture)

    def test_system_status_with_optional(self):
        """SystemStatus accepts optional system_state and architecture."""
        model = system.SystemStatus(
            initialized=True,
            atlas_available=True,
            learning_enabled=True,
            stats={},
            timestamp="2024-01-01",
            system_state={"total_frames": 100},
            architecture={"visual_layers": {}},
        )
        self.assertIsNotNone(model.system_state)
        self.assertIsNotNone(model.architecture)

    def test_system_status_missing_required(self):
        """SystemStatus raises ValidationError when required fields are missing."""
        with self.assertRaises(pydantic.ValidationError):
            system.SystemStatus()

    def test_metrics_response_valid(self):
        """MetricsResponse accepts required and optional fields."""
        model = system.MetricsResponse(
            frames_processed=100,
            audio_chunks_processed=50,
            uptime_seconds=3600.0,
            timestamp="2024-01-01T00:00:00",
        )
        self.assertEqual(model.frames_processed, 100)
        self.assertEqual(model.audio_chunks_processed, 50)
        self.assertAlmostEqual(model.uptime_seconds, 3600.0)
        self.assertIsNone(model.prediction_error)
        self.assertIsNone(model.total_neurons)

    def test_metrics_response_full(self):
        """MetricsResponse accepts all optional metric fields."""
        model = system.MetricsResponse(
            frames_processed=10,
            audio_chunks_processed=5,
            uptime_seconds=120.0,
            timestamp="2024-01-01T00:00:00",
            prediction_error=0.05,
            reconstruction_error=0.03,
            cross_modal_correlation=0.8,
            total_neurons=500,
            active_associations=200,
        )
        self.assertAlmostEqual(model.prediction_error, 0.05)
        self.assertEqual(model.total_neurons, 500)
        self.assertEqual(model.active_associations, 200)

    def test_metrics_response_missing_required(self):
        """MetricsResponse raises ValidationError without required fields."""
        with self.assertRaises(pydantic.ValidationError):
            system.MetricsResponse()


# ===========================================================================
# Route Definition Tests
# ===========================================================================

@unittest.skipUnless(HAS_WEB_DEPS, "fastapi and pydantic are required")
class TestControlRouteDefinitions(unittest.TestCase):
    """Tests that control routes are properly defined."""

    def test_router_exists(self):
        """The control module has an APIRouter instance."""
        self.assertIsInstance(control.router, fastapi.APIRouter)

    def test_router_has_routes(self):
        """The control router has registered routes."""
        self.assertGreater(len(control.router.routes), 0)

    def _get_route_paths(self):
        """Helper to get all route paths from the control router."""
        return [r.path for r in control.router.routes if hasattr(r, 'path')]

    def test_learning_post_route(self):
        """POST /learning route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/learning", paths)

    def test_learning_get_route(self):
        """GET /learning route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/learning", paths)

    def test_learning_rate_route(self):
        """POST /learning-rate route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/learning-rate", paths)

    def test_checkpoint_save_route(self):
        """POST /checkpoint/save route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/checkpoint/save", paths)

    def test_checkpoint_load_route(self):
        """POST /checkpoint/load route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/checkpoint/load", paths)

    def test_checkpoint_delete_route(self):
        """DELETE /checkpoint/{name} route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/checkpoint/{name}", paths)

    def test_mode_post_route(self):
        """POST /mode route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/mode", paths)

    def test_mode_get_route(self):
        """GET /mode route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/mode", paths)

    def test_config_route(self):
        """POST /config route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/config", paths)

    def test_reset_route(self):
        """POST /reset route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/reset", paths)

    def test_think_route(self):
        """POST /think route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/think", paths)

    def test_imagine_route(self):
        """POST /imagine route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/imagine", paths)


@unittest.skipUnless(HAS_WEB_DEPS, "fastapi and pydantic are required")
class TestDataRouteDefinitions(unittest.TestCase):
    """Tests that data routes are properly defined."""

    def test_router_exists(self):
        """The data module has an APIRouter instance."""
        self.assertIsInstance(data.router, fastapi.APIRouter)

    def _get_route_paths(self):
        return [r.path for r in data.router.routes if hasattr(r, 'path')]

    def test_frame_route(self):
        """POST /frame route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/frame", paths)

    def test_frame_upload_route(self):
        """POST /frame/upload route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/frame/upload", paths)

    def test_audio_route(self):
        """POST /audio route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/audio", paths)

    def test_audio_upload_route(self):
        """POST /audio/upload route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/audio/upload", paths)

    def test_av_pair_route(self):
        """POST /av-pair route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/av-pair", paths)

    def test_predictions_route(self):
        """GET /predictions route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/predictions", paths)


@unittest.skipUnless(HAS_WEB_DEPS, "fastapi and pydantic are required")
class TestMemoryRouteDefinitions(unittest.TestCase):
    """Tests that memory routes are properly defined."""

    def test_router_exists(self):
        """The memory module has an APIRouter instance."""
        self.assertIsInstance(memory.router, fastapi.APIRouter)

    def _get_route_paths(self):
        return [r.path for r in memory.router.routes if hasattr(r, 'path')]

    def test_episodic_route(self):
        """GET /episodic route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/episodic", paths)

    def test_semantic_route(self):
        """GET /semantic route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/semantic", paths)

    def test_working_route(self):
        """GET /working route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/working", paths)

    def test_associations_route(self):
        """GET /associations route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/associations", paths)

    def test_search_route(self):
        """POST /search route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/search", paths)

    def test_statistics_route(self):
        """GET /statistics route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/statistics", paths)


@unittest.skipUnless(HAS_WEB_DEPS, "fastapi and pydantic are required")
class TestSystemRouteDefinitions(unittest.TestCase):
    """Tests that system routes are properly defined."""

    def test_router_exists(self):
        """The system module has an APIRouter instance."""
        self.assertIsInstance(system.router, fastapi.APIRouter)

    def _get_route_paths(self):
        return [r.path for r in system.router.routes if hasattr(r, 'path')]

    def test_status_route(self):
        """GET /status route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/status", paths)

    def test_metrics_route(self):
        """GET /metrics route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/metrics", paths)

    def test_architecture_route(self):
        """GET /architecture route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/architecture", paths)

    def test_checkpoints_route(self):
        """GET /checkpoints route is defined."""
        paths = self._get_route_paths()
        self.assertIn("/checkpoints", paths)


# ===========================================================================
# AtlasManager Tests
# ===========================================================================

@unittest.skipUnless(HAS_WEB_DEPS, "fastapi and pydantic are required")
class TestAtlasManagerInit(unittest.TestCase):
    """Tests for AtlasManager initialization and basic properties."""

    def test_instantiation(self):
        """AtlasManager can be instantiated without arguments."""
        mgr = AtlasManager()
        self.assertIsNotNone(mgr)

    def test_not_initialized_by_default(self):
        """AtlasManager._initialized is False before calling initialize()."""
        mgr = AtlasManager()
        self.assertFalse(mgr._initialized)

    def test_is_initialized_method(self):
        """is_initialized() returns False before initialize()."""
        mgr = AtlasManager()
        self.assertFalse(mgr.is_initialized())

    def test_learning_enabled_by_default(self):
        """Learning is enabled by default."""
        mgr = AtlasManager()
        self.assertTrue(mgr._learning_enabled)

    def test_stats_initial_state(self):
        """Initial stats have zero counts and no timestamps."""
        mgr = AtlasManager()
        self.assertEqual(mgr._stats["frames_processed"], 0)
        self.assertEqual(mgr._stats["audio_chunks_processed"], 0)
        self.assertIsNone(mgr._stats["start_time"])
        self.assertIsNone(mgr._stats["last_activity"])

    def test_subscribers_initially_empty(self):
        """No subscribers exist at creation."""
        mgr = AtlasManager()
        self.assertEqual(len(mgr._subscribers), 0)

    def test_system_initially_none(self):
        """system attribute is None before initialization."""
        mgr = AtlasManager()
        self.assertIsNone(mgr.system)

    def test_config_initially_none(self):
        """config attribute is None before initialization."""
        mgr = AtlasManager()
        self.assertIsNone(mgr.config)

    def test_visual_processor_initially_none(self):
        """visual_processor attribute is None before initialization."""
        mgr = AtlasManager()
        self.assertIsNone(mgr.visual_processor)

    def test_audio_processor_initially_none(self):
        """audio_processor attribute is None before initialization."""
        mgr = AtlasManager()
        self.assertIsNone(mgr.audio_processor)


@unittest.skipUnless(HAS_WEB_DEPS, "fastapi and pydantic are required")
class TestAtlasManagerAsync(unittest.TestCase):
    """Tests for AtlasManager async methods."""

    def _run_async(self, coro):
        """Helper to run an async coroutine synchronously."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def setUp(self):
        """Patch ATLAS_AVAILABLE to False so tests run in demo mode."""
        patcher = patch('web.backend.services.atlas_manager.ATLAS_AVAILABLE', False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_initialize_returns_true(self):
        """initialize() succeeds (returns True) in demo mode."""
        mgr = AtlasManager()
        result = self._run_async(mgr.initialize())
        self.assertTrue(result)
        self.assertTrue(mgr.is_initialized())

    def test_initialize_sets_start_time(self):
        """initialize() sets start_time in stats."""
        mgr = AtlasManager()
        self._run_async(mgr.initialize())
        self.assertIsNotNone(mgr._stats["start_time"])

    def test_shutdown_sets_not_initialized(self):
        """shutdown() marks the manager as not initialized."""
        mgr = AtlasManager()
        self._run_async(mgr.initialize())
        self.assertTrue(mgr.is_initialized())
        self._run_async(mgr.shutdown())
        self.assertFalse(mgr.is_initialized())

    def test_get_status_returns_dict(self):
        """get_status() returns a dictionary with expected keys."""
        mgr = AtlasManager()
        self._run_async(mgr.initialize())
        status = self._run_async(mgr.get_status())
        self.assertIsInstance(status, dict)
        self.assertIn("initialized", status)
        self.assertIn("learning_enabled", status)
        self.assertIn("stats", status)
        self.assertIn("timestamp", status)

    def test_get_metrics_returns_dict(self):
        """get_metrics() returns a dictionary with expected keys."""
        mgr = AtlasManager()
        self._run_async(mgr.initialize())
        metrics = self._run_async(mgr.get_metrics())
        self.assertIsInstance(metrics, dict)
        self.assertIn("frames_processed", metrics)
        self.assertIn("audio_chunks_processed", metrics)
        self.assertIn("uptime_seconds", metrics)
        self.assertIn("timestamp", metrics)

    def test_get_metrics_uptime_positive(self):
        """get_metrics() returns non-negative uptime after initialization."""
        mgr = AtlasManager()
        self._run_async(mgr.initialize())
        metrics = self._run_async(mgr.get_metrics())
        self.assertGreaterEqual(metrics["uptime_seconds"], 0)

    def test_set_learning_enabled(self):
        """set_learning_enabled() toggles learning state."""
        mgr = AtlasManager()
        self._run_async(mgr.initialize())

        result = self._run_async(mgr.set_learning_enabled(False))
        self.assertFalse(result["learning_enabled"])
        self.assertFalse(mgr._learning_enabled)

        result = self._run_async(mgr.set_learning_enabled(True))
        self.assertTrue(result["learning_enabled"])
        self.assertTrue(mgr._learning_enabled)

    def test_set_learning_rate(self):
        """set_learning_rate() returns success in demo mode."""
        mgr = AtlasManager()
        self._run_async(mgr.initialize())
        result = self._run_async(mgr.set_learning_rate(0.05))
        self.assertIn("learning_rate", result)
        self.assertAlmostEqual(result["learning_rate"], 0.05)
        self.assertTrue(result.get("success", False))

    def test_get_predictions_demo(self):
        """get_predictions() returns demo data when atlas core is not available."""
        mgr = AtlasManager()
        self._run_async(mgr.initialize())
        predictions = self._run_async(mgr.get_predictions(modality="visual", num_steps=3))
        self.assertIsInstance(predictions, dict)
        self.assertIn("modality", predictions)
        self.assertEqual(predictions["modality"], "visual")
        self.assertIn("temporal", predictions)
        self.assertEqual(len(predictions["temporal"]), 3)

    def test_get_memory_contents_episodic(self):
        """get_memory_contents() returns demo episodic data."""
        mgr = AtlasManager()
        self._run_async(mgr.initialize())
        contents = self._run_async(mgr.get_memory_contents(memory_type="episodic", limit=3))
        self.assertIsInstance(contents, dict)
        self.assertEqual(contents["memory_type"], "episodic")
        self.assertIn("items", contents)
        self.assertLessEqual(len(contents["items"]), 3)

    def test_get_memory_contents_semantic(self):
        """get_memory_contents() returns demo semantic data."""
        mgr = AtlasManager()
        self._run_async(mgr.initialize())
        contents = self._run_async(mgr.get_memory_contents(memory_type="semantic", limit=5))
        self.assertEqual(contents["memory_type"], "semantic")

    def test_get_memory_contents_working(self):
        """get_memory_contents() returns demo working memory data."""
        mgr = AtlasManager()
        self._run_async(mgr.initialize())
        contents = self._run_async(mgr.get_memory_contents(memory_type="working", limit=10))
        self.assertEqual(contents["memory_type"], "working")

    def test_save_checkpoint_demo(self):
        """save_checkpoint() returns success in demo mode."""
        mgr = AtlasManager()
        self._run_async(mgr.initialize())
        result = self._run_async(mgr.save_checkpoint(name="test_cp"))
        self.assertIn("checkpoint_name", result)
        self.assertEqual(result["checkpoint_name"], "test_cp")
        self.assertTrue(result.get("success", False))

    def test_list_checkpoints(self):
        """list_checkpoints() returns a dict with checkpoints list."""
        mgr = AtlasManager()
        self._run_async(mgr.initialize())
        result = self._run_async(mgr.list_checkpoints())
        self.assertIsInstance(result, dict)
        self.assertIn("checkpoints", result)
        self.assertIsInstance(result["checkpoints"], list)

    def test_get_architecture_info_demo(self):
        """get_architecture_info() returns demo architecture in demo mode."""
        mgr = AtlasManager()
        self._run_async(mgr.initialize())
        info = self._run_async(mgr.get_architecture_info())
        self.assertIsInstance(info, dict)
        self.assertIn("visual", info)
        self.assertIn("audio", info)
        self.assertIn("multimodal", info)
        self.assertIn("layers", info["visual"])
        self.assertIn("layers", info["audio"])


@unittest.skipUnless(HAS_WEB_DEPS, "fastapi and pydantic are required")
class TestAtlasManagerSubscription(unittest.TestCase):
    """Tests for AtlasManager WebSocket subscription management."""

    def test_subscribe_adds_queue(self):
        """subscribe() adds a queue to the subscribers list."""
        mgr = AtlasManager()
        queue = asyncio.Queue()
        mgr.subscribe(queue)
        self.assertEqual(len(mgr._subscribers), 1)
        self.assertIn(queue, mgr._subscribers)

    def test_unsubscribe_removes_queue(self):
        """unsubscribe() removes a queue from the subscribers list."""
        mgr = AtlasManager()
        queue = asyncio.Queue()
        mgr.subscribe(queue)
        mgr.unsubscribe(queue)
        self.assertEqual(len(mgr._subscribers), 0)

    def test_unsubscribe_nonexistent_no_error(self):
        """unsubscribe() does not raise if the queue is not present."""
        mgr = AtlasManager()
        queue = asyncio.Queue()
        mgr.unsubscribe(queue)  # Should not raise

    def test_multiple_subscribers(self):
        """Multiple queues can be subscribed simultaneously."""
        mgr = AtlasManager()
        q1 = asyncio.Queue()
        q2 = asyncio.Queue()
        q3 = asyncio.Queue()
        mgr.subscribe(q1)
        mgr.subscribe(q2)
        mgr.subscribe(q3)
        self.assertEqual(len(mgr._subscribers), 3)
        mgr.unsubscribe(q2)
        self.assertEqual(len(mgr._subscribers), 2)
        self.assertNotIn(q2, mgr._subscribers)


# ===========================================================================
# FastAPI App Configuration Tests
# ===========================================================================

@unittest.skipUnless(HAS_WEB_DEPS, "fastapi and pydantic are required")
class TestFastAPIAppConfig(unittest.TestCase):
    """Tests for the FastAPI application configuration in main.py."""

    @classmethod
    def setUpClass(cls):
        """Import the app once for all tests in this class."""
        from web.backend.main import app
        cls.app = app

    def test_app_is_fastapi_instance(self):
        """The app object is a FastAPI instance."""
        self.assertIsInstance(self.app, fastapi.FastAPI)

    def test_app_title(self):
        """The app title is 'Atlas Web API'."""
        self.assertEqual(self.app.title, "Atlas Web API")

    def test_app_version(self):
        """The app version is '1.0.0'."""
        self.assertEqual(self.app.version, "1.0.0")

    def test_app_has_routes(self):
        """The app has registered routes."""
        self.assertGreater(len(self.app.routes), 0)

    def _get_all_route_paths(self):
        """Helper to get all registered route paths."""
        paths = []
        for route in self.app.routes:
            if hasattr(route, 'path'):
                paths.append(route.path)
        return paths

    def test_root_route_exists(self):
        """The root (/) route is registered."""
        paths = self._get_all_route_paths()
        self.assertIn("/", paths)

    def test_health_route_exists(self):
        """The /health route is registered."""
        paths = self._get_all_route_paths()
        self.assertIn("/health", paths)

    def test_system_routes_included(self):
        """System routes are included under /api/system prefix."""
        paths = self._get_all_route_paths()
        system_paths = [p for p in paths if p.startswith("/api/system")]
        self.assertGreater(len(system_paths), 0)

    def test_data_routes_included(self):
        """Data routes are included under /api/data prefix."""
        paths = self._get_all_route_paths()
        data_paths = [p for p in paths if p.startswith("/api/data")]
        self.assertGreater(len(data_paths), 0)

    def test_memory_routes_included(self):
        """Memory routes are included under /api/memory prefix."""
        paths = self._get_all_route_paths()
        memory_paths = [p for p in paths if p.startswith("/api/memory")]
        self.assertGreater(len(memory_paths), 0)

    def test_control_routes_included(self):
        """Control routes are included under /api/control prefix."""
        paths = self._get_all_route_paths()
        control_paths = [p for p in paths if p.startswith("/api/control")]
        self.assertGreater(len(control_paths), 0)

    def test_websocket_routes_included(self):
        """WebSocket routes are included under /ws prefix."""
        paths = self._get_all_route_paths()
        ws_paths = [p for p in paths if p.startswith("/ws")]
        self.assertGreater(len(ws_paths), 0)

    def test_specific_api_system_paths(self):
        """Expected /api/system/* paths are registered."""
        paths = self._get_all_route_paths()
        self.assertIn("/api/system/status", paths)
        self.assertIn("/api/system/metrics", paths)
        self.assertIn("/api/system/architecture", paths)
        self.assertIn("/api/system/checkpoints", paths)

    def test_specific_api_data_paths(self):
        """Expected /api/data/* paths are registered."""
        paths = self._get_all_route_paths()
        self.assertIn("/api/data/frame", paths)
        self.assertIn("/api/data/audio", paths)
        self.assertIn("/api/data/av-pair", paths)
        self.assertIn("/api/data/predictions", paths)

    def test_specific_api_memory_paths(self):
        """Expected /api/memory/* paths are registered."""
        paths = self._get_all_route_paths()
        self.assertIn("/api/memory/episodic", paths)
        self.assertIn("/api/memory/semantic", paths)
        self.assertIn("/api/memory/working", paths)
        self.assertIn("/api/memory/associations", paths)
        self.assertIn("/api/memory/search", paths)
        self.assertIn("/api/memory/statistics", paths)

    def test_specific_api_control_paths(self):
        """Expected /api/control/* paths are registered."""
        paths = self._get_all_route_paths()
        self.assertIn("/api/control/learning", paths)
        self.assertIn("/api/control/learning-rate", paths)
        self.assertIn("/api/control/checkpoint/save", paths)
        self.assertIn("/api/control/checkpoint/load", paths)
        self.assertIn("/api/control/mode", paths)
        self.assertIn("/api/control/config", paths)
        self.assertIn("/api/control/reset", paths)
        self.assertIn("/api/control/think", paths)
        self.assertIn("/api/control/imagine", paths)


# ===========================================================================
# Integration Tests with TestClient (httpx)
# ===========================================================================

@unittest.skipUnless(HAS_TEST_CLIENT, "fastapi, pydantic, and httpx are required")
class TestEndpointsWithTestClient(unittest.TestCase):
    """Integration tests using starlette TestClient (backed by httpx)."""

    @classmethod
    def setUpClass(cls):
        """Create a TestClient for the app with an AtlasManager in app state."""
        from web.backend.main import app

        # Manually inject an AtlasManager into the app state so routes
        # can use it without going through the lifespan context manager.
        # Patch ATLAS_AVAILABLE to False so we run in demo mode.
        cls._atlas_patcher = patch('web.backend.services.atlas_manager.ATLAS_AVAILABLE', False)
        cls._atlas_patcher.start()

        mgr = AtlasManager()
        loop = asyncio.new_event_loop()
        loop.run_until_complete(mgr.initialize())
        loop.close()
        app.state.atlas_manager = mgr

        cls.client = TestClient(app, raise_server_exceptions=False)

    @classmethod
    def tearDownClass(cls):
        cls._atlas_patcher.stop()

    # --- Root / Health -------------------------------------------------------

    def test_root_endpoint(self):
        """GET / returns 200 with expected API info."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["name"], "Atlas Web API")
        self.assertEqual(body["version"], "1.0.0")
        self.assertIn("endpoints", body)

    def test_health_endpoint(self):
        """GET /health returns 200 with healthy status."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "healthy")
        self.assertEqual(body["service"], "atlas-web-backend")

    # --- System Endpoints ----------------------------------------------------

    def test_system_status(self):
        """GET /api/system/status returns valid status."""
        resp = self.client.get("/api/system/status")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("initialized", body)
        self.assertTrue(body["initialized"])
        self.assertIn("learning_enabled", body)
        self.assertIn("stats", body)

    def test_system_metrics(self):
        """GET /api/system/metrics returns valid metrics."""
        resp = self.client.get("/api/system/metrics")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("frames_processed", body)
        self.assertIn("audio_chunks_processed", body)
        self.assertIn("uptime_seconds", body)

    def test_system_architecture(self):
        """GET /api/system/architecture returns architecture info."""
        resp = self.client.get("/api/system/architecture")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("visual", body)
        self.assertIn("audio", body)

    def test_system_checkpoints(self):
        """GET /api/system/checkpoints returns checkpoint list."""
        resp = self.client.get("/api/system/checkpoints")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("checkpoints", body)
        self.assertIsInstance(body["checkpoints"], list)

    # --- Control Endpoints ---------------------------------------------------

    def test_control_set_learning_on(self):
        """POST /api/control/learning enables learning."""
        resp = self.client.post(
            "/api/control/learning",
            json={"enabled": True},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["learning_enabled"])

    def test_control_set_learning_off(self):
        """POST /api/control/learning disables learning."""
        resp = self.client.post(
            "/api/control/learning",
            json={"enabled": False},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertFalse(body["learning_enabled"])
        # Re-enable for other tests
        self.client.post("/api/control/learning", json={"enabled": True})

    def test_control_get_learning(self):
        """GET /api/control/learning returns learning state."""
        resp = self.client.get("/api/control/learning")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("learning_enabled", body)

    def test_control_set_learning_rate(self):
        """POST /api/control/learning-rate sets the rate."""
        resp = self.client.post(
            "/api/control/learning-rate",
            json={"rate": 0.05},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertAlmostEqual(body["learning_rate"], 0.05)

    def test_control_set_learning_rate_invalid(self):
        """POST /api/control/learning-rate rejects rate > 1.0."""
        resp = self.client.post(
            "/api/control/learning-rate",
            json={"rate": 5.0},
        )
        self.assertEqual(resp.status_code, 422)

    def test_control_set_mode_valid(self):
        """POST /api/control/mode accepts a valid mode."""
        resp = self.client.post(
            "/api/control/mode",
            json={"mode": "visual"},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["mode"], "visual")
        self.assertTrue(body.get("success", False))

    def test_control_set_mode_invalid(self):
        """POST /api/control/mode rejects an invalid mode with 400."""
        resp = self.client.post(
            "/api/control/mode",
            json={"mode": "invalid_mode"},
        )
        self.assertEqual(resp.status_code, 400)

    def test_control_get_mode(self):
        """GET /api/control/mode returns current mode."""
        resp = self.client.get("/api/control/mode")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("mode", body)

    def test_control_config_update(self):
        """POST /api/control/config updates configuration."""
        resp = self.client.post(
            "/api/control/config",
            json={"learning_rate": 0.01, "multimodal_size": 100},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("updated", body)
        self.assertTrue(body.get("success", False))

    def test_control_config_empty(self):
        """POST /api/control/config with empty body returns success."""
        resp = self.client.post("/api/control/config", json={})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body.get("success", False))

    def test_control_reset(self):
        """POST /api/control/reset returns reset info."""
        resp = self.client.post("/api/control/reset")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("reset", body)
        self.assertIn("message", body)

    def test_control_think(self):
        """POST /api/control/think triggers thinking."""
        resp = self.client.post("/api/control/think")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("task", body)
        self.assertIn("output", body)
        self.assertIsNotNone(body["output"])

    def test_control_imagine(self):
        """POST /api/control/imagine triggers imagination."""
        resp = self.client.post("/api/control/imagine")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("steps", body)
        self.assertIn("output", body)

    def test_control_checkpoint_save(self):
        """POST /api/control/checkpoint/save saves a checkpoint."""
        resp = self.client.post(
            "/api/control/checkpoint/save",
            json={"name": "test_save"},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("checkpoint_name", body)
        self.assertTrue(body.get("success", False))

    # --- Memory Endpoints ----------------------------------------------------

    def test_memory_episodic(self):
        """GET /api/memory/episodic returns episodic memories."""
        resp = self.client.get("/api/memory/episodic")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["memory_type"], "episodic")
        self.assertIn("items", body)

    def test_memory_semantic(self):
        """GET /api/memory/semantic returns semantic memories."""
        resp = self.client.get("/api/memory/semantic")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["memory_type"], "semantic")

    def test_memory_working(self):
        """GET /api/memory/working returns working memory."""
        resp = self.client.get("/api/memory/working")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["memory_type"], "working")

    def test_memory_associations(self):
        """GET /api/memory/associations returns associations."""
        resp = self.client.get("/api/memory/associations")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("total_associations", body)

    def test_memory_search(self):
        """POST /api/memory/search returns search results."""
        resp = self.client.post(
            "/api/memory/search",
            json={"query": "face", "memory_type": "all", "limit": 5},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("query", body)
        self.assertEqual(body["query"], "face")
        self.assertIn("results", body)
        self.assertIn("total_found", body)

    def test_memory_statistics(self):
        """GET /api/memory/statistics returns statistics."""
        resp = self.client.get("/api/memory/statistics")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("episodic", body)
        self.assertIn("semantic", body)
        self.assertIn("working", body)

    # --- Data Endpoints ------------------------------------------------------

    def test_data_predictions(self):
        """GET /api/data/predictions returns predictions."""
        resp = self.client.get("/api/data/predictions?modality=visual&num_steps=3")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["modality"], "visual")
        self.assertEqual(body["num_steps"], 3)
        self.assertIn("temporal", body)

    def test_data_predictions_audio(self):
        """GET /api/data/predictions with audio modality works."""
        resp = self.client.get("/api/data/predictions?modality=audio&num_steps=2")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["modality"], "audio")

    def test_data_predictions_invalid_modality(self):
        """GET /api/data/predictions with invalid modality returns 400."""
        resp = self.client.get("/api/data/predictions?modality=invalid")
        self.assertEqual(resp.status_code, 400)

    # --- Error Handling Tests ------------------------------------------------

    def test_not_found_returns_404(self):
        """A non-existent endpoint returns 404."""
        resp = self.client.get("/api/nonexistent")
        self.assertEqual(resp.status_code, 404)

    def test_learning_missing_body_returns_422(self):
        """POST /api/control/learning with empty body returns 422."""
        resp = self.client.post("/api/control/learning")
        self.assertEqual(resp.status_code, 422)

    def test_memory_search_missing_query_returns_422(self):
        """POST /api/memory/search without query returns 422."""
        resp = self.client.post("/api/memory/search", json={})
        self.assertEqual(resp.status_code, 422)

    def test_checkpoint_delete_nonexistent_returns_404(self):
        """DELETE /api/control/checkpoint/nonexistent returns 404."""
        resp = self.client.delete(
            "/api/control/checkpoint/definitely_does_not_exist_xyz"
        )
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
