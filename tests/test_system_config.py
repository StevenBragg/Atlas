"""
Comprehensive tests for SystemConfig classes from both
self_organizing_av_system/core/system_config.py and
self_organizing_av_system/config/configuration.py.

Tests cover initialization with defaults, visual settings, audio settings,
capture settings, monitor/checkpoint settings, YAML file support,
configuration validation, deep update, dot-path set, save/load round-trip,
and dictionary-style access.
"""
import sys
import os
import unittest
import tempfile
import shutil
import logging
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.system_config import SystemConfig as CoreSystemConfig
from self_organizing_av_system.config.configuration import SystemConfig as ConfigSystemConfig


# ---------------------------------------------------------------------------
# Helper: build a CoreSystemConfig that uses _set_minimal_defaults() rather
# than the package's default_config.yaml.  We achieve this by first
# constructing normally (which loads the YAML) and then replacing the config
# with the minimal defaults and re-validating.
# ---------------------------------------------------------------------------

def _core_config_with_minimal_defaults():
    """Return a CoreSystemConfig whose config comes from _set_minimal_defaults."""
    cfg = CoreSystemConfig()
    cfg._set_minimal_defaults()
    cfg._validate_config()
    return cfg


# ---------------------------------------------------------------------------
# Tests for self_organizing_av_system.core.system_config.SystemConfig
# ---------------------------------------------------------------------------

class TestCoreSystemConfigInitialization(unittest.TestCase):
    """Test CoreSystemConfig construction and default state.

    The default constructor discovers and loads the package-level
    config/default_config.yaml.  Values tested here reflect that file.
    """

    def setUp(self):
        self.cfg = CoreSystemConfig()

    def test_config_is_dict(self):
        self.assertIsInstance(self.cfg.get_config(), dict)

    def test_system_section_present(self):
        self.assertIn("system", self.cfg.get_config())

    def test_system_defaults_from_yaml(self):
        sys_cfg = self.cfg.get_system_config()
        self.assertEqual(sys_cfg["multimodal_size"], 10)
        self.assertAlmostEqual(sys_cfg["learning_rate"], 0.01)
        self.assertEqual(sys_cfg["learning_rule"], "oja")
        self.assertEqual(sys_cfg["prune_interval"], 1000)
        self.assertEqual(sys_cfg["structural_plasticity_interval"], 2000)

    def test_validation_adds_required_sections(self):
        """After loading the YAML, _validate_config adds missing sections."""
        config = self.cfg.get_config()
        required = [
            "system", "visual_processor", "audio_processor",
            "multimodal_association", "temporal_prediction",
            "stability", "structural_plasticity",
            "av_capture", "monitoring",
        ]
        for section in required:
            self.assertIn(section, config, f"Missing section: {section}")


class TestCoreVisualSettings(unittest.TestCase):
    """Test visual settings loaded from the package default_config.yaml.

    The YAML file uses the section name 'visual' (not 'visual_processor').
    """

    def setUp(self):
        self.cfg = CoreSystemConfig()
        self.visual = self.cfg.get_component_config("visual")

    def test_resolution(self):
        self.assertEqual(self.visual["input_width"], 24)
        self.assertEqual(self.visual["input_height"], 24)

    def test_grayscale_enabled(self):
        self.assertTrue(self.visual["use_grayscale"])

    def test_patch_extraction(self):
        self.assertEqual(self.visual["patch_size"], 12)
        self.assertEqual(self.visual["stride"], 12)

    def test_contrast_normalization(self):
        self.assertTrue(self.visual["contrast_normalize"])

    def test_layer_sizes(self):
        self.assertEqual(self.visual["layer_sizes"], [12, 8, 4])


class TestCoreAudioSettings(unittest.TestCase):
    """Test audio settings loaded from the package default_config.yaml."""

    def setUp(self):
        self.cfg = CoreSystemConfig()
        self.audio = self.cfg.get_component_config("audio")

    def test_sample_rate(self):
        self.assertEqual(self.audio["sample_rate"], 22050)

    def test_mel_spectrogram_settings(self):
        self.assertEqual(self.audio["window_size"], 128)
        self.assertEqual(self.audio["hop_length"], 96)
        self.assertEqual(self.audio["n_mels"], 8)

    def test_frequency_range(self):
        self.assertEqual(self.audio["min_freq"], 50)
        self.assertEqual(self.audio["max_freq"], 8000)

    def test_normalization_enabled(self):
        self.assertTrue(self.audio["normalize"])

    def test_layer_sizes(self):
        self.assertEqual(self.audio["layer_sizes"], [10, 6, 4])


class TestCoreCaptureSettings(unittest.TestCase):
    """Test capture settings loaded from the package default_config.yaml."""

    def setUp(self):
        self.cfg = CoreSystemConfig()
        self.capture = self.cfg.get_component_config("capture")

    def test_webcam_resolution(self):
        self.assertEqual(self.capture["video_width"], 120)
        self.assertEqual(self.capture["video_height"], 90)

    def test_fps(self):
        self.assertEqual(self.capture["fps"], 30)

    def test_audio_channels(self):
        self.assertEqual(self.capture["audio_channels"], 1)

    def test_chunk_size(self):
        self.assertEqual(self.capture["chunk_size"], 128)


class TestCoreMonitorSettings(unittest.TestCase):
    """Test monitor settings loaded from the package default_config.yaml."""

    def setUp(self):
        self.cfg = CoreSystemConfig()
        self.monitor = self.cfg.get_component_config("monitor")

    def test_update_interval(self):
        self.assertAlmostEqual(self.monitor["update_interval"], 0.033)

    def test_save_snapshots_disabled(self):
        self.assertFalse(self.monitor["save_snapshots"])

    def test_snapshot_interval(self):
        self.assertEqual(self.monitor["snapshot_interval"], 1000)

    def test_snapshot_path(self):
        self.assertEqual(self.monitor["snapshot_path"], "snapshots")


class TestCoreCheckpointSettings(unittest.TestCase):
    """Test checkpointing settings loaded from the package default_config.yaml."""

    def setUp(self):
        self.cfg = CoreSystemConfig()
        self.checkpoint = self.cfg.get_component_config("checkpointing")

    def test_enabled(self):
        self.assertTrue(self.checkpoint["enabled"])

    def test_checkpoint_interval(self):
        self.assertEqual(self.checkpoint["checkpoint_interval"], 1000)

    def test_checkpoint_dir(self):
        self.assertEqual(self.checkpoint["checkpoint_dir"], "checkpoints")

    def test_max_checkpoints(self):
        self.assertEqual(self.checkpoint["max_checkpoints"], 3)

    def test_load_latest(self):
        self.assertTrue(self.checkpoint["load_latest"])

    def test_save_on_exit(self):
        self.assertTrue(self.checkpoint["save_on_exit"])


# ---------------------------------------------------------------------------
# Tests for CoreSystemConfig._set_minimal_defaults
# ---------------------------------------------------------------------------

class TestCoreMinimalDefaults(unittest.TestCase):
    """Test the hard-coded minimal defaults fallback."""

    def setUp(self):
        self.cfg = _core_config_with_minimal_defaults()

    def test_system_defaults(self):
        sys_cfg = self.cfg.get_system_config()
        self.assertEqual(sys_cfg["multimodal_size"], 128)
        self.assertAlmostEqual(sys_cfg["learning_rate"], 0.01)
        self.assertEqual(sys_cfg["prune_interval"], 100)
        self.assertEqual(sys_cfg["structural_plasticity_interval"], 200)
        self.assertEqual(sys_cfg["snapshot_interval"], 1000)
        self.assertTrue(sys_cfg["enable_learning"])
        self.assertTrue(sys_cfg["enable_visualization"])

    def test_log_level_converted_to_int(self):
        """After validation the string log level should become an int."""
        sys_cfg = self.cfg.get_system_config()
        self.assertEqual(sys_cfg["log_level"], logging.INFO)

    def test_visual_processor_resolution(self):
        vp = self.cfg.get_component_config("visual_processor")
        self.assertEqual(vp["input_width"], 128)
        self.assertEqual(vp["input_height"], 128)

    def test_visual_processor_grayscale(self):
        vp = self.cfg.get_component_config("visual_processor")
        self.assertTrue(vp["use_grayscale"])

    def test_visual_processor_sparse_coding(self):
        vp = self.cfg.get_component_config("visual_processor")
        self.assertTrue(vp["use_sparse_coding"])

    def test_audio_processor_sample_rate(self):
        ap = self.cfg.get_component_config("audio_processor")
        self.assertEqual(ap["sample_rate"], 16000)

    def test_audio_processor_mel_spectrogram(self):
        ap = self.cfg.get_component_config("audio_processor")
        self.assertEqual(ap["window_size"], 1024)
        self.assertEqual(ap["hop_length"], 512)
        self.assertEqual(ap["n_mels"], 64)

    def test_audio_processor_sparse_coding(self):
        ap = self.cfg.get_component_config("audio_processor")
        self.assertTrue(ap["use_sparse_coding"])

    def test_av_capture_webcam(self):
        cap = self.cfg.get_component_config("av_capture")
        self.assertEqual(cap["video_width"], 640)
        self.assertEqual(cap["video_height"], 480)

    def test_av_capture_fps(self):
        cap = self.cfg.get_component_config("av_capture")
        self.assertEqual(cap["fps"], 30)

    def test_av_capture_device_ids(self):
        cap = self.cfg.get_component_config("av_capture")
        self.assertEqual(cap["video_device_id"], 0)
        self.assertEqual(cap["audio_device_id"], 0)

    def test_monitoring_defaults(self):
        mon = self.cfg.get_component_config("monitoring")
        self.assertAlmostEqual(mon["update_interval"], 1.0)
        self.assertTrue(mon["save_snapshots"])
        self.assertEqual(mon["snapshot_dir"], "./snapshots")
        self.assertEqual(mon["log_stats_interval"], 10)


# ---------------------------------------------------------------------------
# Core: YAML file support
# ---------------------------------------------------------------------------

class TestCoreYAMLFileSupport(unittest.TestCase):
    """Test YAML load/save round-trip for CoreSystemConfig."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_reload(self):
        cfg = CoreSystemConfig()
        save_path = os.path.join(self.tmpdir, "saved_config.yaml")
        cfg.save_config(save_path)
        self.assertTrue(os.path.isfile(save_path))

        # Reload using load_from_file
        cfg2 = CoreSystemConfig.load_from_file(save_path)
        # The reloaded config should contain the same top-level sections
        for section in cfg.get_config():
            self.assertIn(section, cfg2.get_config())

    def test_saved_yaml_is_valid(self):
        cfg = CoreSystemConfig()
        save_path = os.path.join(self.tmpdir, "saved.yaml")
        cfg.save_config(save_path)
        with open(save_path, 'r') as f:
            loaded = yaml.safe_load(f)
        self.assertIsInstance(loaded, dict)
        self.assertIn("system", loaded)

    def test_user_config_overrides_defaults(self):
        """A user YAML file should override specific values via deep merge."""
        user_yaml_path = os.path.join(self.tmpdir, "user.yaml")
        user_overrides = {
            "system": {
                "learning_rate": 0.05,
            },
            "visual": {
                "input_width": 256,
            },
        }
        with open(user_yaml_path, 'w') as f:
            yaml.dump(user_overrides, f)

        cfg = CoreSystemConfig(config_path=user_yaml_path)
        # Overridden values
        self.assertAlmostEqual(cfg.get_system_config()["learning_rate"], 0.05)
        self.assertEqual(cfg.get_component_config("visual")["input_width"], 256)
        # Non-overridden values from YAML should still be present
        self.assertEqual(cfg.get_system_config()["multimodal_size"], 10)
        self.assertEqual(cfg.get_component_config("visual")["input_height"], 24)

    def test_load_from_file_classmethod(self):
        """load_from_file returns a usable CoreSystemConfig."""
        yaml_path = os.path.join(self.tmpdir, "test.yaml")
        data = {"system": {"multimodal_size": 999}}
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f)
        cfg = CoreSystemConfig.load_from_file(yaml_path)
        self.assertEqual(cfg.get_system_config()["multimodal_size"], 999)


# ---------------------------------------------------------------------------
# Core: configuration validation
# ---------------------------------------------------------------------------

class TestCoreConfigValidation(unittest.TestCase):
    """Test that validation fills in missing sections and keys."""

    def _fresh_core(self):
        return CoreSystemConfig()

    def test_empty_config_gets_filled(self):
        """Starting from an empty config dict, validation adds required sections."""
        cfg = self._fresh_core()
        cfg.config = {}
        cfg._validate_config()
        required = [
            "system", "visual_processor", "audio_processor",
            "multimodal_association", "temporal_prediction",
            "stability", "structural_plasticity",
            "av_capture", "monitoring",
        ]
        for section in required:
            self.assertIn(section, cfg.config)

    def test_missing_multimodal_size_filled(self):
        cfg = self._fresh_core()
        cfg.config["system"] = {}
        cfg._validate_config()
        self.assertEqual(cfg.config["system"]["multimodal_size"], 128)

    def test_missing_learning_rate_filled(self):
        cfg = self._fresh_core()
        cfg.config["system"] = {}
        cfg._validate_config()
        self.assertAlmostEqual(cfg.config["system"]["learning_rate"], 0.01)

    def test_log_level_strings_converted(self):
        cfg = self._fresh_core()
        for level_str, level_int in [
            ("DEBUG", logging.DEBUG),
            ("INFO", logging.INFO),
            ("WARNING", logging.WARNING),
            ("ERROR", logging.ERROR),
            ("CRITICAL", logging.CRITICAL),
        ]:
            cfg.config["system"]["log_level"] = level_str
            cfg._validate_config()
            self.assertEqual(cfg.config["system"]["log_level"], level_int)

    def test_existing_values_not_overwritten(self):
        cfg = self._fresh_core()
        cfg.config["system"] = {"multimodal_size": 512, "learning_rate": 0.5}
        cfg._validate_config()
        self.assertEqual(cfg.config["system"]["multimodal_size"], 512)
        self.assertAlmostEqual(cfg.config["system"]["learning_rate"], 0.5)


# ---------------------------------------------------------------------------
# Core: deep update
# ---------------------------------------------------------------------------

class TestCoreDeepUpdate(unittest.TestCase):
    """Test _deep_update merges nested dicts without losing existing keys."""

    def _fresh_core(self):
        return CoreSystemConfig()

    def test_deep_update_preserves_existing(self):
        cfg = self._fresh_core()
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        update = {"a": {"x": 10, "z": 30}}
        cfg._deep_update(base, update)
        self.assertEqual(base["a"]["x"], 10)
        self.assertEqual(base["a"]["y"], 2)
        self.assertEqual(base["a"]["z"], 30)
        self.assertEqual(base["b"], 3)

    def test_deep_update_replaces_non_dict(self):
        cfg = self._fresh_core()
        base = {"a": 1}
        update = {"a": {"nested": True}}
        cfg._deep_update(base, update)
        self.assertEqual(base["a"], {"nested": True})

    def test_deep_update_adds_new_top_level_key(self):
        cfg = self._fresh_core()
        base = {"a": 1}
        update = {"b": 2}
        cfg._deep_update(base, update)
        self.assertEqual(base["b"], 2)
        self.assertEqual(base["a"], 1)


# ---------------------------------------------------------------------------
# Core: set_config_value
# ---------------------------------------------------------------------------

class TestCoreSetConfigValue(unittest.TestCase):
    """Test dot-path set_config_value."""

    def setUp(self):
        self.cfg = CoreSystemConfig()

    def test_set_existing_value(self):
        self.cfg.set_config_value("system.learning_rate", 0.1)
        self.assertAlmostEqual(self.cfg.get_system_config()["learning_rate"], 0.1)

    def test_set_new_nested_value(self):
        self.cfg.set_config_value("new_section.new_key", 42)
        self.assertEqual(self.cfg.get_config()["new_section"]["new_key"], 42)

    def test_set_deeply_nested_value(self):
        self.cfg.set_config_value("a.b.c.d", "deep")
        self.assertEqual(self.cfg.get_config()["a"]["b"]["c"]["d"], "deep")

    def test_set_single_level_key(self):
        self.cfg.set_config_value("top_level", True)
        self.assertTrue(self.cfg.get_config()["top_level"])


# ---------------------------------------------------------------------------
# Core: dictionary-style access
# ---------------------------------------------------------------------------

class TestCoreDictAccess(unittest.TestCase):
    """Test __getitem__ and __contains__ on CoreSystemConfig."""

    def setUp(self):
        self.cfg = CoreSystemConfig()

    def test_getitem_returns_section(self):
        sys_cfg = self.cfg["system"]
        self.assertIsInstance(sys_cfg, dict)
        self.assertIn("multimodal_size", sys_cfg)

    def test_getitem_missing_returns_empty(self):
        result = self.cfg["nonexistent_section"]
        self.assertEqual(result, {})

    def test_contains_existing(self):
        self.assertIn("system", self.cfg)

    def test_contains_missing(self):
        self.assertNotIn("nonexistent_section", self.cfg)

    def test_getitem_matches_get_component_config(self):
        self.assertEqual(self.cfg["system"], self.cfg.get_component_config("system"))


# ---------------------------------------------------------------------------
# Tests for self_organizing_av_system.config.configuration.SystemConfig
# ---------------------------------------------------------------------------

class TestConfigSystemConfigInitialization(unittest.TestCase):
    """Test ConfigSystemConfig construction and default state."""

    def setUp(self):
        self.cfg = ConfigSystemConfig()

    def test_config_is_dict(self):
        self.assertIsInstance(self.cfg.config, dict)

    def test_required_sections_present(self):
        required = ["system", "visual", "audio", "capture", "monitor", "checkpointing"]
        for section in required:
            self.assertIn(section, self.cfg.config, f"Missing section: {section}")

    def test_system_defaults(self):
        sys_cfg = self.cfg.get_system_config()
        self.assertEqual(sys_cfg["multimodal_size"], 100)
        self.assertAlmostEqual(sys_cfg["learning_rate"], 0.01)
        self.assertEqual(sys_cfg["learning_rule"], "oja")
        self.assertEqual(sys_cfg["prune_interval"], 1000)
        self.assertEqual(sys_cfg["structural_plasticity_interval"], 5000)


class TestConfigVisualSettings(unittest.TestCase):
    """Test visual processor defaults from ConfigSystemConfig."""

    def setUp(self):
        self.visual = ConfigSystemConfig().get_visual_config()

    def test_resolution(self):
        self.assertEqual(self.visual["input_width"], 64)
        self.assertEqual(self.visual["input_height"], 64)

    def test_grayscale_enabled(self):
        self.assertTrue(self.visual["use_grayscale"])

    def test_patch_extraction(self):
        self.assertEqual(self.visual["patch_size"], 8)
        self.assertEqual(self.visual["stride"], 4)

    def test_contrast_normalization(self):
        self.assertTrue(self.visual["contrast_normalize"])

    def test_layer_sizes(self):
        self.assertEqual(self.visual["layer_sizes"], [200, 100, 50])


class TestConfigAudioSettings(unittest.TestCase):
    """Test audio processor defaults from ConfigSystemConfig."""

    def setUp(self):
        self.audio = ConfigSystemConfig().get_audio_config()

    def test_sample_rate(self):
        self.assertEqual(self.audio["sample_rate"], 22050)

    def test_mel_spectrogram_settings(self):
        self.assertEqual(self.audio["window_size"], 1024)
        self.assertEqual(self.audio["hop_length"], 512)
        self.assertEqual(self.audio["n_mels"], 64)

    def test_frequency_range(self):
        self.assertEqual(self.audio["min_freq"], 50)
        self.assertEqual(self.audio["max_freq"], 8000)

    def test_normalization_enabled(self):
        self.assertTrue(self.audio["normalize"])

    def test_layer_sizes(self):
        self.assertEqual(self.audio["layer_sizes"], [150, 75, 40])


class TestConfigCaptureSettings(unittest.TestCase):
    """Test capture defaults from ConfigSystemConfig."""

    def setUp(self):
        self.capture = ConfigSystemConfig().get_capture_config()

    def test_webcam_resolution(self):
        self.assertEqual(self.capture["video_width"], 640)
        self.assertEqual(self.capture["video_height"], 480)

    def test_fps(self):
        self.assertEqual(self.capture["fps"], 30)

    def test_audio_channels(self):
        self.assertEqual(self.capture["audio_channels"], 1)

    def test_chunk_size(self):
        self.assertEqual(self.capture["chunk_size"], 1024)


class TestConfigMonitorSettings(unittest.TestCase):
    """Test monitor defaults from ConfigSystemConfig."""

    def setUp(self):
        self.monitor = ConfigSystemConfig().get_monitor_config()

    def test_update_interval(self):
        self.assertAlmostEqual(self.monitor["update_interval"], 0.5)

    def test_save_snapshots_disabled(self):
        self.assertFalse(self.monitor["save_snapshots"])

    def test_snapshot_interval(self):
        self.assertEqual(self.monitor["snapshot_interval"], 1000)

    def test_snapshot_path(self):
        self.assertEqual(self.monitor["snapshot_path"], "snapshots")


class TestConfigCheckpointSettings(unittest.TestCase):
    """Test checkpointing defaults from ConfigSystemConfig."""

    def setUp(self):
        self.checkpoint = ConfigSystemConfig().get_checkpointing_config()

    def test_enabled(self):
        self.assertTrue(self.checkpoint["enabled"])

    def test_checkpoint_interval(self):
        self.assertEqual(self.checkpoint["checkpoint_interval"], 5000)

    def test_checkpoint_dir(self):
        self.assertEqual(self.checkpoint["checkpoint_dir"], "checkpoints")

    def test_max_checkpoints(self):
        self.assertEqual(self.checkpoint["max_checkpoints"], 3)

    def test_load_latest(self):
        self.assertTrue(self.checkpoint["load_latest"])

    def test_save_on_exit(self):
        self.assertTrue(self.checkpoint["save_on_exit"])


class TestConfigYAMLFileSupport(unittest.TestCase):
    """Test YAML load/save round-trip for ConfigSystemConfig."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_reload(self):
        cfg = ConfigSystemConfig()
        save_path = os.path.join(self.tmpdir, "saved_config.yaml")
        result = cfg.save_config(save_path)
        self.assertTrue(result)
        self.assertTrue(os.path.isfile(save_path))

        # Reload from the saved file
        cfg2 = ConfigSystemConfig(config_file=save_path)
        for section in cfg.config:
            self.assertIn(section, cfg2.config)
            for key in cfg.config[section]:
                self.assertEqual(
                    cfg.config[section][key],
                    cfg2.config[section][key],
                    f"Mismatch for {section}.{key}",
                )

    def test_load_nonexistent_file_uses_defaults(self):
        cfg = ConfigSystemConfig(config_file="/tmp/__nonexistent_atlas_config__.yaml")
        # Should gracefully fall back to defaults
        sys_cfg = cfg.get_system_config()
        self.assertEqual(sys_cfg["multimodal_size"], 100)

    def test_user_config_overrides_defaults(self):
        """A user YAML file should override specific values while keeping defaults."""
        user_yaml_path = os.path.join(self.tmpdir, "user.yaml")
        user_overrides = {
            "system": {
                "learning_rate": 0.05,
                "multimodal_size": 256,
            },
            "visual": {
                "input_width": 128,
            },
        }
        with open(user_yaml_path, 'w') as f:
            yaml.dump(user_overrides, f)

        cfg = ConfigSystemConfig(config_file=user_yaml_path)
        # Overridden values
        self.assertAlmostEqual(cfg.get_system_config()["learning_rate"], 0.05)
        self.assertEqual(cfg.get_system_config()["multimodal_size"], 256)
        self.assertEqual(cfg.get_visual_config()["input_width"], 128)
        # Non-overridden values should still be present from defaults
        self.assertEqual(cfg.get_system_config()["learning_rule"], "oja")
        self.assertEqual(cfg.get_visual_config()["input_height"], 64)
        self.assertTrue(cfg.get_visual_config()["use_grayscale"])

    def test_partial_section_preserves_missing_keys(self):
        """Loading a YAML with only some keys in a section keeps the rest."""
        partial_yaml = os.path.join(self.tmpdir, "partial.yaml")
        with open(partial_yaml, 'w') as f:
            yaml.dump({"audio": {"sample_rate": 44100}}, f)
        cfg = ConfigSystemConfig(config_file=partial_yaml)
        audio = cfg.get_audio_config()
        self.assertEqual(audio["sample_rate"], 44100)
        # Other audio keys should remain at defaults
        self.assertEqual(audio["window_size"], 1024)
        self.assertEqual(audio["n_mels"], 64)


class TestConfigConfigValidation(unittest.TestCase):
    """Test ConfigSystemConfig validation via _update_config."""

    def test_update_ignores_unknown_sections(self):
        cfg = ConfigSystemConfig()
        original_keys = set(cfg.config.keys())
        cfg._update_config({"unknown_section": {"foo": "bar"}})
        # No new sections should appear
        self.assertEqual(set(cfg.config.keys()), original_keys)

    def test_update_ignores_unknown_keys_in_known_section(self):
        cfg = ConfigSystemConfig()
        cfg._update_config({"system": {"nonexistent_key": 999}})
        self.assertNotIn("nonexistent_key", cfg.config["system"])

    def test_update_changes_known_keys(self):
        cfg = ConfigSystemConfig()
        cfg._update_config({"system": {"learning_rate": 0.1}})
        self.assertAlmostEqual(cfg.get_system_config()["learning_rate"], 0.1)


class TestConfigDictAccess(unittest.TestCase):
    """Test __getitem__ on ConfigSystemConfig."""

    def setUp(self):
        self.cfg = ConfigSystemConfig()

    def test_getitem_returns_section(self):
        sys_cfg = self.cfg["system"]
        self.assertIsInstance(sys_cfg, dict)
        self.assertIn("multimodal_size", sys_cfg)

    def test_getitem_missing_raises_keyerror(self):
        with self.assertRaises(KeyError):
            _ = self.cfg["nonexistent_section"]


class TestConfigUpdateMethod(unittest.TestCase):
    """Test the update(section, key, value) method on ConfigSystemConfig."""

    def setUp(self):
        self.cfg = ConfigSystemConfig()

    def test_update_existing_value(self):
        self.cfg.update("system", "learning_rate", 0.1)
        self.assertAlmostEqual(self.cfg.get_system_config()["learning_rate"], 0.1)

    def test_update_nonexistent_section_raises(self):
        with self.assertRaises(KeyError):
            self.cfg.update("nonexistent", "key", "value")

    def test_update_nonexistent_key_raises(self):
        with self.assertRaises(KeyError):
            self.cfg.update("system", "nonexistent_key", "value")

    def test_update_all_sections(self):
        """Verify we can update at least one key in every section."""
        updates = {
            "system": ("learning_rate", 0.99),
            "visual": ("input_width", 256),
            "audio": ("sample_rate", 44100),
            "capture": ("fps", 60),
            "monitor": ("update_interval", 2.0),
            "checkpointing": ("max_checkpoints", 10),
        }
        for section, (key, value) in updates.items():
            self.cfg.update(section, key, value)
            self.assertEqual(self.cfg.config[section][key], value)


if __name__ == '__main__':
    unittest.main()
