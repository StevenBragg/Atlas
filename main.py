#!/usr/bin/env python3
"""
Atlas - Self-Organizing Audio-Visual Learning System
Main entry point with unified configuration management.
"""

import argparse
import logging
import os
import sys
import time
import numpy as np
import cv2
import glob
from typing import Optional, Dict, Any, List

# Add self_organizing_av_system to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'self_organizing_av_system'))

from self_organizing_av_system.models.visual.processor import VisualProcessor
from self_organizing_av_system.models.audio.processor import AudioProcessor
from self_organizing_av_system.models.multimodal.system import SelfOrganizingAVSystem
from self_organizing_av_system.utils.capture import AVCapture, VideoFileReader
from self_organizing_av_system.gui.tk_monitor import TkMonitor
from config.configuration import (
    ConfigurationManager, init_config, get_config,
    ConfigMode, SystemConfig, VisualConfig, AudioConfig
)


def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level to use
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def create_processors(config: ConfigurationManager) -> tuple:
    """
    Create the visual and audio processors.
    
    Args:
        config: Configuration manager
        
    Returns:
        Tuple of (visual_processor, audio_processor)
    """
    visual_config = config.visual
    audio_config = config.audio
    
    logger = logging.getLogger("Setup")
    
    # Create visual processor
    logger.info("Creating visual processor...")
    visual_processor = VisualProcessor(
        input_width=visual_config.input_width,
        input_height=visual_config.input_height,
        use_grayscale=visual_config.use_grayscale,
        patch_size=visual_config.patch_size,
        stride=visual_config.stride,
        contrast_normalize=visual_config.contrast_normalize,
        layer_sizes=visual_config.layer_sizes
    )
    
    # Create audio processor
    logger.info("Creating audio processor...")
    audio_processor = AudioProcessor(
        sample_rate=audio_config.sample_rate,
        window_size=audio_config.window_size,
        hop_length=audio_config.hop_length,
        n_mels=audio_config.n_mels,
        min_freq=audio_config.min_freq,
        max_freq=audio_config.max_freq,
        normalize=audio_config.normalize,
        layer_sizes=audio_config.layer_sizes
    )
    
    return visual_processor, audio_processor


def create_system(
    config: ConfigurationManager,
    visual_processor: VisualProcessor,
    audio_processor: AudioProcessor
) -> SelfOrganizingAVSystem:
    """
    Create the self-organizing system.
    
    Args:
        config: Configuration manager
        visual_processor: Visual processor
        audio_processor: Audio processor
        
    Returns:
        SelfOrganizingAVSystem instance
    """
    system_config = config.system
    
    logger = logging.getLogger("Setup")
    logger.info("Creating self-organizing AV system...")
    
    system = SelfOrganizingAVSystem(
        visual_processor=visual_processor,
        audio_processor=audio_processor,
        multimodal_size=system_config.multimodal_size,
        prune_interval=system_config.prune_interval,
        structural_plasticity_interval=system_config.structural_plasticity_interval,
        learning_rate=system_config.learning_rate,
        learning_rule=system_config.learning_rule
    )
    
    return system


def load_checkpoint(
    system: SelfOrganizingAVSystem,
    config: ConfigurationManager,
    session_name: str = None
) -> bool:
    """
    Try to load the latest checkpoint for a system.
    
    Args:
        system: The system to load checkpoint for
        config: Configuration manager
        session_name: Optional name of the session for specific checkpoints
        
    Returns:
        Whether a checkpoint was successfully loaded
    """
    logger = logging.getLogger("Checkpoint")
    
    # Get checkpointing configuration
    checkpoint_config = config.checkpointing
    
    if not checkpoint_config.enabled or not checkpoint_config.load_latest:
        return False
    
    checkpoint_dir = checkpoint_config.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Find latest checkpoint
    if session_name:
        checkpoint_pattern = os.path.join(checkpoint_dir, f"*{session_name}*.pkl")
    else:
        checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint_*.pkl")
    
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        logger.info("No checkpoints found")
        return False
    
    # Get the most recent checkpoint
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    logger.info(f"Loading checkpoint: {latest_checkpoint}")
    
    # Load the checkpoint
    if system.load_state(latest_checkpoint):
        logger.info(f"Successfully loaded checkpoint with {system.frame_count} processed frames")
        return True
    else:
        logger.warning("Failed to load checkpoint")
        return False


def save_checkpoint(
    system: SelfOrganizingAVSystem,
    config: ConfigurationManager,
    session_name: str = None,
    is_final: bool = False
) -> bool:
    """
    Save a checkpoint for the system.
    
    Args:
        system: The system to save
        config: Configuration manager
        session_name: Optional name of the session for specific checkpoints
        is_final: Whether this is a final checkpoint
        
    Returns:
        Whether the checkpoint was successfully saved
    """
    logger = logging.getLogger("Checkpoint")
    
    # Get checkpointing configuration
    checkpoint_config = config.checkpointing
    
    if not checkpoint_config.enabled:
        return False
    
    checkpoint_dir = checkpoint_config.checkpoint_dir
    max_checkpoints = checkpoint_config.max_checkpoints
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create checkpoint name
    if is_final:
        prefix = "final"
    else:
        prefix = "checkpoint"
    
    if session_name:
        checkpoint_file = os.path.join(
            checkpoint_dir, 
            f"{prefix}_{session_name}_{system.frame_count}.pkl"
        )
    else:
        checkpoint_file = os.path.join(
            checkpoint_dir,
            f"{prefix}_{system.frame_count}.pkl"
        )
    
    # Save the checkpoint
    if system.save_state(checkpoint_file):
        logger.info(f"Saved checkpoint at frame {system.frame_count}")
        
        # Manage checkpoint files (keep only max_checkpoints most recent)
        if session_name:
            checkpoint_pattern = os.path.join(checkpoint_dir, f"*{session_name}*.pkl")
        else:
            checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint_*.pkl")
        
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if len(checkpoint_files) > max_checkpoints:
            # Sort by creation time
            checkpoint_files.sort(key=os.path.getctime)
            # Remove oldest files
            for old_file in checkpoint_files[:-max_checkpoints]:
                try:
                    os.remove(old_file)
                    logger.debug(f"Removed old checkpoint: {old_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove old checkpoint {old_file}: {e}")
        
        return True
    else:
        logger.warning(f"Failed to save checkpoint")
        return False


def run_live_session(
    system: SelfOrganizingAVSystem,
    config: ConfigurationManager
) -> None:
    """
    Run a live learning session using webcam and microphone.
    
    Args:
        system: Self-organizing system
        config: Configuration manager
    """
    logger = logging.getLogger("LiveSession")
    logger.info("Starting live learning session...")
    
    capture_config = config.capture
    monitor_config = config.monitor
    checkpoint_config = config.checkpointing
    audio_config = config.audio
    
    # Try to load existing checkpoint
    load_checkpoint(system, config, "live")
    
    # Create audio-video capture
    capture = AVCapture(
        video_width=capture_config.video_width,
        video_height=capture_config.video_height,
        fps=capture_config.fps,
        sample_rate=audio_config.sample_rate,
        channels=capture_config.audio_channels,
        chunk_size=capture_config.chunk_size
    )
    
    # Create monitor
    monitor = TkMonitor(system=system)
    
    # Start capture
    if not capture.start():
        logger.error("Failed to start AV capture")
        return
    
    try:
        # Start monitor (non-blocking)
        monitor.start()
        
        logger.info("Press Ctrl+C to stop the session")
        
        last_checkpoint_frame = system.frame_count
        
        # Process frames
        while True:
            # Get synchronized AV pair
            frame, audio_chunk = capture.get_av_pair(block=True, timeout=1.0)
            if frame is None or audio_chunk is None:
                logger.warning("No AV data received, retrying...")
                continue
            
            # Process through system
            system.process_av_pair(frame, audio_chunk, learn=True)
            
            # Save checkpoint if interval reached
            frames_processed = system.frame_count - last_checkpoint_frame
            if frames_processed >= checkpoint_config.checkpoint_interval:
                if save_checkpoint(system, config, "live"):
                    last_checkpoint_frame = system.frame_count
    
    except KeyboardInterrupt:
        logger.info("Session interrupted by user")
    finally:
        # Stop everything
        capture.stop()
        monitor.stop()
        
        logger.info(f"Session ended after {system.frame_count} frames")
        
        # Save final checkpoint if configured
        if checkpoint_config.save_on_exit:
            save_checkpoint(system, config, "live", is_final=True)


def run_file_session(
    system: SelfOrganizingAVSystem,
    video_file: str,
    config: ConfigurationManager
) -> None:
    """
    Run a learning session from a video file.
    
    Args:
        system: Self-organizing system
        video_file: Path to video file
        config: Configuration manager
    """
    logger = logging.getLogger("FileSession")
    logger.info(f"Starting file-based learning session with {video_file}...")
    
    monitor_config = config.monitor
    visual_config = config.visual
    checkpoint_config = config.checkpointing
    audio_config = config.audio
    
    # Video basename for checkpoint naming
    video_basename = os.path.splitext(os.path.basename(video_file))[0]
    
    # Try to load existing checkpoint
    load_checkpoint(system, config, video_basename)
    
    # Create video file reader
    reader = VideoFileReader(
        video_file=video_file,
        target_width=visual_config.input_width,
        target_height=visual_config.input_height
    )
    
    # Create monitor
    monitor = TkMonitor(system=system)
    
    # Load video
    if not reader.load():
        logger.error(f"Failed to load video file: {video_file}")
        return
    
    # Start reader
    if not reader.start():
        logger.error(f"Failed to start video reader")
        return
    
    try:
        # Get all frames and audio
        frames = reader.get_all_frames()
        audio_waveform, sample_rate = reader.get_audio_waveform()
        
        logger.info(f"Loaded {len(frames)} frames and {len(audio_waveform)} audio samples")
        
        # Skip already processed frames if resuming from checkpoint
        if system.frame_count > 0:
            logger.info(f"Resuming from frame {system.frame_count} (skipping already processed frames)")
            if system.frame_count < len(frames):
                frames = frames[system.frame_count:]
                # Adjust audio if possible
                audio_samples_per_frame = len(audio_waveform) // len(frames) if len(frames) > 0 else 0
                audio_offset = system.frame_count * audio_samples_per_frame
                if audio_offset < len(audio_waveform):
                    audio_waveform = audio_waveform[audio_offset:]
            else:
                logger.warning("Checkpoint frame count exceeds available frames - reprocessing entire video")
                system.frame_count = 0  # Reset to avoid issues
        
        # Start monitor (non-blocking)
        monitor.start()
        
        # Process the entire sequence in batches to allow checkpointing
        total_frames = len(frames)
        batch_size = min(1000, total_frames)  # Process up to 1000 frames at a time
        last_checkpoint_frame = system.frame_count
        activation_sequence = []
        
        for batch_start in range(0, total_frames, batch_size):
            batch_end = min(batch_start + batch_size, total_frames)
            logger.info(f"Processing frames {batch_start} to {batch_end-1} of {total_frames}")
            
            # Process this batch
            batch_frames = frames[batch_start:batch_end]
            
            # Calculate corresponding audio slice
            frames_ratio = len(batch_frames) / total_frames if total_frames > 0 else 0
            audio_length = int(len(audio_waveform) * frames_ratio)
            audio_start = int(len(audio_waveform) * (batch_start / total_frames)) if total_frames > 0 else 0
            audio_end = min(audio_start + audio_length, len(audio_waveform))
            batch_audio = audio_waveform[audio_start:audio_end]
            
            # Process batch
            batch_activations = system.process_video_sequence(
                batch_frames,
                batch_audio,
                sample_rate,
                learn=True
            )
            
            # Add to full sequence
            activation_sequence.extend(batch_activations)
            
            # Save checkpoint if interval reached
            frames_processed = system.frame_count - last_checkpoint_frame
            if frames_processed >= checkpoint_config.checkpoint_interval:
                if save_checkpoint(system, config, video_basename):
                    last_checkpoint_frame = system.frame_count
        
        logger.info(f"Processed {len(activation_sequence)} frames")
        
        # Analyze the learned representations
        logger.info("Analysis of learned representations:")
        associations = system.multimodal.analyze_associations()
        for key, assocs in associations.items():
            if assocs:
                logger.info(f"Found {len(assocs)} strong {key} associations")
            else:
                logger.info(f"No strong {key} associations found yet")
                
        # Keep visualization open until interrupted
        logger.info("Processing complete. Press Ctrl+C to exit.")
        while True:
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        logger.info("Session interrupted by user")
    finally:
        # Stop everything
        reader.stop()
        monitor.stop()
        
        logger.info(f"Session ended after {system.frame_count} frames")
        
        # Save final checkpoint if configured
        if checkpoint_config.save_on_exit:
            save_checkpoint(system, config, video_basename, is_final=True)


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Atlas - Self-Organizing Audio-Visual Learning System'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['development', 'production', 'testing', 'minimal'],
        default='development',
        help='Configuration mode (default: development)'
    )
    parser.add_argument(
        '--log-level', '-l',
        type=str,
        default=None,
        help='Logging level (overrides config)'
    )
    parser.add_argument(
        '--video-file', '-v',
        type=str,
        default=None,
        help='Path to video file (for file-based session)'
    )
    parser.add_argument(
        '--no-display', '-nd',
        action='store_true',
        help='Run without visual display'
    )
    parser.add_argument(
        '--no-checkpoints', '-nc',
        action='store_true',
        help='Disable checkpointing'
    )
    parser.add_argument(
        '--no-load-checkpoint', '-nlc',
        action='store_true',
        help='Do not load existing checkpoint'
    )
    parser.add_argument(
        '--checkpoint-name', '-cn',
        type=str,
        help='Specific checkpoint to load'
    )
    parser.add_argument(
        '--hot-reload', '-hr',
        action='store_true',
        help='Enable hot-reloading of configuration'
    )
    parser.add_argument(
        '--save-config',
        type=str,
        default=None,
        help='Save current configuration to file and exit'
    )
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = init_config(
        config_path=args.config,
        mode=args.mode,
        enable_hot_reload=args.hot_reload
    )
    
    # Override log level if specified
    log_level = args.log_level or config.system.log_level
    setup_logging(log_level)
    
    logger = logging.getLogger("Main")
    logger.info(f"Starting Atlas in {args.mode} mode")
    
    # Save config and exit if requested
    if args.save_config:
        config.save(args.save_config)
        logger.info(f"Configuration saved to {args.save_config}")
        return
    
    # Override checkpoint config if requested
    if args.no_checkpoints:
        config.set("checkpointing.enabled", False)
    if args.no_load_checkpoint:
        config.set("checkpointing.load_latest", False)
    
    # Create processors and system
    visual_processor, audio_processor = create_processors(config)
    system = create_system(config, visual_processor, audio_processor)
    
    # Load specific checkpoint if requested
    if args.checkpoint_name and not args.no_load_checkpoint:
        checkpoint_dir = config.checkpointing.checkpoint_dir
        checkpoint_file = os.path.join(checkpoint_dir, args.checkpoint_name)
        if os.path.exists(checkpoint_file):
            logger.info(f"Loading specified checkpoint: {checkpoint_file}")
            if system.load_state(checkpoint_file):
                logger.info(f"Successfully loaded checkpoint with {system.frame_count} processed frames")
            else:
                logger.warning("Failed to load specified checkpoint")
    
    # Run session
    try:
        if args.video_file:
            run_file_session(system, args.video_file, config)
        else:
            run_live_session(system, config)
    finally:
        # Stop hot-reload if enabled
        if args.hot_reload:
            config.stop_hot_reload()


if __name__ == "__main__":
    main()
