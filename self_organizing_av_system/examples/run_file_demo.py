#!/usr/bin/env python3
"""
File-based demo of the Self-Organizing Audio-Visual Learning System.

This script processes a video file using the self-organizing system.
"""
import os
import sys
import logging
import argparse
import glob

# Add parent directory to path to import the package
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)

# Import directly from modules to avoid circular imports
from models.visual.processor import VisualProcessor
from models.audio.processor import AudioProcessor
# Import system directly from core.system instead of through core
from core.system import SelfOrganizingAVSystem
from utils.capture import VideoFileReader
from utils.monitor import NetworkMonitor
from config.configuration import SystemConfig


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process a video file with the Self-Organizing AV System')
    parser.add_argument('video_file', type=str, help='Path to the video file to process')
    parser.add_argument('--config', type=str, help='Path to custom configuration file')
    parser.add_argument('--no-checkpoints', action='store_true', help='Disable loading/saving checkpoints')
    parser.add_argument('--no-load-checkpoint', action='store_true', help='Do not load existing checkpoint')
    parser.add_argument('--checkpoint-name', type=str, help='Specific checkpoint to load (filename only)')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("FileDemo")
    
    # Check if video file exists
    if not os.path.exists(args.video_file):
        logger.error(f"Video file not found: {args.video_file}")
        return
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        config = SystemConfig(args.config)
        logger.info(f"Using custom configuration from {args.config}")
    else:
        # Use default config
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default_config.yaml')
        config = SystemConfig(config_path)
        logger.info("Using default configuration")
    
    logger.info("Creating processors and system...")
    
    # Get configurations
    visual_config = config.get_visual_config()
    audio_config = config.get_audio_config()
    system_config = config.get_system_config()
    
    # Create processors
    visual_processor = VisualProcessor(**visual_config)
    audio_processor = AudioProcessor(**audio_config)
    
    # Create self-organizing system
    system = SelfOrganizingAVSystem(
        visual_processor=visual_processor,
        audio_processor=audio_processor,
        config=system_config
    )
    
    # Get checkpointing configuration
    checkpoint_config = config.get_checkpointing_config()
    checkpoint_enabled = checkpoint_config["enabled"] and not args.no_checkpoints
    checkpoint_interval = checkpoint_config["checkpoint_interval"]
    checkpoint_dir = checkpoint_config["checkpoint_dir"]
    max_checkpoints = checkpoint_config["max_checkpoints"]
    load_latest = checkpoint_config["load_latest"] and not args.no_load_checkpoint
    save_on_exit = checkpoint_config["save_on_exit"] and not args.no_checkpoints
    
    # Video file base name (for checkpoint naming)
    video_basename = os.path.splitext(os.path.basename(args.video_file))[0]
    
    # Create checkpoint directory if enabled
    if checkpoint_enabled:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Checkpointing enabled. Directory: {checkpoint_dir}")
        
        # Try to load checkpoint if requested
        if load_latest:
            if args.checkpoint_name:
                # Load specific checkpoint
                checkpoint_file = os.path.join(checkpoint_dir, args.checkpoint_name)
                if os.path.exists(checkpoint_file):
                    logger.info(f"Loading specified checkpoint: {checkpoint_file}")
                    if system.load_state(checkpoint_file):
                        logger.info(f"Successfully loaded checkpoint with {system.frame_count} processed frames")
                    else:
                        logger.warning("Failed to load specified checkpoint, starting fresh")
                else:
                    logger.warning(f"Specified checkpoint not found: {checkpoint_file}")
            else:
                # Find latest checkpoint file for this video
                checkpoint_pattern = os.path.join(checkpoint_dir, f"*{video_basename}*.pkl")
                checkpoint_files = glob.glob(checkpoint_pattern)
                if checkpoint_files:
                    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
                    logger.info(f"Loading latest checkpoint for this video: {latest_checkpoint}")
                    
                    if system.load_state(latest_checkpoint):
                        logger.info(f"Successfully loaded checkpoint with {system.frame_count} processed frames")
                    else:
                        logger.warning("Failed to load checkpoint, starting fresh")
                else:
                    logger.info("No existing checkpoints found for this video, starting fresh")
    
    # Create video file reader
    reader = VideoFileReader(
        video_file=args.video_file,
        target_width=visual_config["input_width"],
        target_height=visual_config["input_height"]
    )
    
    # Create monitor
    monitor_config = config.get_monitor_config()
    monitor = NetworkMonitor(update_interval=monitor_config["update_interval"])
    
    logger.info(f"Loading video file: {args.video_file}")
    if not reader.load():
        logger.error("Failed to load video file")
        return
    
    # Start reader
    if not reader.start():
        logger.error("Failed to start video reader")
        return
    
    try:
        # Get all frames and audio
        frames = reader.get_all_frames()
        audio_waveform, sample_rate = reader.get_audio_waveform()
        
        logger.info(f"Loaded {len(frames)} frames and {len(audio_waveform)} audio samples")
        
        # Start monitor (non-blocking)
        monitor.start(system, display=True)
        
        # Skip frames if we've already processed some (from checkpoint)
        if system.frame_count > 0:
            logger.info(f"Resuming from frame {system.frame_count} (skipping already processed frames)")
            if system.frame_count < len(frames):
                frames = frames[system.frame_count:]
                # Adjust audio if possible
                # This is a simplification - in a real implementation, we'd need to calculate
                # the corresponding audio segment more precisely
                audio_samples_per_frame = len(audio_waveform) // len(frames)
                audio_offset = system.frame_count * audio_samples_per_frame
                if audio_offset < len(audio_waveform):
                    audio_waveform = audio_waveform[audio_offset:]
            else:
                logger.warning("Checkpoint frame count exceeds available frames - reprocessing entire video")
                system.frame_count = 0  # Reset to avoid issues
        
        # Process the entire sequence
        logger.info("Processing video sequence...")
        
        total_frames = len(frames)
        last_checkpoint_frame = system.frame_count
        
        # Process frames in batches to allow checkpointing
        batch_size = min(1000, total_frames)  # Process 1000 frames at a time
        activation_sequence = []
        
        for batch_start in range(0, total_frames, batch_size):
            batch_end = min(batch_start + batch_size, total_frames)
            logger.info(f"Processing frames {batch_start} to {batch_end-1} of {total_frames}")
            
            # Process this batch
            batch_frames = frames[batch_start:batch_end]
            
            # Calculate corresponding audio slice
            frames_ratio = len(batch_frames) / total_frames
            audio_length = int(len(audio_waveform) * frames_ratio)
            audio_start = int(len(audio_waveform) * (batch_start / total_frames))
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
            if checkpoint_enabled and frames_processed >= checkpoint_interval:
                checkpoint_file = os.path.join(
                    checkpoint_dir,
                    f"checkpoint_{video_basename}_{system.frame_count}.pkl"
                )
                if system.save_state(checkpoint_file):
                    logger.info(f"Saved checkpoint at frame {system.frame_count}")
                    last_checkpoint_frame = system.frame_count
                    
                    # Manage checkpoint files
                    checkpoint_pattern = os.path.join(checkpoint_dir, f"*{video_basename}*.pkl")
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
        
        logger.info(f"Processed {len(activation_sequence)} frames")
        
        # Analyze learned representations
        logger.info("Analysis of learned representations:")
        
        # Check for cross-modal associations
        associations = system.multimodal.analyze_associations()
        for key, assocs in associations.items():
            if assocs:
                logger.info(f"Found {len(assocs)} strong {key} associations")
            else:
                logger.info(f"No strong {key} associations found yet")
        
        # Save final checkpoint if enabled
        if checkpoint_enabled and save_on_exit:
            checkpoint_file = os.path.join(
                checkpoint_dir,
                f"checkpoint_{video_basename}_final_{system.frame_count}.pkl"
            )
            if system.save_state(checkpoint_file):
                logger.info(f"Saved final checkpoint at frame {system.frame_count}")
        
        # Keep the visualization open until user interrupts
        logger.info("Processing complete. Press Ctrl+C to exit.")
        while True:
            pass
    
    except KeyboardInterrupt:
        logger.info("Demo stopped by user")
    finally:
        # Clean up resources
        reader.stop()
        monitor.stop()
        
        logger.info(f"Demo ended after processing {system.frame_count} frames")
        
        # Save snapshot if configured
        if monitor_config["save_snapshots"]:
            os.makedirs(monitor_config["snapshot_path"], exist_ok=True)
            filename = os.path.join(
                monitor_config["snapshot_path"],
                f"{video_basename}_final_snapshot.png"
            )
            monitor.save_snapshot(filename)
            logger.info(f"Saved final snapshot to {filename}")


if __name__ == "__main__":
    main() 