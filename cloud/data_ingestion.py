#!/usr/bin/env python3
"""
Atlas Data Ingestion Service

This service provides continuous data streams for Atlas to learn from:
1. YouTube/Video streaming
2. Webcam capture
3. Sample video generation
4. Image dataset ingestion
5. Real-time data augmentation
"""

import os
import sys
import time
import logging
import threading
import random
from datetime import datetime
from typing import Optional, List, Dict, Any
import numpy as np

# Video processing
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# HTTP requests for downloading
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class DataIngestionService:
    """
    Service for ingesting data from various sources for Atlas to learn from.
    """

    def __init__(self):
        self.logger = logging.getLogger("DataIngestion")
        self.running = False
        self.input_dir = os.environ.get('ATLAS_INPUT_DIR', '/data/input')
        self.ingestion_interval = int(os.environ.get('INGESTION_INTERVAL', '60'))

        # Create input directory
        os.makedirs(self.input_dir, exist_ok=True)

        # Feature flags
        self.enable_youtube = os.environ.get('INGESTION_YOUTUBE_ENABLED', 'false').lower() == 'true'
        self.enable_webcam = os.environ.get('INGESTION_WEBCAM_ENABLED', 'false').lower() == 'true'
        self.enable_sample_videos = os.environ.get('INGESTION_SAMPLE_VIDEOS', 'true').lower() == 'true'
        self.enable_synthetic = os.environ.get('INGESTION_SYNTHETIC_ENABLED', 'true').lower() == 'true'

        self.logger.info(f"Data Ingestion Service initialized")
        self.logger.info(f"Input directory: {self.input_dir}")

    def start(self):
        """Start the data ingestion service."""
        self.logger.info("Starting Data Ingestion Service...")
        self.running = True

        while self.running:
            try:
                self._ingestion_cycle()
                time.sleep(self.ingestion_interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Ingestion error: {e}")
                time.sleep(5)

    def stop(self):
        """Stop the data ingestion service."""
        self.running = False
        self.logger.info("Data Ingestion Service stopped")

    def _ingestion_cycle(self):
        """Run one cycle of data ingestion."""
        self.logger.info("Running ingestion cycle...")

        # Generate synthetic data
        if self.enable_synthetic:
            self._generate_synthetic_data()

        # Generate sample videos
        if self.enable_sample_videos:
            self._generate_sample_videos()

        # Download sample data
        self._download_sample_images()

    def _generate_synthetic_data(self):
        """Generate synthetic training data."""
        if not HAS_OPENCV:
            return

        self.logger.info("Generating synthetic data...")

        # Generate various patterns
        for i in range(10):
            # Random geometric patterns
            img = self._create_geometric_pattern()
            filename = f"synthetic_geometric_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.png"
            filepath = os.path.join(self.input_dir, filename)
            cv2.imwrite(filepath, img)

        # Generate moving patterns (short videos)
        for i in range(2):
            self._create_motion_video(i)

        self.logger.info(f"Generated synthetic data in {self.input_dir}")

    def _create_geometric_pattern(self) -> np.ndarray:
        """Create a geometric pattern image."""
        size = 128
        img = np.zeros((size, size, 3), dtype=np.uint8)

        # Background color
        img[:, :] = [random.randint(0, 50), random.randint(0, 50), random.randint(0, 50)]

        # Random shapes
        num_shapes = random.randint(1, 5)
        for _ in range(num_shapes):
            shape_type = random.choice(['circle', 'rectangle', 'line', 'ellipse'])
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

            if shape_type == 'circle':
                center = (random.randint(10, size-10), random.randint(10, size-10))
                radius = random.randint(5, 30)
                cv2.circle(img, center, radius, color, -1)

            elif shape_type == 'rectangle':
                pt1 = (random.randint(0, size//2), random.randint(0, size//2))
                pt2 = (random.randint(size//2, size), random.randint(size//2, size))
                cv2.rectangle(img, pt1, pt2, color, -1)

            elif shape_type == 'line':
                pt1 = (random.randint(0, size), random.randint(0, size))
                pt2 = (random.randint(0, size), random.randint(0, size))
                cv2.line(img, pt1, pt2, color, random.randint(1, 5))

            elif shape_type == 'ellipse':
                center = (random.randint(20, size-20), random.randint(20, size-20))
                axes = (random.randint(5, 30), random.randint(5, 30))
                angle = random.randint(0, 180)
                cv2.ellipse(img, center, axes, angle, 0, 360, color, -1)

        return img

    def _create_motion_video(self, index: int):
        """Create a short video with moving objects."""
        if not HAS_OPENCV:
            return

        size = 128
        fps = 30
        duration = 5  # seconds
        num_frames = fps * duration

        filename = f"synthetic_motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{index}.mp4"
        filepath = os.path.join(self.input_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, fps, (size, size))

        # Moving object parameters
        obj_x, obj_y = size // 2, size // 2
        obj_vx, obj_vy = random.uniform(-3, 3), random.uniform(-3, 3)
        obj_radius = random.randint(5, 15)
        obj_color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        bg_color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))

        for frame_idx in range(num_frames):
            # Create frame
            frame = np.zeros((size, size, 3), dtype=np.uint8)
            frame[:, :] = bg_color

            # Update position with bouncing
            obj_x += obj_vx
            obj_y += obj_vy

            if obj_x <= obj_radius or obj_x >= size - obj_radius:
                obj_vx *= -1
                obj_x = max(obj_radius, min(size - obj_radius, obj_x))

            if obj_y <= obj_radius or obj_y >= size - obj_radius:
                obj_vy *= -1
                obj_y = max(obj_radius, min(size - obj_radius, obj_y))

            # Draw object
            cv2.circle(frame, (int(obj_x), int(obj_y)), obj_radius, obj_color, -1)

            out.write(frame)

        out.release()
        self.logger.info(f"Created motion video: {filepath}")

    def _generate_sample_videos(self):
        """Generate sample training videos."""
        if not HAS_OPENCV:
            return

        self.logger.info("Generating sample videos...")

        # Create diverse sample videos
        video_types = [
            ('wave', self._create_wave_video),
            ('spiral', self._create_spiral_video),
            ('grid', self._create_grid_video),
            ('noise', self._create_noise_video),
        ]

        for video_type, generator in video_types:
            try:
                generator(video_type)
            except Exception as e:
                self.logger.warning(f"Failed to generate {video_type} video: {e}")

    def _create_wave_video(self, name: str):
        """Create a wave pattern video."""
        if not HAS_OPENCV:
            return

        size = 128
        fps = 30
        duration = 3
        num_frames = fps * duration

        filename = f"sample_wave_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        filepath = os.path.join(self.input_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, fps, (size, size))

        for t in range(num_frames):
            frame = np.zeros((size, size), dtype=np.uint8)
            for y in range(size):
                for x in range(size):
                    val = int(127 + 127 * np.sin(x * 0.1 + t * 0.2) * np.sin(y * 0.1 + t * 0.15))
                    frame[y, x] = val
            frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(frame_color)

        out.release()
        self.logger.info(f"Created wave video: {filepath}")

    def _create_spiral_video(self, name: str):
        """Create a spiral pattern video."""
        if not HAS_OPENCV:
            return

        size = 128
        fps = 30
        duration = 3
        num_frames = fps * duration

        filename = f"sample_spiral_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        filepath = os.path.join(self.input_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, fps, (size, size))

        for t in range(num_frames):
            frame = np.zeros((size, size, 3), dtype=np.uint8)
            cx, cy = size // 2, size // 2
            angle = t * 0.1
            for r in range(0, size // 2, 2):
                x = int(cx + r * np.cos(angle + r * 0.1))
                y = int(cy + r * np.sin(angle + r * 0.1))
                if 0 <= x < size and 0 <= y < size:
                    color = (int(r * 4) % 256, int(r * 2 + t) % 256, int(255 - r * 2) % 256)
                    cv2.circle(frame, (x, y), 2, color, -1)
            out.write(frame)

        out.release()
        self.logger.info(f"Created spiral video: {filepath}")

    def _create_grid_video(self, name: str):
        """Create a moving grid pattern video."""
        if not HAS_OPENCV:
            return

        size = 128
        fps = 30
        duration = 3
        num_frames = fps * duration

        filename = f"sample_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        filepath = os.path.join(self.input_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, fps, (size, size))

        for t in range(num_frames):
            frame = np.zeros((size, size, 3), dtype=np.uint8)
            offset = int(t * 0.5) % 16
            for y in range(0, size, 16):
                for x in range(0, size, 16):
                    px = (x + offset) % size
                    py = (y + offset // 2) % size
                    cv2.rectangle(frame, (px, py), (min(px + 8, size - 1), min(py + 8, size - 1)),
                                (255, 255, 255), -1)
            out.write(frame)

        out.release()
        self.logger.info(f"Created grid video: {filepath}")

    def _create_noise_video(self, name: str):
        """Create a noise pattern video for texture learning."""
        if not HAS_OPENCV:
            return

        size = 128
        fps = 30
        duration = 2
        num_frames = fps * duration

        filename = f"sample_noise_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        filepath = os.path.join(self.input_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, fps, (size, size))

        for _ in range(num_frames):
            # Perlin-like noise (simplified)
            noise = np.random.randint(0, 256, (size // 4, size // 4), dtype=np.uint8)
            noise = cv2.resize(noise, (size, size), interpolation=cv2.INTER_LINEAR)
            frame_color = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
            out.write(frame_color)

        out.release()
        self.logger.info(f"Created noise video: {filepath}")

    def _download_sample_images(self):
        """Download sample images from public sources."""
        if not HAS_REQUESTS:
            return

        # Check if we already have enough images
        existing_images = [f for f in os.listdir(self.input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if len(existing_images) > 100:
            self.logger.info("Sufficient images in input directory, skipping download")
            return

        self.logger.info("Generating placeholder images (no external downloads for safety)...")

        # Instead of downloading, generate diverse images locally
        if HAS_OPENCV:
            for i in range(20):
                img = self._create_geometric_pattern()
                filename = f"generated_pattern_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.png"
                filepath = os.path.join(self.input_dir, filename)
                cv2.imwrite(filepath, img)

    def get_status(self) -> Dict[str, Any]:
        """Get current service status."""
        input_files = os.listdir(self.input_dir) if os.path.exists(self.input_dir) else []

        return {
            "status": "running" if self.running else "stopped",
            "input_directory": self.input_dir,
            "files_in_queue": len(input_files),
            "features": {
                "youtube": self.enable_youtube,
                "webcam": self.enable_webcam,
                "sample_videos": self.enable_sample_videos,
                "synthetic": self.enable_synthetic
            }
        }


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger("Main")
    logger.info("Starting Atlas Data Ingestion Service...")

    service = DataIngestionService()

    try:
        service.start()
    except KeyboardInterrupt:
        service.stop()


if __name__ == "__main__":
    main()
