#!/usr/bin/env python3
"""
Enhanced Depth Processing Example for Project Daredevil

This example demonstrates the enhanced depth processing capabilities with:
- Advanced reference point selection
- Temporal depth tracking
- Quality assessment
- Spatial context analysis
- Audio-optimized metrics

Usage:
    python3 depth/enhanced_example.py
"""

import cv2
import numpy as np
import time
import sys
import os
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera import CameraStream, create_phone_camera_stream
from depth import (
    EnhancedDepthProcessor,
    create_enhanced_depth_processor,
    DepthLayer,
    MovementDirection,
)


class EnhancedDepthDemo:
    """
    Demonstration of enhanced depth processing with advanced metrics.
    """

    def __init__(self, use_phone_camera: bool = True):
        """
        Initialize the enhanced depth demo.

        Args:
            use_phone_camera: Whether to try using phone camera first
        """
        self.use_phone_camera = use_phone_camera

        # Initialize camera
        self.camera = self._initialize_camera()

        # Initialize enhanced depth processor
        self.depth_processor = create_enhanced_depth_processor()

        # Simulated object detection results
        self.detection_results = []
        self.object_counter = 0

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()

        print("Enhanced Depth Demo initialized")
        print(f"Camera info: {self.camera.get_frame_info()}")

    def _initialize_camera(self):
        """Initialize camera stream."""
        if self.use_phone_camera:
            print("Attempting to connect to phone camera...")
            phone_camera = create_phone_camera_stream()
            if phone_camera:
                print("✓ Phone camera connected")
                return phone_camera

        print("Using built-in camera...")
        return CameraStream()

    def _simulate_object_detection(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Simulate object detection results with multiple objects.

        In a real implementation, this would call the actual object detection module.
        """
        h, w = frame.shape[:2]

        # Create mock detections with different classes and positions
        mock_detections = [
            {
                "bbox": [w // 4, h // 4, w // 4 + 80, h // 4 + 120],
                "class": "person",
                "confidence": 0.95,
            },
            {
                "bbox": [3 * w // 4 - 60, h // 2, 3 * w // 4, h // 2 + 100],
                "class": "bottle",
                "confidence": 0.87,
            },
            {
                "bbox": [w // 2 - 40, 3 * h // 4 - 60, w // 2 + 40, 3 * h // 4],
                "class": "chair",
                "confidence": 0.78,
            },
        ]

        return mock_detections

    def _process_detections_with_enhanced_depth(
        self, frame: np.ndarray, detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process detections with enhanced depth information."""
        processed_detections = []

        for detection in detections:
            bbox = detection["bbox"]
            object_class = detection["class"]

            # Generate unique object ID
            object_id = f"{object_class}_{self.object_counter}"
            self.object_counter += 1

            # Get enhanced depth information
            depth_result = self.depth_processor.get_enhanced_depth_for_spatial_audio(
                frame, bbox, object_id=object_id, object_class=object_class
            )

            # Combine detection and depth information
            processed_detection = detection.copy()
            processed_detection.update(depth_result)

            processed_detections.append(processed_detection)

        return processed_detections

    def _draw_enhanced_detections(
        self, frame: np.ndarray, detections: List[Dict[str, Any]]
    ):
        """Draw detections with enhanced depth information on frame."""
        for i, detection in enumerate(detections):
            bbox = detection["bbox"]
            x1, y1, x2, y2 = bbox

            # Choose color based on depth layer
            depth_layer = detection.get("depth_layer", "medium")
            layer_colors = {
                "very_close": (0, 0, 255),  # Red - urgent
                "close": (0, 165, 255),  # Orange - warning
                "medium": (0, 255, 0),  # Green - normal
                "far": (128, 128, 128),  # Gray - background
            }
            color = layer_colors.get(depth_layer, (0, 255, 0))

            # Draw bounding box with thickness based on audio priority
            audio_priority = detection.get("spatial_audio", {}).audio_priority
            thickness = max(1, min(5, audio_priority // 2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Draw labels with enhanced depth info
            class_name = detection["class"]
            confidence = detection["confidence"]
            raw_depth = detection.get("raw_depth", 0)
            normalized_depth = detection.get("normalized_depth", 0)
            quality_confidence = detection.get("quality_metrics", {}).overall_confidence

            # Primary label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(
                frame, label, (x1, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

            # Depth information
            depth_text = f"D: {raw_depth:.2f}m ({depth_layer})"
            cv2.putText(
                frame,
                depth_text,
                (x1, y1 - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

            # Quality and audio info
            quality_text = f"Q: {quality_confidence:.2f} | A: {audio_priority}"
            cv2.putText(
                frame,
                quality_text,
                (x1, y1 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

            # Movement information
            movement_direction = detection.get("movement_direction", "stationary")
            movement_velocity = detection.get("movement_velocity", 0)
            movement_text = f"M: {movement_direction} ({movement_velocity:.2f})"
            cv2.putText(
                frame,
                movement_text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

    def _display_enhanced_stats(
        self, frame: np.ndarray, detections: List[Dict[str, Any]]
    ):
        """Display enhanced detection and depth statistics."""
        y_offset = 30

        # Display reference depth information
        reference_depth = None
        for detection in detections:
            if detection.get("reference_depth") is not None:
                reference_depth = detection["reference_depth"]
                break

        if reference_depth is not None:
            cv2.putText(
                frame,
                f"Background: {reference_depth:.2f}m",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            y_offset += 25

        # Display depth layer distribution
        layer_counts = {}
        for detection in detections:
            layer = detection.get("depth_layer", "unknown")
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

        cv2.putText(
            frame,
            "Depth Layers:",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        y_offset += 25

        for layer, count in layer_counts.items():
            cv2.putText(
                frame,
                f"  {layer}: {count}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            y_offset += 20

        # Display audio priorities
        audio_priorities = [
            d.get("spatial_audio", {}).audio_priority for d in detections
        ]
        if audio_priorities:
            avg_priority = np.mean(audio_priorities)
            cv2.putText(
                frame,
                f"Avg Audio Priority: {avg_priority:.1f}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            y_offset += 20

        # Display quality metrics
        quality_scores = [
            d.get("quality_metrics", {}).overall_confidence for d in detections
        ]
        if quality_scores:
            avg_quality = np.mean(quality_scores)
            cv2.putText(
                frame,
                f"Avg Quality: {avg_quality:.3f}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            y_offset += 20

        # Display individual detection details (limit to 3)
        for i, detection in enumerate(detections[:3]):
            class_name = detection["class"]
            depth_layer = detection.get("depth_layer", "unknown")
            audio_priority = detection.get("spatial_audio", {}).audio_priority
            quality = detection.get("quality_metrics", {}).overall_confidence

            text = f"#{i+1} {class_name}: {depth_layer}, P:{audio_priority}, Q:{quality:.2f}"
            cv2.putText(
                frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
            y_offset += 20

    def _calculate_fps(self) -> float:
        """Calculate current FPS."""
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.frame_count / elapsed
        return 0.0

    def run_demo(self):
        """Run the enhanced depth demo."""
        print("\nStarting Enhanced Depth Demo...")
        print("Features:")
        print("  - Advanced reference point selection")
        print("  - Temporal depth tracking")
        print("  - Quality assessment")
        print("  - Spatial audio metrics")
        print("  - Motion detection")
        print("\nControls:")
        print("  q - Quit")
        print("  s - Save snapshot")
        print("  r - Reset tracking")
        print("  d - Toggle detection display")
        print("  c - Show spatial context")

        cv2.namedWindow("Enhanced Depth Demo", cv2.WINDOW_NORMAL)

        show_detections = True
        show_context = False

        try:
            while True:
                # Get frame from camera
                frame = self.camera.get_frame()
                if frame is None:
                    print("Failed to read frame")
                    break

                # Simulate object detection
                detections = self._simulate_object_detection(frame)

                # Process detections with enhanced depth
                if show_detections:
                    processed_detections = self._process_detections_with_enhanced_depth(
                        frame, detections
                    )

                    # Draw enhanced detections
                    self._draw_enhanced_detections(frame, processed_detections)

                    # Display enhanced statistics
                    self._display_enhanced_stats(frame, processed_detections)

                # Show spatial context if requested
                if show_context and detections:
                    context = self.depth_processor.get_spatial_context(
                        frame, detections
                    )
                    self._display_spatial_context(frame, context)

                # Add FPS and info
                fps = self._calculate_fps()
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                # Display frame
                cv2.imshow("Enhanced Depth Demo", frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    self._save_snapshot(frame)
                elif key == ord("r"):
                    self.depth_processor.reset_tracking()
                    print("Enhanced tracking reset")
                elif key == ord("d"):
                    show_detections = not show_detections
                    print(f"Detection display: {'ON' if show_detections else 'OFF'}")
                elif key == ord("c"):
                    show_context = not show_context
                    print(f"Spatial context: {'ON' if show_context else 'OFF'}")

        except KeyboardInterrupt:
            print("Demo interrupted by user")

        finally:
            self.cleanup()

    def _display_spatial_context(self, frame: np.ndarray, context: Dict[str, Any]):
        """Display spatial context information."""
        y_offset = frame.shape[0] - 150

        # Scene type
        scene_type = context.get("scene_type", "unknown")
        cv2.putText(
            frame,
            f"Scene: {scene_type}",
            (frame.shape[1] - 200, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        y_offset += 20

        # Spatial density
        density = context.get("spatial_density", 0)
        cv2.putText(
            frame,
            f"Density: {density:.3f}",
            (frame.shape[1] - 200, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        y_offset += 20

        # Movement patterns
        movement_patterns = context.get("movement_patterns", {})
        approaching = len(movement_patterns.get("approaching_objects", []))
        receding = len(movement_patterns.get("receding_objects", []))

        cv2.putText(
            frame,
            f"Approaching: {approaching}",
            (frame.shape[1] - 200, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        y_offset += 20

        cv2.putText(
            frame,
            f"Receding: {receding}",
            (frame.shape[1] - 200, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    def _save_snapshot(self, frame: np.ndarray):
        """Save current frame as snapshot."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_snapshot_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f"Enhanced snapshot saved: {filename}")

    def cleanup(self):
        """Clean up resources."""
        self.camera.release()
        cv2.destroyAllWindows()
        print("Enhanced demo cleanup completed")


def main():
    """Main function."""
    print("=== Enhanced Depth Processing Demo ===")
    print("This demo shows advanced depth processing with:")
    print("  - Multi-layer reference point selection")
    print("  - Temporal depth tracking and motion detection")
    print("  - Comprehensive quality assessment")
    print("  - Spatial audio optimization")
    print("  - Context-aware depth analysis")
    print()

    # Ask user about phone camera
    use_phone = input("Try phone camera first? (y/n): ").lower().startswith("y")

    try:
        demo = EnhancedDepthDemo(use_phone_camera=use_phone)
        demo.run_demo()
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
