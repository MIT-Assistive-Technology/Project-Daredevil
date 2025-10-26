#!/usr/bin/env python3
"""
Example integration of depth module with camera and detection modules.

This example shows how to:
1. Stream video from camera (including phone camera)
2. Process frames with depth estimation
3. Integrate with object detection bounding boxes
4. Provide depth information for spatial audio

Usage:
    python depth/example_integration.py
"""

import cv2
import numpy as np
import time
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera import CameraStream, create_phone_camera_stream
from depth import DepthProcessor, create_depth_processor


class IntegratedDepthDemo:
    """
    Demonstration of integrated depth processing with camera and detection.
    """

    def __init__(self, use_phone_camera: bool = True):
        """
        Initialize the integrated demo.

        Args:
            use_phone_camera: Whether to try using phone camera first
        """
        self.use_phone_camera = use_phone_camera

        # Initialize camera
        self.camera = self._initialize_camera()

        # Initialize depth processor
        self.depth_processor = create_depth_processor()

        # Simulated object detection results (water bottles)
        self.detection_results = []

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()

        print("Integrated Depth Demo initialized")
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
        from camera import CameraStream

        return CameraStream()

    def _simulate_object_detection(self, frame: np.ndarray) -> list:
        """
        Simulate object detection results for water bottles.

        In a real implementation, this would call the actual object detection module.
        """
        # For demo purposes, create some mock bounding boxes
        # In reality, these would come from YOLO or similar detection
        h, w = frame.shape[:2]

        # Create mock detections (water bottles)
        mock_detections = [
            {
                "bbox": [w // 4, h // 4, w // 4 + 100, h // 4 + 150],
                "class": "water bottle",
                "confidence": 0.95,
            },
            {
                "bbox": [3 * w // 4 - 100, h // 2, 3 * w // 4, h // 2 + 150],
                "class": "water bottle",
                "confidence": 0.87,
            },
        ]

        return mock_detections

    def _process_detections_with_depth(
        self, frame: np.ndarray, detections: list
    ) -> list:
        """Process detections with depth information."""
        processed_detections = []

        for detection in detections:
            bbox = detection["bbox"]

            # Get depth information for this bounding box
            depth_result = self.depth_processor.get_depth_for_spatial_audio(
                frame, bbox, method="median", normalization="relative"
            )

            # Add depth information to detection
            processed_detection = detection.copy()
            processed_detection.update(
                {
                    "raw_depth": depth_result["raw_depth"],
                    "normalized_depth": depth_result["normalized_depth"],
                    "roi_stats": depth_result["roi_stats"],
                }
            )

            processed_detections.append(processed_detection)

        return processed_detections

    def _draw_detections_with_depth(self, frame: np.ndarray, detections: list):
        """Draw detections with depth information on frame."""
        for i, detection in enumerate(detections):
            bbox = detection["bbox"]
            x1, y1, x2, y2 = bbox

            # Draw bounding box
            color = (0, 255, 0) if i == 0 else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw labels with depth info
            label = f"{detection['class']}: {detection['confidence']:.2f}"
            depth_label = f"Depth: {detection['normalized_depth']:.3f}"

            cv2.putText(
                frame, label, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
            cv2.putText(
                frame,
                depth_label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    def _calculate_fps(self) -> float:
        """Calculate current FPS."""
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.frame_count / elapsed
        return 0.0

    def run_demo(self):
        """Run the integrated depth demo."""
        print("\nStarting integrated depth demo...")
        print("Controls:")
        print("  q - Quit")
        print("  s - Save snapshot")
        print("  r - Reset depth statistics")
        print("  d - Toggle detection display")

        cv2.namedWindow("Integrated Depth Demo", cv2.WINDOW_NORMAL)

        show_detections = True

        try:
            while True:
                # Get frame from camera
                frame = self.camera.get_frame()
                if frame is None:
                    print("Failed to read frame")
                    break

                # Simulate object detection
                detections = self._simulate_object_detection(frame)

                # Process detections with depth
                if show_detections:
                    processed_detections = self._process_detections_with_depth(
                        frame, detections
                    )

                    # Draw detections with depth info
                    self._draw_detections_with_depth(frame, processed_detections)

                    # Display depth statistics
                    y_offset = 30
                    for i, detection in enumerate(processed_detections):
                        if i >= 3:  # Limit display
                            break

                        text = f"Object {i+1}: Raw={detection['raw_depth']:.1f}, Norm={detection['normalized_depth']:.3f}"
                        cv2.putText(
                            frame,
                            text,
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )
                        y_offset += 25

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
                cv2.imshow("Integrated Depth Demo", frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    self._save_snapshot(frame)
                elif key == ord("r"):
                    self.depth_processor.reset_depth_stats()
                    print("Depth statistics reset")
                elif key == ord("d"):
                    show_detections = not show_detections
                    print(f"Detection display: {'ON' if show_detections else 'OFF'}")

        except KeyboardInterrupt:
            print("Demo interrupted by user")

        finally:
            self.cleanup()

    def _save_snapshot(self, frame: np.ndarray):
        """Save current frame as snapshot."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f"Snapshot saved: {filename}")

    def cleanup(self):
        """Clean up resources."""
        self.camera.release()
        cv2.destroyAllWindows()
        print("Demo cleanup completed")


def main():
    """Main function."""
    print("=== Integrated Depth Demo ===")
    print("This demo shows depth processing integrated with camera and detection")
    print()

    # Ask user about phone camera
    use_phone = input("Try phone camera first? (y/n): ").lower().startswith("y")

    try:
        demo = IntegratedDepthDemo(use_phone_camera=use_phone)
        demo.run_demo()
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
