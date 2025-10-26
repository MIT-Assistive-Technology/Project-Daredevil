#!/usr/bin/env python3
"""
Simple configuration and launcher for Project Daredevil depth processing.

This script provides easy commands to run the depth processing system
with different camera configurations.

Usage:
    python run_depth.py                    # Auto-detect best camera
    python run_depth.py --laptop           # Force laptop camera
    python run_depth.py --external         # Force external camera
    python run_depth.py --test             # Run tests only
    python run_depth.py --demo             # Run integrated demo
"""

import argparse
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from camera.index import (
    CameraStream,
    create_phone_camera_stream,
    list_available_cameras,
)
from depth.depth_processor import create_depth_processor


def run_depth_stream(camera_index=None, use_external=False):
    """Run the depth streaming with specified camera configuration."""
    print("Starting Project Daredevil Depth Processing")
    print("=" * 50)
    
    # Import here to avoid issues if modules aren't available
    from depth.depth_stream import DepthStream
    
    if camera_index is not None:
        print(f"Using camera index: {camera_index}")
        stream = DepthStream(camera_index=camera_index)
    elif use_external:
        print("Looking for external camera (phone/USB)...")
        external_camera = create_phone_camera_stream()
        if external_camera:
            print("External camera found!")
            stream = DepthStream(camera_index=external_camera.camera_index)
        else:
            print("No external camera found, falling back to laptop camera")
            stream = DepthStream(camera_index=0)
    else:
        print("Using laptop camera")
        stream = DepthStream(camera_index=0)
    
    print("\nControls:")
    print("  q - Quit")
    print("  s - Save snapshot")
    print("  r - Reset depth statistics")
    print("  b - Toggle bounding box display")
    print("\nStarting depth stream...")
    
    try:
        stream.start()
    except KeyboardInterrupt:
        print("\nDepth stream stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        print("Try running with --test to check your setup")


def run_integrated_demo(use_external=False):
    """Run the integrated demo with camera + depth + mock detection."""
    print("Starting Project Daredevil Integrated Demo")
    print("=" * 50)
    
    # Import here to avoid issues if modules aren't available
    from depth.example_integration import IntegratedDepthDemo
    
    try:
        demo = IntegratedDepthDemo(use_phone_camera=use_external)
        demo.run_demo()
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        print("Try running with --test to check your setup")


def run_tests():
    """Run the depth integration tests."""
    print("Running Project Daredevil Tests")
    print("=" * 50)
    
    try:
        # Import and run the test
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                os.path.join(
                    os.path.dirname(__file__), "depth", "test_depth_integration.py"
                ),
            ],
            capture_output=True,
            text=True,
        )

        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)

        if result.returncode == 0:
            print("All tests passed!")
        else:
            print("Some tests failed")

    except Exception as e:
        print(f"Error running tests: {e}")


def list_cameras():
    """List all available cameras."""
    print("Available Cameras")
    print("=" * 30)
    
    cameras = list_available_cameras()
    if cameras:
        for cam in cameras:
            print(f"  Camera {cam}")
    else:
        print("  No cameras detected")
    
    print("\nCamera Usage:")
    print("  Camera 0: Usually laptop built-in camera")
    print("  Camera 1+: External cameras (phones, USB cameras)")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Project Daredevil Depth Processing Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_depth.py                    # Auto-detect best camera
  python run_depth.py --laptop           # Force laptop camera
  python run_depth.py --external         # Force external camera
  python run_depth.py --test             # Run tests only
  python run_depth.py --demo             # Run integrated demo
  python run_depth.py --list-cameras     # List available cameras
  python run_depth.py --camera 1         # Use specific camera index
        """,
    )

    parser.add_argument(
        "--laptop", action="store_true", help="Force use of laptop camera (index 0)"
    )
    parser.add_argument(
        "--external",
        action="store_true",
        help="Force use of external camera (phone/USB)",
    )
    parser.add_argument("--camera", type=int, help="Use specific camera index")
    parser.add_argument(
        "--test", action="store_true", help="Run integration tests only"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run integrated demo (camera + depth + detection)",
    )
    parser.add_argument(
        "--list-cameras", action="store_true", help="List all available cameras"
    )

    args = parser.parse_args()

    # Handle different modes
    if args.list_cameras:
        list_cameras()
    elif args.test:
        run_tests()
    elif args.demo:
        run_integrated_demo(use_external=args.external)
    else:
        # Determine camera configuration
        if args.camera is not None:
            run_depth_stream(camera_index=args.camera)
        elif args.laptop:
            run_depth_stream(camera_index=0)
        elif args.external:
            run_depth_stream(use_external=True)
        else:
            # Auto-detect: try external first, fallback to laptop
            run_depth_stream(use_external=True)


if __name__ == "__main__":
    main()
