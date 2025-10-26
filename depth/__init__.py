#!/usr/bin/env python3
"""
Depth module for monocular depth estimation.

This module provides depth estimation capabilities for the Project Daredevil system.
It can be used by detection and spatial audio pipelines to get depth information
from camera frames and bounding boxes.

Main components:
- DepthProcessor: Core depth estimation and processing
- DepthStream: Live depth streaming with visualization
- Utility functions for integration

Usage:
    from depth import DepthProcessor, create_depth_processor

    # Create processor
    processor = create_depth_processor()

    # Get depth from frame and bounding box
    result = processor.get_depth_for_spatial_audio(frame, bbox)
    depth_value = result['normalized_depth']
"""

from .depth_processor import DepthProcessor, create_depth_processor, process_frame_depth

from .depth_stream import HFDepthEstimator, DepthStream

__all__ = [
    "DepthProcessor",
    "create_depth_processor",
    "process_frame_depth",
    "HFDepthEstimator",
    "DepthStream",
]

__version__ = "1.0.0"
