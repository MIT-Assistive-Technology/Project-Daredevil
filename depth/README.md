# Depth Module Documentation

This module provides monocular depth estimation capabilities for the Project Daredevil system. It processes camera frames and bounding boxes to generate normalized depth values for spatial audio applications.

## Overview

The depth module consists of several components:

- **DepthProcessor**: Core depth estimation and processing
- **DepthStream**: Live depth streaming with visualization
- **Integration Tests**: Comprehensive testing suite
- **Example Integration**: Demo with camera and mock detection

## Installation

### Prerequisites

- Python 3.8+
- macOS (for Apple Silicon optimization)
- Camera access permissions

### Dependencies

```bash
# Activate virtual environment
source env/bin/activate

# Install required packages
pip install torch torchvision transformers opencv-python numpy ultralytics
```

## Quick Start

### Basic Usage

```bash
# Run depth processing with auto-detected camera
python run_depth.py

# Force laptop camera
python run_depth.py --laptop

# Force external camera
python run_depth.py --external

# Run tests
python run_depth.py --test

# List available cameras
python run_depth.py --list-cameras
```

### Individual Components

```bash
# Test depth processing
python depth/test_depth_integration.py

# Live depth streaming
python depth/depth_stream.py

# Integrated demo
python depth/example_integration.py
```

## API Reference

### DepthProcessor

Main class for depth processing operations.

#### Methods

- `estimate_depth(frame)`: Estimate depth map from camera frame
- `get_depth_from_bbox(depth_map, bbox, method)`: Extract depth value from bounding box
- `normalize_depth(depth_value, method)`: Normalize depth value to 0.0-1.0 range
- `get_depth_for_spatial_audio(frame, bbox, method, normalization)`: Complete processing pipeline

#### Parameters

- `method`: Depth calculation method ('mean', 'median', 'min', 'max')
- `normalization`: Normalization method ('relative', 'statistical', 'reference')

### DepthStream

Live depth streaming with real-time visualization.

#### Controls

- `q`: Quit application
- `s`: Save snapshot (if save directory specified)
- `r`: Reset depth statistics
- `b`: Toggle bounding box display

## Testing

### Integration Tests

The test suite verifies all core functionality:

```bash
python depth/test_depth_integration.py
```

**Test Coverage:**
- Depth processing with bounding boxes
- Multiple normalization methods
- Different depth calculation methods
- Integration interface
- Performance benchmarks

### Performance Tests

Expected performance metrics:
- Processing time: ~74ms per frame
- FPS: ~13.5 on Apple Silicon
- Memory usage: ~1GB for model

## Camera Configuration

### Built-in Laptop Camera

```bash
python run_depth.py --laptop
```

### External Camera (iPhone/USB)

```bash
python run_depth.py --external
```

**Requirements for iPhone:**
- iPhone signed into same Apple ID as Mac
- iPhone nearby and unlocked
- macOS Ventura or later

### Specific Camera Index

```bash
python run_depth.py --camera 1
```

## Coordinate System

The module uses a universal coordinate system:

- **Origin**: Top-left corner (0, 0)
- **X-axis**: Left to right (0 to frame_width)
- **Y-axis**: Top to bottom (0 to frame_height)
- **Bounding Box Format**: [x1, y1, x2, y2]

## Integration Examples

### With Object Detection

```python
from depth.depth_processor import create_depth_processor

# Initialize processor
processor = create_depth_processor()

# Process detection results
for detection in detections:
    bbox = detection['bbox']
    result = processor.get_depth_for_spatial_audio(frame, bbox)
    
    # Add depth to detection
    detection['depth'] = result['normalized_depth']
```

### With Spatial Audio

```python
# Get depth for spatial audio
result = processor.get_depth_for_spatial_audio(frame, bbox)

# Use normalized depth (0.0-1.0 range)
spatial_audio.set_depth(result['normalized_depth'])
```

## Configuration Options

### Depth Calculation Methods

- **mean**: Average depth in ROI
- **median**: Median depth in ROI (recommended)
- **min**: Minimum depth in ROI
- **max**: Maximum depth in ROI

### Normalization Methods

- **relative**: Normalize based on current depth statistics
- **statistical**: Z-score based normalization
- **reference**: Normalize relative to reference depth

### Camera Settings

```python
from camera.index import CameraStream

camera = CameraStream(
    camera_index=1,
    width=1280,
    height=720,
    fps=30
)
```

## Troubleshooting

### Camera Issues

**Problem**: "Could not open camera"
**Solution**: 
1. Check camera permissions in System Preferences
2. Try different camera indices
3. Restart camera applications

**Problem**: External camera not detected
**Solution**:
1. Ensure iPhone is signed into same Apple ID
2. Check USB/WiFi connection
3. Verify iPhone is nearby and unlocked

### Depth Processing Issues

**Problem**: Model loading fails
**Solution**:
1. Check internet connection for first download
2. Verify sufficient disk space (~1GB)
3. Check PyTorch installation

**Problem**: Low performance
**Solution**:
1. Use smaller frame sizes
2. Process only necessary bounding boxes
3. Use median method for depth calculation

### Permission Issues

**Problem**: Camera access denied
**Solution**:
1. Go to System Preferences > Security & Privacy > Camera
2. Enable camera access for Terminal/Python
3. Restart the application

## File Structure

```
depth/
├── __init__.py              # Module exports
├── depth_processor.py       # Core depth processing
├── depth_stream.py          # Live streaming
├── test_depth_integration.py # Integration tests
├── example_integration.py   # Demo with camera
└── README.md               # This file
```

## Performance Optimization

### For Real-time Applications

1. **Use median method**: Fastest depth calculation
2. **Process only necessary bounding boxes**: Reduce computation
3. **Use smaller frame sizes**: Reduce processing time
4. **Enable Apple Silicon acceleration**: Automatic MPS detection

### Memory Management

- Model loads once and reuses for all frames
- Process frames one at a time to avoid memory buildup
- Use CPU instead of GPU if memory constrained

## Development

### Adding New Features

1. **New depth models**: Extend `DepthProcessor` class
2. **New normalization methods**: Add to `normalize_depth` method
3. **New visualization**: Modify `DepthStream` class
4. **New tests**: Add to `test_depth_integration.py`

### Code Standards

- Follow PEP 8 style guidelines
- Use type hints for all function parameters
- Include comprehensive docstrings
- Add unit tests for new functionality

## Support

For issues or questions:

1. Check the troubleshooting section
2. Run integration tests to verify functionality
3. Check camera permissions and connections
4. Verify all dependencies are installed

## License

See LICENSE file for details.