# Project Daredevil

A computer vision and spatial audio system that combines object detection, monocular depth estimation, and spatial audio processing to create an immersive audio experience based on detected objects in the camera view.

## Overview

Project Daredevil consists of four main modules:

1. **Camera Module** - Handles video streaming from various sources including phone cameras
2. **Detection Module** - Object detection and tracking (YOLO-based)
3. **Depth Module** - Monocular depth estimation using Hugging Face DPT models
4. **Spatial Audio Module** - Converts object positions and depths to spatial audio

## Features

- **Phone Camera Support**: Stream from iPhone/Android via various methods
- **Real-time Object Detection**: Live tracking of objects (focused on water bottles)
- **Monocular Depth Estimation**: Depth estimation without depth sensors
- **Spatial Audio**: Stereo panning based on object position and depth
- **Modular Design**: Each component can be used independently
- **Apple Silicon Optimized**: Uses MPS acceleration for depth processing

## Installation

### Prerequisites

- Python 3.8+
- macOS (for Apple Silicon optimization)
- Camera access permissions

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Project-Daredevil
```

2. Create and activate virtual environment:
```bash
python3 -m venv env
source env/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install torch torchvision transformers opencv-python numpy ultralytics
```

## Phone Camera Setup

### Method 1: EpocCam (Recommended)

1. **Install EpocCam**:
   - Download EpocCam from the App Store (iPhone) or Google Play (Android)
   - Install EpocCam Server on your Mac from [https://www.elgato.com/epoccam](https://www.elgato.com/epoccam)

2. **Setup**:
   - Connect iPhone to Mac via USB or WiFi
   - Open EpocCam on iPhone
   - EpocCam will appear as a camera device (usually index 1 or 2)

3. **Test**:
```bash
python camera/index.py
```

### Method 2: DroidCam (Android)

1. **Install DroidCam**:
   - Download DroidCam from Google Play Store
   - Install DroidCam Server on Mac from [https://droidcam.com](https://droidcam.com)

2. **Setup**:
   - Connect via USB or WiFi
   - DroidCam will appear as camera device

### Method 3: IP Webcam (Android)

1. **Install IP Webcam**:
   - Download IP Webcam from Google Play Store

2. **Setup**:
   - Start IP Webcam on phone
   - Note the IP address (e.g., 192.168.1.100:8080)
   - Use IP camera streaming in the camera module

### Method 4: Continuity Camera (iPhone)

1. **Enable Continuity Camera**:
   - iPhone must be signed into same Apple ID as Mac
   - iPhone must be nearby and unlocked

2. **Usage**:
   - Continuity Camera appears as camera index 1+
   - Automatically detected by the camera module

## Quick Start Commands

### Run Everything (Recommended)
```bash
# Auto-detect best camera (tries external first, falls back to laptop)
python run_depth.py

# Force laptop camera
python run_depth.py --laptop

# Force external camera (phone/USB)
python run_depth.py --external
```

### Test Everything Works
```bash
# Run all tests
python run_depth.py --test

# List available cameras
python run_depth.py --list-cameras

# Run integrated demo
python run_depth.py --demo
```

### Individual Components
```bash
# Test depth processing only
python depth/test_depth_integration.py

# Live depth streaming
python depth/depth_stream.py

# Integrated demo
python depth/example_integration.py
```

## Module Documentation

### Camera Module (`camera/`)

Handles video streaming from various sources.

**Key Features**:
- Auto-detection of available cameras
- Phone camera support via EpocCam, DroidCam, Continuity Camera
- IP camera streaming
- Unified interface for all camera types

**Usage**:
```python
from camera import CameraStream, create_phone_camera_stream

# Auto-detect phone camera
camera = create_phone_camera_stream()

# Or use specific camera
camera = CameraStream(camera_index=1)

# Get frame
frame = camera.get_frame()
```

### Depth Module (`depth/`)

Monocular depth estimation using Hugging Face DPT models.

**Key Features**:
- Offline-capable depth estimation
- Bounding box ROI processing
- Multiple normalization methods
- Apple Silicon acceleration
- Integration-ready interface

**Usage**:
```python
from depth import DepthProcessor, create_depth_processor

processor = create_depth_processor()
result = processor.get_depth_for_spatial_audio(frame, bbox)
depth_value = result['normalized_depth']  # 0.0 to 1.0
```

### Detection Module (`detection/`)

Object detection and tracking (YOLO-based).

**Status**: In development

**Focus**: Water bottle detection for demo purposes

### Spatial Audio Module (`spatial-audio/`)

Converts object positions and depths to spatial audio.

**Status**: In development

**Features**: Stereo panning, depth-based audio positioning

## Coordinate System

The system uses a universal coordinate system:

- **Origin**: Top-left corner (0, 0)
- **X-axis**: Left to right (0 to frame_width)
- **Y-axis**: Top to bottom (0 to frame_height)
- **Bounding Box Format**: [x1, y1, x2, y2]
- **Depth Values**: Normalized to 0.0-1.0 range

## Performance

- **Depth Processing**: ~12.8 FPS on Apple Silicon
- **Camera Streaming**: 30 FPS
- **Memory Usage**: ~1GB for depth model
- **Latency**: ~78ms per frame for depth processing

## Troubleshooting

### Camera Issues

**Problem**: "Could not open camera"
**Solution**: 
1. Check camera permissions in System Preferences
2. Try different camera indices (0, 1, 2)
3. Restart camera applications

**Problem**: Phone camera not detected
**Solution**:
1. Ensure EpocCam/DroidCam is running on phone
2. Check USB/WiFi connection
3. Try IP camera method

### Depth Processing Issues

**Problem**: Model loading fails
**Solution**:
1. Check internet connection for first download
2. Verify sufficient disk space
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

## Development

### Adding New Features

1. **New Camera Source**: Add to `camera/index.py`
2. **New Depth Model**: Extend `depth/depth_processor.py`
3. **New Detection Class**: Modify detection module
4. **New Audio Effect**: Extend spatial audio module

### Testing

Run integration tests:
```bash
python depth/test_depth_integration.py
```

### Code Structure

```
Project-Daredevil/
├── camera/           # Camera streaming module
├── detection/        # Object detection module
├── depth/           # Depth estimation module
├── spatial-audio/   # Spatial audio module
├── env/             # Virtual environment
└── README.md        # This file
```

## Acknowledgments

- Hugging Face for the DPT depth estimation models
- Ultralytics for YOLO object detection
- OpenCV for computer vision utilities
- Apple for Continuity Camera support
# Project Daredevil

**Spatial Audio Blind Assistance**

Project Daredevil explores how consumer devices (such as iPhone, AirPods, webcams) can be used to provide affordable/real-time spatial audio feedback to blind and low-vision users. Our goal is to create a proof-of-concept system that translates depth perception into sound—like digital echolocation—to enhance spatial awareness in everyday environments.

## Project Overview

Project Daredevil consists of four main modules:

1. **Camera Module** - Handles video streaming from various sources including phone cameras
2. **Detection Module** - Object detection and tracking (YOLO-based)
3. **Depth Module** - Monocular depth estimation using Hugging Face DPT models
4. **Spatial Audio Module** - Converts object positions and depths to spatial audio

## Project Motivations 

- Make spatial awareness assistance affordable and portable!
- Provide subtle, continuous cues (such as ambient whooshes, localized pitch shifts) instead of overwhelming object-to-sound mappings.
- Enable detection of key social and safety cues such as:
  - An approaching handshake
  - Objects moving into one's path
  - "The last 10 feet" problem
  - Ambient depth shifts in hallways or open spaces

## External Camera Options

Four ways to use a phone as your camera:

1. **DroidCam (Android)**
   - Install DroidCam from Play Store
   - Get DroidCam Server from [droidcam.com](https://droidcam.com)
   - Connect via USB/WiFi
   - Shows up as camera device

2. **IP Webcam (Android)**
   - Install IP Webcam from Play Store 
   - Run on phone to get IP (e.g., 192.168.1.100:8080)
   - Configure in camera module settings

3. **Continuity Camera (iPhone)**
   - Uses Apple Continuity feature
   - iPhone + Mac need same Apple ID
   - Phone must be nearby and unlocked
   - Auto-detected as camera index 1+

## Repository Structure

```bash
Project-Daredevil/
├── camera/           # Camera streaming module
├── detection/        # Object detection & tracking
├── depth/           # Depth estimation module
├── spatial-audio/   # Spatial audio processing
├── env/             # Virtual environment
└── README.md        # This file
```

## Success Criteria

Our prototype should demonstrate:

- Real-time object detection, depth estimation and directional audio
- Clear, intuitive audio depth cues for our codesigners
- Smooth integration of all system components

Future goals include:

- Voice command integration
- iOS native app (LiDAR + ARKit + AirPods)
- Continuous ambient spatial audio

## Current Limitations

- **Safety**: Audio feedback must not interfere with natural hearing
- **Hardware**: Standard webcams have limited field of view
- **Learning Curve**: Users need time to interpret depth-based audio cues

## Development

Project progress is tracked in our [GitHub Project Board](https://github.com/orgs/MIT-Assistive-Technology/projects/1/views/1).

## Credits

- Hugging Face - DPT depth estimation models
- Ultralytics - YOLO object detection
- OpenCV - Computer vision framework
- Apple - Continuity Camera API

---

## About

MIT Assistive Technology Club
Fall 2025 Project
