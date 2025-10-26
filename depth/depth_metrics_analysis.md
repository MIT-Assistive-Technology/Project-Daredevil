# Depth Metrics and Reference Point Analysis for Project Daredevil

## Project Context and Goals

Project Daredevil aims to create **spatial audio blind assistance** by translating depth perception into sound—like digital echolocation. The key goals are:

1. **Real-time spatial awareness** for blind/low-vision users
2. **Affordable, portable solution** using consumer devices
3. **Subtle, continuous audio cues** (ambient whooshes, localized pitch shifts)
4. **Detection of key social and safety cues**:
   - Approaching handshakes
   - Objects moving into one's path
   - "The last 10 feet" problem
   - Ambient depth shifts in hallways/open spaces

## Current Depth Implementation Analysis

### Existing Metrics
The current `DepthProcessor` tracks basic statistics:
- `min`, `max`, `mean`, `std` depth values
- ROI-based depth calculation (mean, median, min, max)
- Three normalization methods: relative, statistical, reference

### Current Limitations
1. **No temporal consistency** - depth values fluctuate between frames
2. **Poor reference point selection** - uses simple min/max normalization
3. **No depth quality assessment** - doesn't validate depth reliability
4. **Limited spatial context** - doesn't consider object relationships
5. **No motion tracking** - can't handle moving objects effectively

## Recommended Depth Metrics for Spatial Audio

### 1. Primary Depth Metrics

#### **Object-Centric Depth Values**
```python
object_depth_metrics = {
    'primary_depth': float,        # Most reliable depth estimate
    'depth_confidence': float,     # 0.0-1.0 confidence in depth reading
    'depth_stability': float,      # Temporal consistency over N frames
    'spatial_coherence': float,    # Consistency across object surface
}
```

#### **Spatial Context Metrics**
```python
spatial_context = {
    'relative_to_user': float,      # Distance from camera/user
    'relative_to_background': float, # Distance from background reference
    'relative_to_other_objects': dict, # Distance relationships
    'depth_layer': int,            # Foreground (1), Midground (2), Background (3)
}
```

#### **Motion and Temporal Metrics**
```python
temporal_metrics = {
    'depth_velocity': float,       # Rate of depth change (m/s)
    'depth_acceleration': float,   # Acceleration of depth change
    'trajectory_prediction': dict,  # Predicted future positions
    'motion_consistency': float,   # Consistency of motion pattern
}
```

### 2. Reference Point Selection Strategies

#### **Multi-Layer Reference System**

1. **User Reference (Camera Position)**
   - Use camera height and viewing angle
   - Establish "personal space" boundaries (0.5m, 1.5m, 3m)
   - Critical for spatial audio positioning

2. **Background Reference**
   - Sample corners and edges of frame
   - Use median depth of background areas
   - Update slowly to avoid noise
   - Essential for relative depth calculations

3. **Scene-Specific References**
   - **Indoor**: Floor level, wall distances, furniture heights
   - **Outdoor**: Ground plane, horizon line, building facades
   - **Social**: Typical conversation distances (1-2m)

4. **Dynamic Reference Points**
   - Moving objects that maintain consistent depth
   - Objects with known sizes (standardized items)
   - Previously validated depth measurements

#### **Reference Point Quality Assessment**
```python
reference_quality = {
    'stability_score': float,      # How stable the reference is
    'confidence_score': float,    # How reliable the reference is
    'temporal_consistency': float, # Consistency over time
    'spatial_coverage': float,    # How much of scene it covers
}
```

### 3. Depth Quality and Validation Metrics

#### **Depth Reliability Indicators**
```python
depth_quality = {
    'signal_to_noise_ratio': float,    # Depth map quality
    'edge_sharpness': float,           # Object boundary clarity
    'occlusion_detection': bool,       # Is object partially occluded?
    'illumination_quality': float,     # Lighting conditions
    'texture_richness': float,         # Surface detail availability
}
```

#### **Validation Metrics**
```python
validation_metrics = {
    'depth_consistency': float,        # Internal consistency checks
    'cross_frame_validation': float,   # Consistency across frames
    'geometric_validation': float,     # Geometric plausibility
    'temporal_smoothness': float,      # Smoothness over time
}
```

### 4. Spatial Audio-Specific Metrics

#### **Audio Positioning Metrics**
```python
audio_positioning = {
    'azimuth_angle': float,           # Left-right position (-180° to +180°)
    'elevation_angle': float,         # Up-down position (-90° to +90°)
    'distance_category': str,         # 'very_close', 'close', 'medium', 'far'
    'audio_priority': int,            # Priority for audio rendering (1-10)
    'spatial_uncertainty': float,     # Uncertainty in spatial position
}
```

#### **Audio Cue Parameters**
```python
audio_cues = {
    'proximity_warning': bool,        # Should trigger proximity alert?
    'movement_direction': str,        # 'approaching', 'receding', 'lateral'
    'movement_speed': str,            # 'slow', 'medium', 'fast'
    'audio_intensity': float,         # Volume/intensity (0.0-1.0)
    'audio_frequency': float,        # Pitch/frequency modulation
}
```

## Implementation Recommendations

### 1. Enhanced Depth Processor

Create a new `EnhancedDepthProcessor` class with:

```python
class EnhancedDepthProcessor(DepthProcessor):
    def __init__(self):
        super().__init__()
        self.reference_manager = ReferencePointManager()
        self.temporal_tracker = TemporalDepthTracker()
        self.quality_assessor = DepthQualityAssessor()
        self.spatial_analyzer = SpatialContextAnalyzer()
    
    def get_enhanced_depth_for_spatial_audio(self, frame, bbox, object_id=None):
        """Get comprehensive depth information optimized for spatial audio."""
        # Implementation details...
```

### 2. Reference Point Manager

```python
class ReferencePointManager:
    def __init__(self):
        self.user_reference = None
        self.background_reference = None
        self.scene_references = {}
        self.dynamic_references = []
    
    def update_references(self, depth_map, detections):
        """Update all reference points based on current scene."""
        # Implementation details...
    
    def get_best_reference_for_object(self, object_depth, object_class):
        """Get the most appropriate reference for an object."""
        # Implementation details...
```

### 3. Temporal Depth Tracker

```python
class TemporalDepthTracker:
    def __init__(self, max_history=30):
        self.depth_history = {}
        self.motion_tracker = {}
    
    def track_object_depth(self, object_id, depth_value, timestamp):
        """Track depth changes over time for an object."""
        # Implementation details...
    
    def predict_future_depth(self, object_id, time_horizon=0.5):
        """Predict future depth for motion compensation."""
        # Implementation details...
```

### 4. Spatial Context Analyzer

```python
class SpatialContextAnalyzer:
    def analyze_spatial_relationships(self, detections, depth_map):
        """Analyze spatial relationships between objects."""
        # Implementation details...
    
    def determine_depth_layer(self, object_depth, scene_context):
        """Determine if object is foreground, midground, or background."""
        # Implementation details...
```

## Key Metrics for Spatial Audio Success

### Critical Success Metrics

1. **Depth Accuracy**: ±10cm for objects within 3m
2. **Temporal Stability**: <5% variation over 1 second
3. **Spatial Resolution**: 5° angular resolution for audio positioning
4. **Latency**: <100ms total processing time
5. **Reliability**: >95% valid depth readings in good conditions

### Audio-Specific Requirements

1. **Distance Categories**:
   - Very Close (0-0.5m): High priority, urgent audio cues
   - Close (0.5-1.5m): Medium priority, proximity warnings
   - Medium (1.5-3m): Normal priority, spatial awareness
   - Far (3m+): Low priority, ambient awareness

2. **Movement Detection**:
   - Approaching: Increasing audio intensity
   - Receding: Decreasing audio intensity
   - Lateral: Panning audio cues
   - Stationary: Ambient spatial positioning

3. **Audio Quality Metrics**:
   - Spatial accuracy: ±15° for azimuth, ±10° for elevation
   - Depth resolution: 0.1m increments for close objects
   - Temporal smoothness: No audio artifacts from depth jumps

## Implementation Priority

### Phase 1: Core Improvements
1. Implement background reference estimation
2. Add temporal depth tracking
3. Create depth quality assessment
4. Implement spatial context analysis

### Phase 2: Advanced Features
1. Multi-object relationship tracking
2. Motion prediction and compensation
3. Scene-specific reference adaptation
4. Advanced audio positioning algorithms

### Phase 3: Optimization
1. Real-time performance optimization
2. Adaptive quality vs. speed tradeoffs
3. Machine learning-based depth validation
4. User-specific calibration and adaptation

This comprehensive approach will provide the robust depth sensing foundation needed for effective spatial audio feedback in Project Daredevil.
