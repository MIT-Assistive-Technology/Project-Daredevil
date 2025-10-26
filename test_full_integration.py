#!/usr/bin/env python3
"""Quick test of full integration - Detection + Depth + Spatial Audio"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Will import integration module

print("=" * 60)
print("Project Daredevil - Full System Integration Test")
print("=" * 60)
print("\nThis will test:")
print("  1. Camera capture")
print("  2. Object detection (YOLO)")
print("  3. Depth estimation")
print("  4. Spatial audio positioning")
print("\nPress Ctrl+C to stop")
print("=" * 60)
print()

try:
    # Import and run integration
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'spatial-audio'))
    from integration import IntegratedSpatialAudioSystem
    
    system = IntegratedSpatialAudioSystem(
        target_classes=['person', 'bottle', 'cup'],
        master_volume=0.2,
        verbose=True
    )
    
    system.run()
    
except KeyboardInterrupt:
    print("\n\nStopped by user")
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
