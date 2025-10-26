# Project Daredevil 
**Spatial Audio Blind Assistance**

Project Daredevil explores how consumer devices (such as iPhone, AirPods, webcams) can be used to provide affordable/real-time spatial audio feedback to blind and low-vision users. Our goal is to create a proof-of-concept system that translates depth perception into sound—like digital echolocation—to enhance spatial awareness in everyday environments.

---

## Overview
With our codesigners Eric and Scott, we are working to unlock the capabilities of everyday devices—like smartphones and headphones—to provide accessible, real-time assistance for blind users.

**Our approach:**
1. Detect objects with computer vision (i.e., water bottle, glass of water).
2. Estimate depth using monocular depth models, using the object detection to clean up the signal.
3. Stream continuous spatial audio cues that adapt as the user moves.
4. Work toward an iOS-based system (LiDAR, ARKit, AirPods spatial audio), or an SDK version for AR glasses.

---

## Motivations
- Make spatial awareness assistance affordable and portable!
- Provide subtle, continuous cues (such as ambient whooshes, localized pitch shifts) instead of overwhelming object-to-sound mappings.
- Enable detection of key social and safety cues such as:
  - An approaching handshake.
  - Objects moving into one’s path.
  - "The last 10 feet" problem.
  - Ambient depth shifts in hallways or open spaces.

---

## Current Semester Workflow
1. **Object Detection** – YOLO with COCO dataset.
2. **Depth Estimation** – Explore monocular CV models (MiDaS, DPT) in Python.
3. **Spatial Audio Rendering** – Experiment with libraries such as [spaudiopy](https://github.com/chris-hld/spaudiopy).
4. **Integration** – Prototype real-time depth-to-audio pipeline.

---

Tasks and issues are tracked in our [GitHub Project Board](https://github.com/orgs/MIT-Assistive-Technology/projects/1/views/1).

---

## Goals
We’ll consider this semester’s prototype successful if:
- A webcam-based MVP can detect objects, estimate depth, and produce directional audio feedback in real time.
- The system demonstrates **clear, intuitive audio depth cues** for our codesigners.
Therefore, most of our effort will lay in understanding how to best implement the feedback system.

**Stretch Goals:**
- Add spoken triggers or voice commands.
- Move toward iOS app development (LiDAR + ARKit + AirPods).
- Explore continuous ambient spatial audio for full-scene awareness.

---

## Limitations
- **User safety** – Audio cues must not interfere with natural hearing or overwhelm.
- **Tech limits** – Webcams lack 360° coverage, unlike high-end LiDAR rigs or more complicated setups.
- **Training** – Users may need adaptation time to interpret depth cues, especially if they are relative depths.

---

*MIT Assistive Technology Club — Fall 2025*
