#!/bin/bash

# Quick wrapper to run detection depth integration
# Usage: ./run_detection_depth.sh [additional arguments]
source env/bin/activate && python3 detection_depth_integration.py "$@"

