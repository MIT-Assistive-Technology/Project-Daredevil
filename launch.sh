#!/bin/bash

# Project Daredevil - Launch Script (Configurable)
# Usage: ./launch.sh [--camera N] [--classes A B C] [--volume X] [--confidence Y]

set -e  # Exit on error

# Default values
CAMERA=1
CLASSES="person bottle"
VOLUME=0.1
CONFIDENCE=0.3

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --camera)
            CAMERA="$2"
            shift 2
            ;;
        --classes)
            shift
            CLASSES="$@"
            break
            ;;
        --volume)
            VOLUME="$2"
            shift 2
            ;;
        --confidence)
            CONFIDENCE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./launch.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --camera N          Camera index (default: 1)"
            echo "  --classes A B C     Target classes (default: person bottle)"
            echo "  --volume X          Master volume 0.0-1.0 (default: 0.1)"
            echo "  --confidence Y      Confidence threshold 0.0-1.0 (default: 0.3)"
            echo ""
            echo "Examples:"
            echo "  ./launch.sh"
            echo "  ./launch.sh --camera 0 --classes person bottle cup --volume 0.2"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# Check if we're in the project directory
if [ ! -d "env" ]; then
    echo "Error: Virtual environment not found. Please run this script from the Project-Daredevil directory."
    exit 1
fi

# Display banner
echo "============================================"
echo "  Project Daredevil - Starting System"
echo "============================================"
echo ""
print_info "Camera: $CAMERA"
print_info "Classes: $CLASSES"
print_info "Volume: $VOLUME"
print_info "Confidence: $CONFIDENCE"
echo ""
echo "Press Ctrl+C to stop"
echo "============================================"
echo ""

# Activate virtual environment and run
source env/bin/activate && python3 test_full_integration.py --camera "$CAMERA" --classes $CLASSES --volume "$VOLUME" --confidence "$CONFIDENCE"

