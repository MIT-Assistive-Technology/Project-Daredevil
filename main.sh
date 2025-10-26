#!/bin/bash

# Project Daredevil - Main Launch Script
# This script runs the full integration test with predefined settings

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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
print_info "Camera: 1 (iPhone Continuity Camera)"
print_info "Classes: person, bottle"
print_info "Volume: 0.1"
print_info "Confidence: 0.3"
echo ""
echo "Press Ctrl+C to stop"
echo "============================================"
echo ""

# Activate virtual environment and run
source env/bin/activate && python3 test_full_integration.py --camera 1 --classes person bottle --volume 0.1 --confidence 0.3

