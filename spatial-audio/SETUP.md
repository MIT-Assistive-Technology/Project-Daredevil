# Spatial Audio Setup Guide

## Installation Steps

### 1. Activate Virtual Environment

**IMPORTANT**: Always activate the virtual environment first!

```bash
cd /Users/alisonsoong/Desktop/Project-Daredevil
source env/bin/activate
```

You should see `(env)` in your terminal prompt.

### 2. Install OpenAL (macOS)

```bash
brew install openal-soft
```

### 3. Verify Installation

```bash
# With virtual environment activated
which python3  # Should point to env/bin/python3
python3 -c "import openal; print('PyOpenAL OK')"
```

## Running Spatial Audio

### Test Spatial Audio Engine

```bash
# Make sure virtual environment is activated!
source env/bin/activate

# Run test
python3 spatial-audio/index.py
```

### Run Full Integration

```bash
# Make sure virtual environment is activated!
source env/bin/activate

# Run with default settings
python3 spatial-audio/integration.py

# With options
python3 spatial-audio/integration.py --classes person bottle cup --volume 0.3
```

## Troubleshooting

### Error: "No module named 'openal'"

**Problem**: Virtual environment is not activated or PyOpenAL not installed in venv.

**Solution**:
```bash
# Activate virtual environment
source env/bin/activate

# Install PyOpenAL in the venv
pip install PyOpenAL

# Verify
python3 -c "import openal; print('OK')"
```

### Error: "OpenAL library couldn't be found"

**Problem**: OpenAL-Soft not installed or library path not set.

**Solution**:
```bash
# Install OpenAL-Soft
brew install openal-soft

# Verify installation
brew list openal-soft

# Create symlink (if needed, with sudo)
sudo ln -sf /opt/homebrew/Cellar/openal-soft/*/lib/libopenal.*.dylib /opt/homebrew/lib/libopenal.dylib
```

### Error: "Failed to open audio device"

**Problem**: Audio device is in use or AirPods not connected.

**Solution**:
1. Make sure no other applications are using audio
2. Connect AirPods to your Mac
3. Check audio output in System Settings
4. Restart coreaudiod:
   ```bash
   sudo killall coreaudiod
   ```

### PyOpenAL Installed But Module Not Found

**Problem**: Python is using system Python instead of venv Python.

**Solution**:
```bash
# Deactivate and reactivate venv
deactivate
source env/bin/activate

# Check which Python is being used
which python3  # Should be env/bin/python3

# Reinstall PyOpenAL
pip install --force-reinstall PyOpenAL
```

## Quick Checklist

Before running spatial audio:

- [ ] Virtual environment is activated (`(env)` in prompt)
- [ ] OpenAL-Soft is installed (`brew install openal-soft`)
- [ ] PyOpenAL is installed in venv (`pip list | grep PyOpenAL`)
- [ ] AirPods are connected (optional, for spatial audio)
- [ ] No other app is using audio

## Testing Installation

Run this command to test everything is set up correctly:

```bash
source env/bin/activate && python3 -c "
import sys
print(f'Python: {sys.executable}')
try:
    import openal
    print('✓ PyOpenAL installed')
except ImportError as e:
    print(f'✗ PyOpenAL error: {e}')
    sys.exit(1)
print('✓ All checks passed!')
"
```

## Common Commands

```bash
# Activate virtual environment
source env/bin/activate

# Run spatial audio test
python3 spatial-audio/index.py

# Run full integration
python3 spatial-audio/integration.py

# Check PyOpenAL version
pip show PyOpenAL

# Reinstall PyOpenAL
pip install --force-reinstall PyOpenAL
```
