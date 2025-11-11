#!/usr/bin/env python3
"""
Project Daredevil - Web Launch Interface
Simple web interface to configure and launch the system
"""

from flask import Flask, render_template_string, request, jsonify, redirect, url_for
import subprocess
import os
import sys
import threading

app = Flask(__name__)

# Store active process
active_process = None
process_status = {"running": False, "output": []}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Project Daredevil - Control Panel</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { font-size: 28px; margin-bottom: 5px; }
        .header p { opacity: 0.9; }
        .content { padding: 30px; }
        .form-group { margin-bottom: 20px; }
        label { 
            display: block; 
            font-weight: 600; 
            margin-bottom: 8px;
            color: #333;
        }
        input, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        .button-group {
            display: flex;
            gap: 10px;
            margin-top: 30px;
        }
        .btn {
            flex: 1;
            padding: 14px;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(102,126,234,0.4); }
        .btn-danger {
            background: #e74c3c;
            color: white;
        }
        .btn-danger:hover { background: #c0392b; }
        .status {
            margin-top: 20px;
            padding: 15px;
            border-radius: 6px;
            background: #f8f9fa;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            max-height: 300px;
            overflow-y: auto;
        }
        .status.running { background: #d4edda; color: #155724; }
        .alert {
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 20px;
        }
        .alert-info {
            background: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎧 Project Daredevil</h1>
            <p>Spatial Audio Blind Assistance Control Panel</p>
        </div>
        <div class="content">
            <div class="alert alert-info">
                Configure your system settings below, then click "Start System" to launch.
            </div>
            
            <form id="configForm">
                <div class="form-group">
                    <label for="camera">Camera Index</label>
                    <input type="number" id="camera" name="camera" value="1" min="0" max="5">
                    <small style="color: #666;">Use 0 for laptop camera, 1 for iPhone Continuity Camera</small>
                </div>
                
                <div class="form-group">
                    <label for="classes">Target Classes</label>
                    <input type="text" id="classes" name="classes" value="person bottle" placeholder="person bottle cup">
                    <small style="color: #666;">Space-separated list of objects to detect</small>
                </div>
                
                <div class="form-group">
                    <label for="volume">Master Volume</label>
                    <input type="number" id="volume" name="volume" value="0.3" min="0" max="1" step="0.05">
                    <small style="color: #666;">Volume level (0.0 = silent, 1.0 = max)</small>
                </div>
                
                <div class="form-group">
                    <label for="confidence">Confidence Threshold</label>
                    <input type="number" id="confidence" name="confidence" value="0.3" min="0" max="1" step="0.1">
                    <small style="color: #666;">Detection confidence threshold (lower = more detections)</small>
                </div>
                
                <div class="button-group">
                    <button type="button" class="btn btn-primary" onclick="startSystem()">Start System</button>
                    <button type="button" class="btn btn-danger" onclick="stopSystem()">Stop System</button>
                </div>
            </form>
            
            <div id="status" class="status" style="display: none;">
                Waiting for system to start...
            </div>
        </div>
    </div>
    
    <script>
        let statusInterval = null;
        
        function startSystem() {
            const form = document.getElementById('configForm');
            const formData = new FormData(form);
            
            fetch('/start', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('status').style.display = 'block';
                document.getElementById('status').className = 'status running';
                document.getElementById('status').textContent = 'System starting...';
                
                if (data.success) {
                    // Poll for status updates
                    statusInterval = setInterval(checkStatus, 2000);
                } else {
                    alert('Error: ' + data.message);
                }
            })
            .catch(error => {
                alert('Error starting system: ' + error);
            });
        }
        
        function stopSystem() {
            fetch('/stop', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                document.getElementById('status').style.display = 'none';
                if (statusInterval) clearInterval(statusInterval);
                alert(data.message);
            });
        }
        
        function checkStatus() {
            fetch('/status')
            .then(response => response.json())
            .then(data => {
                const statusEl = document.getElementById('status');
                if (data.running) {
                    statusEl.textContent = 'System running... (check terminal for output)';
                } else {
                    statusEl.className = 'status';
                    statusEl.textContent = 'System stopped.';
                    if (statusInterval) clearInterval(statusInterval);
                }
            });
        }
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/start", methods=["POST"])
def start_system():
    global active_process, process_status

    print("\n" + "=" * 60)
    print("DEBUG: /start endpoint called")
    print("=" * 60)

    if process_status["running"]:
        print("DEBUG: System is already running, rejecting start request")
        return jsonify({"success": False, "message": "System is already running!"})

    camera = request.form.get("camera", "1")
    classes = request.form.get("classes", "person bottle")
    volume = request.form.get("volume", "0.1")
    confidence = request.form.get("confidence", "0.3")

    print(f"DEBUG: Configuration - camera={camera}, classes={classes}, volume={volume}, confidence={confidence}")

    # Build command
    venv_python = os.path.join(os.path.dirname(__file__), "env", "bin", "python3")
    cmd = (
        [venv_python, "main.py", "--camera", camera, "--classes"]
        + classes.split()
        + ["--volume", volume, "--confidence", confidence]
    )

    print(f"DEBUG: Command to execute: {' '.join(cmd)}")
    print(f"DEBUG: venv_python path: {venv_python}")
    print(f"DEBUG: venv_python exists: {os.path.exists(venv_python)}")
    print(f"DEBUG: main.py exists: {os.path.exists('main.py')}")

    try:
        # Start process
        print("DEBUG: Starting subprocess...")
        active_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=os.path.dirname(__file__)
        )
        print(f"DEBUG: Process started with PID: {active_process.pid}")
        print(f"DEBUG: Process poll() result: {active_process.poll()}")
        
        process_status["running"] = True
        process_status["output"] = []

        # Start output monitoring thread
        print("DEBUG: Starting monitor thread...")
        threading.Thread(
            target=monitor_process, args=(active_process,), daemon=True
        ).start()

        print("DEBUG: System started successfully")
        return jsonify({"success": True, "message": "System started successfully"})
    except Exception as e:
        print(f"ERROR: Failed to start system: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)})


@app.route("/stop", methods=["POST"])
def stop_system():
    global active_process, process_status

    print("\n" + "=" * 60)
    print("DEBUG: /stop endpoint called")
    print("=" * 60)

    if not process_status["running"]:
        print("DEBUG: System is not running, nothing to stop")
        return jsonify({"message": "System is not running"})

    try:
        if active_process:
            poll_result = active_process.poll()
            print(f"DEBUG: active_process exists, PID: {active_process.pid}")
            print(f"DEBUG: Process poll() result: {poll_result} (None = running, number = exit code)")
            
            if poll_result is None:
                print("DEBUG: Process is still running, attempting to terminate...")
                import signal
                import time

                # First, try to gracefully terminate
                print("DEBUG: Sending SIGTERM to process...")
                active_process.terminate()

                try:
                    # Wait up to 2 seconds for graceful shutdown
                    print("DEBUG: Waiting for graceful shutdown (max 2 seconds)...")
                    active_process.wait(timeout=2)
                    print("DEBUG: Process terminated gracefully")
                except subprocess.TimeoutExpired:
                    # If it didn't stop, forcefully kill it
                    print("DEBUG: Process didn't terminate, forcing kill...")
                    active_process.kill()
                    active_process.wait(timeout=1)
                    print("DEBUG: Process killed forcefully")

                # Kill any child processes that might be holding the camera
                try:
                    import psutil

                    try:
                        print("DEBUG: Attempting to kill child processes...")
                        parent = psutil.Process(active_process.pid)
                        children = parent.children(recursive=True)
                        print(f"DEBUG: Found {len(children)} child processes")
                        for child in children:
                            print(f"DEBUG: Killing child process PID: {child.pid}")
                            child.kill()
                        print("DEBUG: All child processes killed")
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                        print(f"DEBUG: Could not kill child processes: {e}")
                except ImportError:
                    print("DEBUG: psutil not available, skipping child process killing")
            else:
                print(f"DEBUG: Process already exited with code: {poll_result}")
        else:
            print("DEBUG: active_process is None")

        # Always update status even if process was None or already dead
        process_status["running"] = False
        active_process = None
        print("DEBUG: Status updated to stopped")
        return jsonify({"message": "System stopped successfully"})
    except Exception as e:
        print(f"ERROR: Exception during stop: {e}")
        import traceback
        traceback.print_exc()
        process_status["running"] = False
        active_process = None
        return jsonify({"message": f"System stopped (with errors: {str(e)})"})


@app.route("/status")
def status():
    global active_process, process_status
    
    # Check if process is actually still running
    if active_process:
        poll_result = active_process.poll()
        if poll_result is not None:
            # Process has exited
            if process_status["running"]:
                print(f"DEBUG: Process status check - process exited with code {poll_result}, but status still shows running!")
                print(f"DEBUG: Updating status to stopped")
                process_status["running"] = False
                process_status["output"].append(f"Process exited with code {poll_result}.")
        else:
            # Process is still running
            if not process_status["running"]:
                print(f"DEBUG: Process status check - process is running (poll=None), but status shows stopped!")
    
    return jsonify(process_status)


def monitor_process(process):
    """Monitor process output"""
    global process_status
    print(f"DEBUG: Monitor thread started for PID: {process.pid}")
    try:
        for line in iter(process.stdout.readline, ""):
            if line:
                line = line.strip()
                if line:  # Only add non-empty lines
                    process_status["output"].append(line)
                    print(f"DEBUG: [Process Output] {line}")
                    # Keep only last 100 lines
                    if len(process_status["output"]) > 100:
                        process_status["output"] = process_status["output"][-100:]
        
        # If we exit the loop, process has finished
        print(f"DEBUG: Process stdout closed, waiting for process to exit...")
        return_code = process.wait()
        print(f"DEBUG: Process exited with return code: {return_code}")
        print(f"DEBUG: Process poll() now returns: {process.poll()}")
        
    except Exception as e:
        print(f"DEBUG: Exception in monitor_process: {e}")
        import traceback
        traceback.print_exc()
        try:
            return_code = process.poll()
            print(f"DEBUG: Process poll() after exception: {return_code}")
        except:
            pass
    finally:
        # Always mark as not running when monitor exits
        print(f"DEBUG: Monitor thread exiting, marking system as stopped")
        print(f"DEBUG: Process status before update: running={process_status['running']}")
        process_status["running"] = False
        process_status["output"].append("Process terminated.")
        print(f"DEBUG: Process status after update: running={process_status['running']}")
        print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🎧 Project Daredevil - Web Control Panel")
    print("=" * 60)
    print("\nAccess the web interface at: http://localhost:8080")
    print("Press Ctrl+C to stop the web server\n")

    app.run(debug=True, host="0.0.0.0", port=8080, use_reloader=False)
