"""Start RamanCopilot on virtual hardware: python run_virtual.py [seed]"""
import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
URL = "http://localhost:3000"

def in_use(port: int) -> bool:
    """Check localhost on both IPv4 and IPv6 (vite may bind ::1 only)."""
    try:
        socket.create_connection(("localhost", port), timeout=1).close()
        return True
    except OSError:
        return False


busy = [p for p in (8000, 3000) if in_use(p)]
if busy:
    sys.exit(f"port {', '.join(map(str, busy))} already in use - stop the running RamanCopilot server first.")

env = {**os.environ, "RAMAN_VIRTUAL_HW": "1"}
if len(sys.argv) > 1:
    env["RAMAN_VBENCH_SEED"] = sys.argv[1]

back = subprocess.Popen([sys.executable, "-m", "backend.controller.web.main"], cwd=ROOT, env=env)
while not in_use(8000):  # start vite only after the backend listens (no proxy errors)
    if back.poll() is not None:
        sys.exit("backend exited during startup - see the log above.")
    time.sleep(1)
front =subprocess.Popen("npm run dev -- --strictPort", cwd=os.path.join(ROOT, "frontend"), shell=True)

for _ in range(60):  # wait for vite before opening the browser
    try:
        urllib.request.urlopen(URL, timeout=1)
        break
    except OSError:
        time.sleep(1)
webbrowser.open(URL)

try:
    back.wait()
except KeyboardInterrupt:  # Ctrl+C reaches both children through the console
    back.wait()
finally:
    subprocess.run(["taskkill", "/PID", str(front.pid), "/T", "/F"], capture_output=True)
