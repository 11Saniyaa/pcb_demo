"""Start the demo frontend and backend, then print the URLs."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
FRONTEND_HOST = "127.0.0.1"
FRONTEND_PORT = 5173

FRONTEND_URL = f"http://{FRONTEND_HOST}:{FRONTEND_PORT}/"
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}/"
API_DOCS_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}/docs"
HEALTH_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}/api/health"


def print_links() -> None:
    print()
    print("=" * 56)
    print("  PCBA placement fault simulator")
    print("=" * 56)
    print()
    print("  Open the app:     ", FRONTEND_URL)
    print("  Backend API:      ", BACKEND_URL)
    print("  API docs:         ", API_DOCS_URL)
    print()
    print("  Press Ctrl+C to stop both servers.")
    print("=" * 56)
    print()


def wait_for(url: str, timeout: float = 40.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 500:
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            time.sleep(0.4)
    return False


def url_is_up(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1.5) as response:
            return 200 <= response.status < 500
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return False


def pids_listening_on(port: int) -> set[int]:
    pids: set[int] = set()
    result = subprocess.run(
        ["netstat", "-ano", "-p", "tcp"],
        capture_output=True,
        text=True,
        check=False,
    )
    suffix = f":{port}"
    for line in result.stdout.splitlines():
        if "LISTENING" not in line.upper():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        local = parts[1]
        if local.endswith(suffix) or local.endswith(f"]{suffix}"):
            try:
                pid = int(parts[-1])
            except ValueError:
                continue
            if pid > 0:
                pids.add(pid)
    return pids


def free_port(port: int) -> None:
    pids = pids_listening_on(port)
    if not pids:
        return
    print(f"Port {port} is busy. Stopping process {', '.join(str(pid) for pid in sorted(pids))}...")
    for pid in pids:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/F", "/T"],
            capture_output=True,
            check=False,
        )
    time.sleep(0.8)


def npm_command() -> list[str]:
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
    if not npm:
        raise RuntimeError("npm was not found. Install Node.js, then run npm install in frontend/.")
    return [npm, "run", "dev"]


def main() -> int:
    if not (FRONTEND_DIR / "node_modules").exists():
        print("Installing frontend packages (first run only)...")
        npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
        if not npm:
            print("npm was not found. Install Node.js first.")
            return 1
        install = subprocess.run([npm, "install"], cwd=FRONTEND_DIR)
        if install.returncode != 0:
            return install.returncode

    print_links()

    if url_is_up(FRONTEND_URL) and url_is_up(HEALTH_URL):
        print("  Both servers are already running.")
        print("  Open this link in your browser:")
        print(" ", FRONTEND_URL)
        print()
        return 0

    if os.name == "nt":
        if not url_is_up(HEALTH_URL):
            free_port(BACKEND_PORT)
        if not url_is_up(FRONTEND_URL):
            free_port(FRONTEND_PORT)

    backend = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            BACKEND_HOST,
            "--port",
            str(BACKEND_PORT),
        ],
        cwd=BACKEND_DIR,
    )
    frontend = subprocess.Popen(npm_command(), cwd=FRONTEND_DIR)

    processes = [backend, frontend]

    def shutdown() -> None:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    try:
        backend_ready = wait_for(HEALTH_URL)
        frontend_ready = wait_for(FRONTEND_URL)

        print()
        if backend_ready:
            print("  Backend is up:   ", BACKEND_URL)
        else:
            print("  Backend did not respond yet. Check the uvicorn output above.")
        if frontend_ready:
            print("  Frontend is up:  ", FRONTEND_URL)
        else:
            print("  Frontend did not respond yet. Check the Vite output above.")
        print()
        print("  Open this link in your browser:")
        print(" ", FRONTEND_URL)
        print()

        while True:
            if backend.poll() is not None:
                print("Backend stopped.")
                break
            if frontend.poll() is not None:
                print("Frontend stopped.")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping servers...")
    finally:
        shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
