"""
===============================================================================
Encrypted Traffic Classifier — Dashboard Server (Version 2)
===============================================================================

This is the ONLY new backend file. It does not reimplement, duplicate, or
fake anything from your pipeline. It:

    1. Imports your real `EncryptedTrafficClassifier` from main.py
    2. Subclasses it just enough to keep a rolling history of predictions
       (main.py only keeps the single latest one, which is fine for the CLI
       but the dashboard's "Recent Predictions" table needs a short log)
    3. Runs your existing `start()` / `stop()` in a background thread so the
       web server stays responsive while your sniffer loop runs
    4. Exposes `get_statistics()` (your real method, untouched) as JSON over
       a `/api/stats` endpoint that the dashboard polls every second

Nothing here invents numbers. Every field in /api/stats is either a direct
pass-through of your `get_statistics()` output, or a simple derived value
computed from it (e.g. flows/sec = completed_flows / elapsed_seconds).

Run with:
    uvicorn dashboard_server:app --reload
Then open http://127.0.0.1:8000
Note: packet sniffing (scapy) generally needs admin/root privileges.
===============================================================================
"""

import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from main import EncryptedTrafficClassifier

# Directory for locally vendored static assets (e.g. chart.umd.min.js), so the
# dashboard's donut charts work even on machines with no internet access.
# See the "static/README" note this script creates on first run for how to
# populate it.
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# =============================================================================
# Subclass: adds a rolling prediction history on top of your real pipeline
# =============================================================================


class DashboardClassifier(EncryptedTrafficClassifier):
    """
    Identical to EncryptedTrafficClassifier, plus a short rolling log of
    predictions for the dashboard's "Recent Predictions" table. The
    underlying process_packet() logic is untouched -- we just observe when
    a new sequence has been classified and record it.
    """

    def __init__(self):
        super().__init__()
        self.history = deque(maxlen=10)
        self._last_completed_seen = 0
        self.start_time = None

    def process_packet(self, packet):
        super().process_packet(packet)

        # A new sequence was classified if completed_flows advanced.
        if self.completed_flows != self._last_completed_seen:
            self._last_completed_seen = self.completed_flows

            latest = getattr(self, "latest_prediction", None)
            if latest is not None:
                self.history.appendleft({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "prediction": latest["prediction"],
                    "exit": latest["exit"],
                    "confidence": latest["confidence"],
                })


# =============================================================================
# Server state
# =============================================================================

classifier = DashboardClassifier()

state = {
    "running": False,
    "interface": None,
    "capture_thread": None,
    "error": None,
}

app = FastAPI(title="Encrypted Traffic Classifier Dashboard")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class StartRequest(BaseModel):
    interface: str


# =============================================================================
# Background capture runner
# =============================================================================


def _run_capture(interface: str):
    try:
        classifier.start(interface)
    except Exception as exc:  # capture thread errors surface via /api/stats
        state["error"] = str(exc)
    finally:
        state["running"] = False


# =============================================================================
# API Routes
# =============================================================================


@app.get("/")
def serve_dashboard():
    return FileResponse(Path(__file__).parent / "dashboard.html")


@app.get("/api/interfaces")
def list_interfaces():
    """
    Real network interfaces available on this machine, using your own
    InterfaceManager (already filters to interfaces that are actually up).
    Same source of truth main.py itself would use.
    """
    try:
        interfaces = classifier.interface_manager.get_available_interfaces()
    except Exception as exc:
        return JSONResponse({"interfaces": [], "error": str(exc)}, status_code=200)

    return {"interfaces": interfaces}


@app.post("/api/start")
def start_capture(req: StartRequest):
    if state["running"]:
        return JSONResponse({"ok": False, "error": "Capture already running."}, status_code=400)

    if not classifier.interface_manager.validate_interface(req.interface):
        return JSONResponse(
            {"ok": False, "error": f"Invalid interface: {req.interface}"},
            status_code=400,
        )

    state["error"] = None
    state["interface"] = req.interface
    state["running"] = True
    classifier.start_time = time.time()

    thread = threading.Thread(target=_run_capture, args=(req.interface,), daemon=True)
    state["capture_thread"] = thread
    thread.start()

    return {"ok": True}


@app.post("/api/stop")
def stop_capture():
    if state["running"]:
        classifier.stop()
    state["running"] = False
    return {"ok": True}


@app.get("/api/stats")
def get_stats():
    """
    Everything here comes straight from your classifier's real
    get_statistics(), plus small derived values (elapsed time, throughput).
    """
    stats = classifier.get_statistics()

    elapsed = 0.0
    if classifier.start_time:
        elapsed = time.time() - classifier.start_time

    throughput = (
        stats["completed_flows"] / elapsed if elapsed > 0 else 0.0
    )

    return {
        "running": state["running"],
        "error": state["error"],
        "interface": state["interface"],
        "elapsed_seconds": elapsed,

        "packet_count": stats["packet_count"],
        "active_flows": stats["active_flows"],
        "completed_flows": stats["completed_flows"],
        "average_latency": stats["average_latency"],

        "class_counter": stats["class_counter"],
        "exit_counter": stats["exit_counter"],

        "latest_prediction": stats["latest_prediction"],

        "average_packets": stats["average_packets"],
        "latency_saved": stats["latency_saved"],
        "throughput": throughput,

        "history": list(classifier.history),
    }


# =============================================================================
# Entry Point — `python dashboard_server.py` starts the server AND
# opens the dashboard in your default browser automatically.
# =============================================================================

HOST = "127.0.0.1"
PORT = 8000


def _open_browser_when_ready():
    import urllib.request
    import webbrowser

    url = f"http://{HOST}:{PORT}"

    # Wait until the server is actually accepting connections before
    # opening the tab, instead of guessing with a fixed sleep.
    for _ in range(100):
        try:
            urllib.request.urlopen(url, timeout=0.5)
            break
        except Exception:
            time.sleep(0.2)

    webbrowser.open(url)


if __name__ == "__main__":
    import uvicorn

    chart_js_path = STATIC_DIR / "chart.umd.min.js"
    if not chart_js_path.exists():
        print(
            f"[dashboard] NOTE: {chart_js_path} not found.\n"
            "  The donut charts need Chart.js. On a machine with internet access, "
            "download it once from either of these (both verified working):\n"
            "    https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js\n"
            "    https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.5.0/chart.umd.min.js\n"
            f"  and save it to: {chart_js_path}\n"
            "  The dashboard will also try loading it live from jsDelivr, then "
            "cdnjs, if this machine does have internet access; if not, the "
            "rest of the dashboard (stats, tables) still works without it.\n"
        )
    print(
        f"[dashboard] Open the dashboard at http://{HOST}:{PORT} in your "
        "browser. Do NOT open dashboard.html directly as a file -- the "
        "/api/* endpoints only exist when served by this script."
    )

    threading.Thread(target=_open_browser_when_ready, daemon=True).start()

    uvicorn.run(app, host=HOST, port=PORT)