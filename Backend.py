# =============================================================================
#  backend.py — Smart Exam Proctoring System
#  Author: Sidak Raj Virdi | Roll: 1024240043
#
#  Runs ALL heavy AI processing in a background process.
#  Streamlit (app.py) reads results from shared state — never touches the camera.
#
#  WHY THIS FIXES THE FPS PROBLEM:
#    Streamlit's rerun model re-executes the entire script on every interaction.
#    Running inference INSIDE Streamlit means every frame triggers a full rerun.
#    Separating them means inference runs at full speed (20-30 FPS) while
#    Streamlit only polls a lightweight JSON file every 100ms for display.
#
#  Run: python backend.py
#  Then separately: streamlit run app.py
# =============================================================================

import cv2
import time
import json
import os
import threading
import queue
import base64
import numpy as np
from datetime import datetime

from modules.yolo_detector import YOLODetector
from modules.head_pose     import HeadPoseDetector
from modules.eye_gaze      import EyeGazeDetector
from logger                import log_event

# Try winsound (Windows only)
try:
    import winsound
    WINSOUND_OK = True
except ImportError:
    WINSOUND_OK = False

# ── Paths shared with app.py ──────────────────────────────────────────────────
STATE_FILE  = "shared_state.json"   # latest detection results
FRAME_FILE  = "shared_frame.jpg"    # latest annotated frame (JPEG on disk)
LOG_FILE    = "logs.txt"
EVIDENCE_DIR = "evidence"

os.makedirs(EVIDENCE_DIR, exist_ok=True)


# =============================================================================
#  Threaded Camera — removes cap.read() blocking from main loop
# =============================================================================
class CameraStream:
    def __init__(self, src=0, width=320, height=240):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame    = None
        self.ret      = False
        self.lock     = threading.Lock()
        self.running  = True
        ret, frame    = self.cap.read()
        self.ret, self.frame = ret, frame
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def release(self):
        self.running = False
        time.sleep(0.1)
        self.cap.release()


# =============================================================================
#  Threaded YOLO Worker — GPU inference parallel to CPU MediaPipe
# =============================================================================
class YOLOWorker:
    def __init__(self, model):
        self.model        = model
        self._q           = queue.Queue(maxsize=1)
        self._result      = None
        self._lock        = threading.Lock()
        self._running     = True
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while self._running:
            try:
                frame = self._q.get(timeout=0.5)
                res   = self.model.detect(frame)
                with self._lock:
                    self._result = res
            except queue.Empty:
                continue

    def submit(self, frame):
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            try: self._q.get_nowait()
            except queue.Empty: pass
            try: self._q.put_nowait(frame)
            except queue.Full: pass

    def get(self):
        with self._lock:
            return self._result

    def stop(self):
        self._running = False


# =============================================================================
#  Background Evidence Saver — disk IO never stalls main loop
# =============================================================================
class EvidenceSaver:
    def __init__(self):
        self._q = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            item = self._q.get()
            if item is None: break
            path, frame = item
            cv2.imwrite(path, frame)

    def save(self, frame, reason="alert"):
        ts   = int(time.time())
        path = os.path.join(EVIDENCE_DIR, f"evidence_{reason}_{ts}.jpg")
        self._q.put((path, frame.copy()))
        return path

    def stop(self):
        self._q.put(None)


# =============================================================================
#  FPS Counter
# =============================================================================
class FPSCounter:
    def __init__(self, n=30):
        self._t = []
        self._n = n

    def tick(self):
        now = time.time()
        self._t.append(now)
        if len(self._t) > self._n:
            self._t.pop(0)

    def fps(self):
        if len(self._t) < 2: return 0.0
        return round((len(self._t)-1) / (self._t[-1]-self._t[0]), 1)


# =============================================================================
#  Shared State Writer
#  Writes a tiny JSON + JPEG every frame so Streamlit can read it.
#  JSON write is ~0.1ms, JPEG encode is ~2ms — negligible.
# =============================================================================
_write_lock = threading.Lock()

def write_shared_state(state: dict, frame: np.ndarray):
    """Write detection results + frame to disk for Streamlit to read."""
    # Encode frame to JPEG bytes (fast, ~2ms)
    _, buf  = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    with _write_lock:
        with open(FRAME_FILE, 'wb') as f:
            f.write(buf.tobytes())
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)


def play_sound():
    def _p():
        if WINSOUND_OK:
            try: winsound.PlaySound("alert-beep.wav",
                                    winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception: pass
    threading.Thread(target=_p, daemon=True).start()


# =============================================================================
#  MAIN PROCESSING LOOP
# =============================================================================
def main():
    print("🚀 Starting AI Proctoring Backend...")

    # Load models
    yolo_model = YOLODetector()
    head_pose  = HeadPoseDetector()
    eye_gaze   = EyeGazeDetector()
    print("✅ Models loaded")

    cam    = CameraStream(src=0, width=320, height=240)
    yolo_w = YOLOWorker(yolo_model)
    saver  = EvidenceSaver()
    fps    = FPSCounter(n=30)

    # Tunable constants
    YOLO_EVERY  = 3   # run YOLO every Nth frame
    POSE_EVERY  = 2   # run head+gaze every Nth frame

    frame_count      = 0
    last_save_time   = 0.0
    last_sound_time  = 0.0
    session_start    = time.time()

    # Alert counters
    counts = {"LOW":0, "MEDIUM":0, "HIGH":0, "CRITICAL":0}

    # Cached results
    cached_yolo   = None
    cached_dir    = "CENTER"
    cached_gaze   = "CENTER"

    print("▶  Backend running — open Streamlit dashboard in your browser")
    print("   Press Ctrl+C to stop\n")

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            fps.tick()

            # ── YOLO (threaded, every Nth frame) ──────────────────────────────
            if frame_count % YOLO_EVERY == 0:
                yolo_w.submit(frame)
            latest = yolo_w.get()
            if latest is not None:
                cached_yolo = latest

            # ── MediaPipe (every Nth frame — smooth signals) ──────────────────
            if frame_count % POSE_EVERY == 0:
                cached_dir  = head_pose.get_head_pose(frame)
                cached_gaze = eye_gaze.get_gaze(frame)

            direction = cached_dir
            gaze      = cached_gaze

            # ── Parse YOLO ────────────────────────────────────────────────────
            alert_phone  = False
            person_count = 0
            detected_obj = []

            if cached_yolo and len(cached_yolo) > 0:
                result = cached_yolo[0]
                if result.boxes is not None:
                    for box in result.boxes:
                        cls   = int(box.cls[0])
                        label = yolo_model.model.names[cls]
                        conf  = float(box.conf[0])
                        if label == "person":
                            person_count += 1
                        if label == "cell phone":
                            alert_phone = True
                            detected_obj.append(f"phone {conf:.0%}")
                        if label == "book":
                            detected_obj.append(f"book {conf:.0%}")
                annotated = result.plot()
            else:
                annotated = frame.copy()

            # ── Determine alert level ─────────────────────────────────────────
            level = "OK"
            msg   = "System nominal"

            if gaze not in ("CENTER","NO FACE","NO_FACE"):
                level = "LOW"
                msg   = f"Gaze away [{gaze}]"
                counts["LOW"] += 1
                log_event("Eyes not on screen")

            if direction not in ("CENTER","NO FACE","NO_FACE"):
                level = "MEDIUM"
                msg   = f"Head rotated [{direction}]"
                counts["MEDIUM"] += 1
                log_event("Looking away")

            if alert_phone or detected_obj:
                level = "HIGH"
                msg   = f"Object detected: {', '.join(detected_obj) or 'phone'}"
                counts["HIGH"] += 1
                log_event("Mobile detected")

            if person_count > 1:
                level = "CRITICAL"
                msg   = f"Multiple persons: {person_count}"
                counts["CRITICAL"] += 1
                log_event("Multiple people detected")

            # ── Overlays on frame ─────────────────────────────────────────────
            colors = {"OK":(0,210,0),"LOW":(0,220,255),"MEDIUM":(0,140,255),
                      "HIGH":(0,0,255),"CRITICAL":(0,0,180)}
            col = colors.get(level,(255,255,255))

            if level != "OK":
                cv2.rectangle(annotated,(0,0),(annotated.shape[1],38),(0,0,0),-1)
                cv2.putText(annotated, f"[{level}] {msg}",
                            (8,26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 2)

            head_col = (0,255,0) if direction in ("CENTER","NO FACE") else (0,165,255)
            gaze_col = (0,255,255) if gaze in ("CENTER","NO FACE") else (0,100,255)
            cv2.putText(annotated, f"Head:{direction}",
                        (8,120), cv2.FONT_HERSHEY_SIMPLEX, 0.50, head_col, 1)
            cv2.putText(annotated, f"Gaze:{gaze}",
                        (8,140), cv2.FONT_HERSHEY_SIMPLEX, 0.50, gaze_col, 1)

            curr_fps = fps.fps()
            cv2.putText(annotated, f"FPS:{curr_fps}",
                        (annotated.shape[1]-90, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,210,0), 2)

            # ── Evidence + sound ──────────────────────────────────────────────
            now = time.time()
            if level in ("HIGH","CRITICAL"):
                if now - last_save_time > 3:
                    saver.save(frame, reason=level.lower())
                    last_save_time = now
                if now - last_sound_time > 3:
                    play_sound()
                    last_sound_time = now

            # ── Write shared state for Streamlit ──────────────────────────────
            elapsed = int(now - session_start)
            state = {
                "fps":          curr_fps,
                "frame_count":  frame_count,
                "elapsed_sec":  elapsed,
                "alert_level":  level,
                "alert_msg":    msg,
                "direction":    direction,
                "gaze":         gaze,
                "person_count": person_count,
                "phone":        alert_phone,
                "objects":      detected_obj,
                "counts":       counts,
                "screenshots":  len(os.listdir(EVIDENCE_DIR)),
                "peak_fps":     curr_fps,
                "timestamp":    datetime.now().strftime("%H:%M:%S"),
            }
            write_shared_state(state, annotated)

            # Also show local OpenCV window (optional — comment out if not needed)
            cv2.imshow("Proctoring Backend", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[STOP] Keyboard interrupt received")

    finally:
        cam.release()
        yolo_w.stop()
        saver.stop()
        cv2.destroyAllWindows()
        print("[STOP] Backend shut down cleanly")


if __name__ == "__main__":
    main()