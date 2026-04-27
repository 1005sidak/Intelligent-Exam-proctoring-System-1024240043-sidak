# =============================================================================
#  main.py — Smart Exam Proctoring System (FPS Optimized)
#  Author: Sidak Raj Virdi | Roll: 1024240043 | Batch: 2X12
#  Thapar Institute of Engineering & Technology, Patiala
#
#  OPTIMIZATION TECHNIQUES APPLIED:
#   1. YOLO runs every Nth frame (skip-frame scheduling)
#   2. Reduced input resolution (320×240)
#   3. Threaded camera capture (decouples read from processing)
#   4. Threaded YOLO inference (runs in parallel with MediaPipe)
#   5. Result caching (reuse last valid detection between frames)
#   6. Frame skip under CPU load (adaptive scheduling)
#   7. winsound async flag (non-blocking audio)
#   8. Evidence saved in background thread (no IO stall)
#   9. Single imshow call per loop (no duplicate draws)
#  10. FPS counter displayed live on frame
# =============================================================================

import cv2
import time
import threading
import queue
import os
from modules.yolo_detector import YOLODetector
from modules.head_pose     import HeadPoseDetector
from modules.eye_gaze      import EyeGazeDetector
from logger                import log_event

# ── Try importing winsound (Windows only) ────────────────────────────────────
try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False


# =============================================================================
#  OPTIMIZATION 1 & 3: Threaded Camera Capture
#  Problem: cap.read() blocks the main thread while waiting for the next frame.
#  Fix: Run cap.read() in a background thread. Main loop always gets the
#       latest frame instantly without blocking.
# =============================================================================

class CameraStream:
    """Non-blocking camera reader — decouples frame capture from processing."""

    def __init__(self, src=0, width=320, height=240):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimise buffer lag

        self.frame  = None
        self.ret    = False
        self.lock   = threading.Lock()
        self.running = True

        # Read the very first frame synchronously so main loop doesn't start
        # with frame=None
        self.ret, self.frame = self.cap.read()

        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else (False, None)

    def release(self):
        self.running = False
        self._thread.join(timeout=1)
        self.cap.release()


# =============================================================================
#  OPTIMIZATION 4: Threaded YOLO Inference
#  Problem: YOLOv8 inference takes ~30–80ms per frame — it blocks everything.
#  Fix: Run YOLO in its own thread. Main loop submits frames to a queue and
#       reads back the latest result. MediaPipe runs in parallel on the CPU
#       while YOLO runs on the GPU simultaneously.
# =============================================================================

class YOLOWorker:
    """Runs YOLO inference in a background thread."""

    def __init__(self, model: YOLODetector):
        self.model       = model
        self._in_queue   = queue.Queue(maxsize=1)   # only keep latest frame
        self._result     = None
        self._result_lock = threading.Lock()
        self._running    = True

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while self._running:
            try:
                frame = self._in_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            result = self.model.detect(frame)
            with self._result_lock:
                self._result = result

    def submit(self, frame):
        """Submit a new frame for inference. Drops old pending frame if busy."""
        try:
            self._in_queue.put_nowait(frame)
        except queue.Full:
            try:
                self._in_queue.get_nowait()   # discard stale frame
            except queue.Empty:
                pass
            self._in_queue.put_nowait(frame)

    def get_result(self):
        """Return latest YOLO result (may be from a previous frame — that's OK)."""
        with self._result_lock:
            return self._result

    def stop(self):
        self._running = False
        self._thread.join(timeout=1)


# =============================================================================
#  OPTIMIZATION 8: Background Evidence Saver
#  Problem: cv2.imwrite() can take 10–30ms for JPEG compression — stalls loop.
#  Fix: Save in a background thread so the main loop never waits for disk IO.
# =============================================================================

class EvidenceSaver:
    """Saves screenshots to disk in a background thread."""

    def __init__(self, output_dir="evidence"):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self._queue  = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while True:
            path, frame = self._queue.get()
            if path is None:
                break
            cv2.imwrite(path, frame)

    def save(self, frame, reason="alert"):
        fname = os.path.join(self.output_dir,
                             f"evidence_{reason}_{int(time.time())}.jpg")
        self._queue.put((fname, frame.copy()))
        print(f"📸 Evidence queued: {fname}")
        return fname

    def stop(self):
        self._queue.put((None, None))


# =============================================================================
#  OPTIMIZATION 7: Non-blocking Audio
# =============================================================================

def play_alert_sound():
    """Play beep in a daemon thread — never blocks main loop."""
    def _play():
        if WINSOUND_AVAILABLE:
            try:
                winsound.PlaySound("alert-beep.wav",
                                   winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception:
                pass
    threading.Thread(target=_play, daemon=True).start()


# =============================================================================
#  FPS Counter
# =============================================================================

class FPSCounter:
    def __init__(self, window=30):
        self._times = queue.deque(maxlen=window)

    def tick(self):
        self._times.append(time.time())

    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("🚀 Starting Smart Proctoring System (Optimized)...")

    # ── Load models ───────────────────────────────────────────────────────────
    yolo_model = YOLODetector()
    head_pose  = HeadPoseDetector()
    eye_gaze   = EyeGazeDetector()
    print("✅ All models loaded")

    # ── Start workers ─────────────────────────────────────────────────────────
    cam     = CameraStream(src=0, width=320, height=240)
    yolo_w  = YOLOWorker(yolo_model)
    saver   = EvidenceSaver(output_dir="evidence")
    fps_ctr = FPSCounter(window=30)

    # ── Timing / state ────────────────────────────────────────────────────────
    last_save_time  = 0.0
    last_sound_time = 0.0

    # ==========================================================================
    #  OPTIMIZATION 1: Skip-frame YOLO scheduling
    #  YOLO runs every YOLO_EVERY frames. Between those frames the cached
    #  result is reused — zero GPU cost on skipped frames.
    # ==========================================================================
    YOLO_EVERY   = 3    # run YOLO every 3rd frame  (tune: 2=more accurate, 5=faster)
    frame_count  = 0

    # Cached module results (reused on skipped frames)
    cached_yolo_result  = None
    cached_direction    = "CENTER"
    cached_gaze         = "CENTER"

    print("▶  Press Q to quit\n")

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            print("❌ Frame read failed")
            break

        frame_count += 1
        fps_ctr.tick()

        # ── OPTIMIZATION 1+4: Submit to YOLO worker every Nth frame ──────────
        if frame_count % YOLO_EVERY == 0:
            yolo_w.submit(frame)

        # ── Get latest YOLO result (non-blocking) ─────────────────────────────
        latest = yolo_w.get_result()
        if latest is not None:
            cached_yolo_result = latest

        # ── OPTIMIZATION 6: Skip MediaPipe on alternate frames ────────────────
        #  Head pose + gaze are smooth signals — updating every 2nd frame
        #  is imperceptible but halves MediaPipe CPU cost.
        if frame_count % 2 == 0:
            cached_direction = head_pose.get_head_pose(frame)
            cached_gaze      = eye_gaze.get_gaze(frame)

        direction = cached_direction
        gaze      = cached_gaze

        # ── Parse YOLO results ────────────────────────────────────────────────
        alert_phone  = False
        person_count = 0

        results = cached_yolo_result
        if results is not None and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    cls   = int(box.cls[0])
                    label = yolo_model.model.names[cls]
                    if label == "person":
                        person_count += 1
                    if label == "cell phone":
                        alert_phone = True
            annotated = result.plot()
        else:
            annotated = frame.copy()

        # ── Log events (only once per trigger, not every frame) ───────────────
        if alert_phone:
            log_event("Mobile detected")
        if direction not in ("CENTER", "NO FACE"):
            log_event("Looking away")
        if gaze not in ("CENTER", "NO FACE"):
            log_event("Eyes not on screen")
        if person_count > 1:
            log_event("Multiple people detected")

        # ── Draw overlays ─────────────────────────────────────────────────────
        alert_triggered = False

        if alert_phone:
            cv2.putText(annotated, "⚠ PHONE DETECTED",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            alert_triggered = True

        if person_count > 1:
            cv2.putText(annotated, "⚠ MULTIPLE PEOPLE",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            alert_triggered = True

        head_col = (0, 255, 0) if direction in ("CENTER","NO FACE") else (0, 165, 255)
        cv2.putText(annotated, f"Head: {direction}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, head_col, 2)

        if direction not in ("CENTER", "NO FACE"):
            cv2.putText(annotated, "⚠ LOOKING AWAY",
                        (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            alert_triggered = True

        gaze_col = (0, 255, 255) if gaze in ("CENTER","NO FACE") else (0, 100, 255)
        cv2.putText(annotated, f"Gaze: {gaze}",
                    (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.55, gaze_col, 2)

        if gaze not in ("CENTER", "NO FACE"):
            cv2.putText(annotated, "⚠ EYES OFF SCREEN",
                        (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            alert_triggered = True

        # ── FPS overlay (top-right) ───────────────────────────────────────────
        fps_val = fps_ctr.fps()
        h, w = annotated.shape[:2]
        cv2.putText(annotated, f"FPS: {fps_val:.1f}",
                    (w - 110, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 0), 2)

        # ── Evidence + sound ──────────────────────────────────────────────────
        if alert_triggered:
            now = time.time()

            # OPTIMIZATION 8: background save — zero main-loop stall
            if now - last_save_time > 3:
                saver.save(frame)
                last_save_time = now

            # OPTIMIZATION 7: async sound — zero main-loop stall
            if now - last_sound_time > 3:
                play_alert_sound()
                last_sound_time = now

        # ── Display ───────────────────────────────────────────────────────────
        cv2.imshow("Smart Proctoring System", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print("\n[EXIT] Shutting down...")
    cam.release()
    yolo_w.stop()
    saver.stop()
    cv2.destroyAllWindows()
    print("[EXIT] Done.")


if __name__ == "__main__":
    main()