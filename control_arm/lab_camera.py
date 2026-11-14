"""Shared observation camera. No tracking, inference, or robot commands."""
import threading
import time


class LabCamera:
    def __init__(self, cv, camera_index, busy=lambda: False):
        self.cv = cv
        self.camera_index = camera_index
        self.busy = busy
        self.lock = threading.Lock()
        self.thread = None
        self.jpeg = None
        self.captured_at = 0
        self.requested_at = 0

    def frame(self):
        with self.lock:
            self.requested_at = time.monotonic()
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._capture, daemon=True)
                self.thread.start()
            if self.busy() or time.monotonic() - self.captured_at > 2:
                return None
            return self.jpeg

    def _capture(self):
        cap = None
        try:
            if self.busy():
                return
            cap = self.cv.VideoCapture(self.camera_index())
            if not cap.isOpened():
                return
            cap.set(self.cv.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(self.cv.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(self.cv.CAP_PROP_FPS, 20)
            while time.monotonic() - self.requested_at < 15 and not self.busy():
                started = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    break
                ok, encoded = self.cv.imencode('.jpg', frame, [self.cv.IMWRITE_JPEG_QUALITY, 65])
                if ok:
                    with self.lock:
                        self.jpeg = encoded.tobytes()
                        self.captured_at = time.monotonic()
                time.sleep(max(0, 0.05 - (time.monotonic() - started)))
        finally:
            if cap is not None:
                cap.release()
            with self.lock:
                self.jpeg = None
                self.captured_at = 0

    def stream(self):
        # Bounded connections recheck platform authentication on reconnect.
        deadline = time.monotonic() + 20
        last = 0
        while time.monotonic() < deadline:
            frame = self.frame()
            captured = self.captured_at
            if frame is not None and captured != last:
                last = captured
                yield (b'--parcframe\r\nContent-Type: image/jpeg\r\nContent-Length: '
                       + str(len(frame)).encode() + b'\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.02)
