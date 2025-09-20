"""YOLO based controller implementation."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - optional dependency
    YOLO = None  # type: ignore[assignment]
    _yolo_import_error = exc
else:  # pragma: no cover - if available we expose no error
    _yolo_import_error = None

try:
    import cv2
except Exception as exc:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore[assignment]
    _cv_import_error = exc
else:
    _cv_import_error = None

from .controller import ControlState, Controller


@dataclass
class YOLOConfig:
    model_path: str = "yolov8n.pt"
    target_class: Optional[int] = None
    camera_index: int = 0
    confidence: float = 0.35
    smoothing: float = 0.2
    launch_timeout: float = 0.4
    max_pull_area: float = 0.18


class YOLOController(Controller):
    """Control implementation based on YOLO object detection."""

    def __init__(self, config: YOLOConfig | None = None, *, visualize: bool = False) -> None:
        if YOLO is None:  # pragma: no cover - depends on optional dependency
            raise RuntimeError(
                "The ultralytics package is required for YOLO control: "
                f"{_yolo_import_error!r}"
            )
        if cv2 is None:  # pragma: no cover - depends on optional dependency
            raise RuntimeError(
                "The OpenCV-python package is required for YOLO control: "
                f"{_cv_import_error!r}"
            )

        self.config = config or YOLOConfig()
        self._visualize = visualize
        self._model = YOLO(self.config.model_path)
        self._cap = cv2.VideoCapture(self.config.camera_index)
        if not self._cap.isOpened():  # pragma: no cover - hardware dependent
            raise RuntimeError("Unable to open camera index %s" % self.config.camera_index)

        self._state_lock = threading.Lock()
        self._latest_state: Optional[ControlState] = None
        self._pending_launch: bool = False
        self._last_pullback: float = 0.0
        self._last_aim: float = 0.0
        self._last_update: float = time.monotonic()
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:  # pragma: no cover - requires camera hardware
        assert cv2 is not None
        assert YOLO is not None
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._model.predict(frame_rgb, conf=self.config.confidence, verbose=False)
            aim_offset = 0.0
            pullback = 0.0
            detected = False
            if results:
                res = results[0]
                if res.boxes is not None and len(res.boxes) > 0:
                    boxes = res.boxes
                    best_idx = int(np.argmax(boxes.conf.cpu().numpy()))
                    box = boxes[best_idx]
                    if self.config.target_class is None or int(box.cls) == self.config.target_class:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        aim_offset, pullback = self._map_detection(frame_rgb.shape, x1, y1, x2, y2)
                        detected = True
                        self._last_update = time.monotonic()
                        with self._state_lock:
                            self._last_pullback = pullback
                            self._last_aim = aim_offset
                            self._latest_state = ControlState(
                                aim_offset=aim_offset, pullback=pullback
                            )
            if not detected:
                since_update = time.monotonic() - self._last_update
                if since_update > self.config.launch_timeout and self._last_pullback > 0.15:
                    with self._state_lock:
                        self._pending_launch = True
                        self._last_pullback = 0.0
                        self._latest_state = ControlState(
                            aim_offset=self._last_aim, pullback=0.0, launch=True
                        )

            if self._visualize:
                display_frame = frame.copy()
                text = f"aim={aim_offset:+.2f} pull={pullback:.2f}"
                cv2.putText(
                    display_frame,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("YOLO Control", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._running = False
                    break

        self._cap.release()
        if self._visualize:
            cv2.destroyAllWindows()

    def _map_detection(
        self, frame_shape: tuple[int, int, int], x1: float, y1: float, x2: float, y2: float
    ) -> tuple[float, float]:
        height, width = frame_shape[:2]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        area = max(0.0, (x2 - x1) * (y2 - y1)) / float(width * height)
        aim_offset = (height / 2 - cy) / (height / 2)
        aim_offset = max(-1.0, min(1.0, aim_offset))
        pullback = min(1.0, area / max(1e-6, self.config.max_pull_area))
        pullback = (1 - self.config.smoothing) * self._last_pullback + self.config.smoothing * pullback
        return aim_offset, pullback

    def poll(self) -> Optional[ControlState]:
        with self._state_lock:
            if self._pending_launch:
                self._pending_launch = False
                return ControlState(
                    aim_offset=self._last_aim,
                    pullback=self._last_pullback,
                    launch=True,
                ).clamp()
            if self._latest_state is not None:
                return ControlState(
                    aim_offset=self._latest_state.aim_offset,
                    pullback=self._latest_state.pullback,
                    launch=self._latest_state.launch,
                ).clamp()
        return None

    def close(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.5)
        if cv2 is not None:
            cv2.destroyAllWindows()
