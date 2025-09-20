"""Interactive YOLO-driven game loop.

This module connects an Ultralytics YOLOv11 model to a lightweight gameplay
context. Frames from a webcam or video file are analysed, detections are drawn
with bounding boxes, and manual keyboard commands (via pygame) are merged into
simple game state updates. The module keeps the overlays consistent with the
original detection snippet (class names and confidence labels) while exposing
configurable runtime options through a CLI entry point.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pygame
from ultralytics import YOLO

import cvzone


DEFAULT_MODEL_PATH = "yolo11n.pt"
WINDOW_NAME = "YOLOv11 Game"


@dataclass
class PlayerAvatar:
    """Simple player representation drawn on top of the frame."""

    position: np.ndarray
    speed: float = 320.0
    radius: int = 18
    color: Tuple[int, int, int] = (0, 255, 0)

    def move(self, delta: Tuple[float, float], bounds: Tuple[int, int]) -> None:
        """Move the avatar within frame bounds."""

        width, height = bounds
        self.position[0] = np.clip(self.position[0] + delta[0], self.radius, width - self.radius)
        self.position[1] = np.clip(self.position[1] + delta[1], self.radius, height - self.radius)


@dataclass
class GameState:
    """Holds the mutable state for the mini game."""

    frame_size: Tuple[int, int]
    tracked_classes: Optional[Sequence[str]] = None
    score: int = 0
    status_message: str = "Awaiting detections"
    player: PlayerAvatar = dataclasses.field(init=False)
    last_detection_time: float = dataclasses.field(default_factory=time.perf_counter)
    message_ttl: float = 3.0

    def __post_init__(self) -> None:
        width, height = self.frame_size
        start_position = np.array([width / 2.0, height * 0.75], dtype=float)
        self.player = PlayerAvatar(position=start_position)
        self.tracked_classes = tuple(self.tracked_classes or [])

    def update_manual(self, pressed: Sequence[bool], delta_time: float) -> None:
        """Update player state using keyboard input."""

        dx = dy = 0.0
        speed = self.player.speed * delta_time
        if pressed[pygame.K_LEFT] or pressed[pygame.K_a]:
            dx -= speed
        if pressed[pygame.K_RIGHT] or pressed[pygame.K_d]:
            dx += speed
        if pressed[pygame.K_UP] or pressed[pygame.K_w]:
            dy -= speed
        if pressed[pygame.K_DOWN] or pressed[pygame.K_s]:
            dy += speed
        if dx or dy:
            self.player.move((dx, dy), self.frame_size)

    def handle_events(self, events: Iterable[pygame.event.Event]) -> bool:
        """Process discrete pygame events. Returns False when quitting."""

        for event in events:
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_SPACE:
                    # Manual trigger for score adjustments.
                    self.score += 1
                    self.status_message = "Manual boost"
                    self.last_detection_time = time.perf_counter()
        return True

    def register_detection(self, cls_name: str, confidence: float) -> None:
        """Update state based on model detections."""

        if not self.tracked_classes or cls_name in self.tracked_classes:
            self.score += 1
            self.status_message = f"Detected: {cls_name} ({confidence:.2f})"
            self.last_detection_time = time.perf_counter()

    def draw_overlay(self, frame: np.ndarray) -> None:
        """Render player avatar and HUD text on the frame."""

        cv2.circle(frame, tuple(self.player.position.astype(int)), self.player.radius, self.player.color, thickness=-1)
        cv2.putText(
            frame,
            f"Score: {self.score}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        now = time.perf_counter()
        if now - self.last_detection_time <= self.message_ttl:
            cv2.putText(
                frame,
                self.status_message,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )


def resolve_source(source: str) -> int | str:
    """Interpret the capture source as webcam index or file path."""

    if len(source) == 1 and source.isdigit():
        return int(source)
    try:
        return int(source)
    except ValueError:
        return source


def create_capture(source: int | str, width: Optional[int], height: Optional[int]) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(source)
    if width is not None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def draw_detections(
    frame: np.ndarray,
    boxes: Sequence[Tuple[int, int, int, int]],
    labels: Sequence[str],
    confidences: Sequence[float],
) -> None:
    """Draw bounding boxes and class labels onto the frame."""

    for (x1, y1, x2, y2), label, conf in zip(boxes, labels, confidences):
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame, (x1, y1, w, h), l=18, rt=2, colorC=(0, 255, 0), colorR=(0, 155, 255))
        cvzone.putTextRect(
            frame,
            f"{label} {conf:.2f}",
            (x1, max(35, y1 - 10)),
            scale=0.9,
            thickness=2,
            colorT=(255, 255, 255),
            colorR=(0, 0, 0),
        )


def run(
    model_path: str,
    source: str,
    confidence: float,
    tracked_classes: Optional[Sequence[str]] = None,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
    mirror: bool = False,
    show_window: bool = True,
    warmup_frames: int = 0,
) -> None:
    """Execute the YOLO powered game loop."""

    resolved_source = resolve_source(source)
    capture = create_capture(resolved_source, frame_width, frame_height)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")

    pygame.init()
    pygame.display.set_mode((1, 1))  # Minimal hidden window for event processing.
    clock = pygame.time.Clock()

    try:
        success, frame = capture.read()
        if not success:
            raise RuntimeError("Unable to read from source during initialisation")
        if mirror:
            frame = cv2.flip(frame, 1)

        game_state = GameState(frame_size=(frame.shape[1], frame.shape[0]), tracked_classes=tracked_classes)

        model = YOLO(model_path)
        name_lookup = getattr(model, "names", None)
        if name_lookup is None and hasattr(model, "model"):
            name_lookup = getattr(model.model, "names", None)

        # Warmup if desired to load weights into memory.
        for _ in range(max(0, warmup_frames)):
            _ = model.predict(frame, conf=confidence, verbose=False)

        running = True
        while running:
            success, frame = capture.read()
            if not success:
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            delta_time = clock.tick(60) / 1000.0
            events = pygame.event.get()
            running = game_state.handle_events(events)
            if not running:
                break

            pressed = pygame.key.get_pressed()
            game_state.update_manual(pressed, delta_time)

            results = model.predict(frame, conf=confidence, verbose=False)
            boxes: List[Tuple[int, int, int, int]] = []
            labels: List[str] = []
            confidences: List[float] = []

            for result in results:
                if not hasattr(result, "boxes") or result.boxes is None:
                    continue
                for box in result.boxes:
                    if box.conf is None or box.cls is None:
                        continue
                    conf_value = float(box.conf[0])
                    cls_index = int(box.cls[0])
                    if conf_value < confidence:
                        continue
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy.tolist()
                    boxes.append((x1, y1, x2, y2))
                    if isinstance(name_lookup, dict):
                        label = name_lookup.get(cls_index, str(cls_index))
                    elif isinstance(name_lookup, (list, tuple)):
                        label = name_lookup[cls_index] if 0 <= cls_index < len(name_lookup) else str(cls_index)
                    else:
                        label = str(cls_index)
                    labels.append(label)
                    confidences.append(conf_value)
                    game_state.register_detection(label, conf_value)

            if boxes:
                draw_detections(frame, boxes, labels, confidences)
            game_state.draw_overlay(frame)

            if show_window:
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        pygame.quit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the YOLOv11 powered game loop.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to YOLOv11 weights")
    parser.add_argument("--source", default="0", help="Video source: webcam index or media path")
    parser.add_argument("--confidence", type=float, default=0.35, help="Confidence threshold for detections")
    parser.add_argument(
        "--tracked-classes",
        nargs="*",
        default=None,
        help="Optional list of class names to count toward the score",
    )
    parser.add_argument("--width", type=int, default=None, help="Desired capture width")
    parser.add_argument("--height", type=int, default=None, help="Desired capture height")
    parser.add_argument("--mirror", action="store_true", help="Mirror the incoming frames for a selfie view")
    parser.add_argument("--no-window", dest="show_window", action="store_false", help="Disable OpenCV display window")
    parser.add_argument("--warmup", type=int, default=0, help="Run a number of warmup forward passes")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(
        model_path=args.model_path,
        source=args.source,
        confidence=args.confidence,
        tracked_classes=args.tracked_classes,
        frame_width=args.width,
        frame_height=args.height,
        mirror=args.mirror,
        show_window=args.show_window,
        warmup_frames=args.warmup,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
