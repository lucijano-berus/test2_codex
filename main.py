"""Entry point for the YOLO controlled Angry Birds demo."""
from __future__ import annotations

import argparse
import sys

from angrybirds.game import run_game
from angrybirds.mouse_controller import MouseController

try:
    from angrybirds.yolo_controller import YOLOConfig, YOLOController
except RuntimeError:
    YOLOController = None  # type: ignore[assignment]
    YOLOConfig = None  # type: ignore[assignment]
except Exception as exc:  # pragma: no cover - optional dependency
    print(f"Unexpected import error for YOLO controller: {exc}", file=sys.stderr)
    YOLOController = None  # type: ignore[assignment]
    YOLOConfig = None  # type: ignore[assignment]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO controlled Angry Birds demo")
    parser.add_argument("--use-mouse", action="store_true", help="Use mouse controls instead of YOLO")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO model (if using YOLO controller)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to use for YOLO controller")
    parser.add_argument("--visualize", action="store_true", help="Show YOLO inference window")
    parser.add_argument(
        "--class",
        dest="class_id",
        type=int,
        default=None,
        help="Optional YOLO class id to use as the control object",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    controller = None

    if not args.use_mouse and YOLOController is not None and YOLOConfig is not None:
        try:
            controller = YOLOController(
                YOLOConfig(
                    model_path=args.model,
                    target_class=args.class_id,
                    camera_index=args.camera,
                ),
                visualize=args.visualize,
            )
        except RuntimeError as exc:
            print(f"Falling back to mouse controller: {exc}")
        except Exception as exc:
            print(f"Unexpected YOLO error, using mouse controller: {exc}")
    if controller is None:
        controller = MouseController()
    run_game(controller)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution entry
    raise SystemExit(main())
