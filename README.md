# YOLO Controlled Angry Birds Demo

This project implements a simplified Angry Birds-style experience that can be
controlled either with a traditional mouse drag or by using a YOLO v8/v11 object
tracking pipeline. The goal is to provide an accessible sandbox for
experimenting with computer-vision-driven gameplay.

## Features

- 2D physics-lite projectile gameplay written with `pygame`.
- Sample level inspired by the classic Angry Birds arrangement.
- Support for YOLO-based input using the `ultralytics` models (v8 or the newer
  v11 series). The bounding box of a tracked object determines the slingshot's
  aim and pullback strength.
- Mouse fallback controller for development without a camera.
- Demonstration of bird abilities (dash and explosion) triggered via keyboard or
  right-click when in flight.

## Installation

1. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\\Scripts\\activate   # Windows PowerShell
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** Installing `ultralytics` pulls in the YOLO models and OpenCV. If
   > you only want to try the mouse controller you can skip installing the
   > computer-vision dependencies and run with `--use-mouse`.

## Running the Game

```bash
python main.py [--use-mouse] [--model PATH] [--camera INDEX] [--visualize] [--class CLASS_ID]
```

- `--use-mouse`: Skip YOLO input and drag with the mouse instead.
- `--model`: Path to a YOLO model supported by Ultralytics (defaults to
  `yolov8n.pt`). You can also pass a YOLOv11 checkpoint.
- `--camera`: Index of the camera device (defaults to `0`).
- `--visualize`: Displays the YOLO inference window with aim/pullback overlays.
- `--class`: Optional integer class id to filter detections (useful if your
  model identifies multiple objects).

When using the YOLO controller point the tracked object (for example, a colorful
card) at the camera. Moving it vertically adjusts the launch angle, and changing
its size relative to the camera (moving closer/further) adjusts pullback power.
Removing the object from view releases the slingshot. Trigger a bird's ability
with the space bar or by right-clicking during flight.

## Controls Summary

| Input                         | Action                                       |
| ----------------------------- | -------------------------------------------- |
| Move tracked object           | Aim slingshot (YOLO controller)              |
| Remove object from camera     | Launch current bird                          |
| Left mouse drag               | Aim/launch (mouse controller)                |
| Space / Right mouse button    | Activate bird ability (dash/explosion)       |
| Escape                        | Quit                                         |

## Project Structure

```
angrybirds/
├── controller.py       # Shared control protocol and state dataclass
├── game.py             # Pygame implementation of the game loop
├── levels.py           # Level definition and entity materials
├── mouse_controller.py # Mouse fallback controller
└── yolo_controller.py  # YOLO based vision controller
main.py                 # Entry point selecting the controller
requirements.txt        # Python dependencies
```

## Future Extensions

- Add more complex physics (e.g., Box2D) for richer interactions.
- Expand the level roster and bird ability roster.
- Train a custom YOLO model for gesture-based ability activation.

Enjoy experimenting with computer-vision-powered slingshot gameplay!
