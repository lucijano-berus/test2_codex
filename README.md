# YOLOv11 Game Loop

This repository demonstrates how to embed the Ultralytics YOLOv11 detection
loop inside a lightweight, controllable game context. The mini-game projects
bounding boxes for the detected objects onto the video stream, keeps the class
labels/confidence overlays consistent with the standard Ultralytics snippet, and
lets testers drive the gameplay with keyboard commands while detections feed a
simple scoring mechanic.

## Features

- Configurable Ultralytics YOLOv11 inference using the familiar detection loop.
- Realtime drawing of bounding boxes and labels using `cvzone` styling.
- Pygame-powered keyboard mappings (WASD/arrow keys + space) that update the
  on-screen avatar and scoring.
- Support for webcams (`--source 0`) or video/image files passed as the source.
- Optional mirroring, custom capture resolution, warmup passes, and class
  filtering to tailor evaluation sessions.

## Requirements

- Python 3.9+
- [Ultralytics YOLOv11 weights](https://docs.ultralytics.com/models/yolo11/).
  Download the desired `.pt` file (for example `yolo11n.pt`) and note its path.

Install Python dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

The core dependencies include `ultralytics`, `opencv-python`, `cvzone`,
`pygame`, and `numpy` for array manipulation.

## Running the Game

Launch the loop directly with Python:

```bash
python -m game.main --model-path /path/to/yolo11n.pt --source 0 --mirror
```

Command-line flags of interest:

- `--source`: Webcam index (e.g., `0`, `1`) or path to a video/image file.
- `--model-path`: Path to the YOLOv11 weights to load (defaults to `yolo11n.pt`).
- `--confidence`: Confidence threshold (default `0.35`).
- `--width` / `--height`: Request a specific capture resolution.
- `--mirror`: Mirror frames for a selfie-like view.
- `--tracked-classes`: List of class names to count towards the score.
- `--warmup`: Number of warmup frames to run before gameplay begins.
- `--no-window`: Skip the OpenCV output window for headless runs.

## Controls and Gameplay

- **Arrow keys / WASD**: Move the on-screen avatar.
- **Space**: Manually increment the score (useful for validating the control
  flow without detections).
- **Q** (while the OpenCV window is focused) or **Esc**: Exit the loop.

Every frame processes YOLO detections, draws class-aligned bounding boxes, and
feeds the detected class names into the scoring system. When a detection matches
(optional) tracked classes, the score increases and a status message is shown.
Manual keyboard commands operate alongside detections so testers can confirm the
combined YOLO/manual-control flow works as expected.

## Video Source Configuration

By default the script opens the first webcam (`--source 0`). To evaluate with a
video file instead, provide the path to the file:

```bash
python -m game.main --model-path /weights/yolo11n.pt --source demo.mp4
```

If your capture device requires a specific resolution, use the `--width` and
`--height` flags. Set `--mirror` for mirrored webcam previews.

## Troubleshooting

- Ensure the YOLO weights path is correct and accessible. If the model fails to
  load, verify that the `.pt` file matches your Ultralytics installation.
- When running inside a headless environment, add `--no-window` to disable the
  OpenCV viewer and rely on logs or custom instrumentation.
- Depending on your GPU/CPU resources, you may want to try a lighter model
  variant such as `yolo11n.pt` or lower the capture resolution.

With the dependencies installed and the commands above, you can validate how the
YOLO detections and manual controls interact within the mini-game loop.
