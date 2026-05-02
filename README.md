# 🎮 Gesture-Controlled Game Controller

Control endless runner games (like Subway Surfers) using your body — no keyboard or mouse needed. This project uses your webcam and MediaPipe's pose detection to translate real-time body movements into game inputs.

---

## How It Works

The app captures your webcam feed, detects 33 body landmarks using MediaPipe's Pose Landmarker model, and maps your movements to keyboard arrow keys in real time.

| Body Movement | Game Action | Key Pressed |
|---|---|---|
| Lean body **Left** | Move lane left | ← Left Arrow |
| Lean body **Right** | Move lane right | → Right Arrow |
| **Jump** (raise torso up) | Jump | ↑ Up Arrow |
| **Crouch** (lower torso) | Slide/Duck | ↓ Down Arrow |
| **Join both hands** (hold ~10 frames) | Start game / Pause | Space / Click |

---

## Features

- **MediaPipe Tasks API** — Uses the modern Tasks API with `pose_landmarker_full.task` (falls back to `pose_landmarker_lite.task` if unavailable).
- **EMA Landmark Smoothing** — Exponential Moving Average filter reduces jitter in landmark positions between frames.
- **Majority-Vote Gesture Smoothing** — Buffers recent gesture decisions across a 5-frame window to eliminate single-frame flickers.
- **Cooldown Timer** — Prevents accidental rapid-fire keypresses (350ms cooldown per action).
- **Dead-Zone Detection** — Left/right and jump/crouch detection includes dead zones around center to prevent false triggers.
- **Torso-Center Calibration** — On game start, calibrates the neutral Y position using the average of both shoulders and both hips for stability.
- **Pose Classification** — Detects named poses: T Pose, Warrior II, and Tree Pose.
- **Live FPS Display** — Shows real-time frame rate on the webcam feed.

---

## Project Structure

```
gesture/
├── main.py                      # Main application script
├── pose_landmarker_full.task    # MediaPipe full pose model (~9 MB, higher accuracy)
├── pose_landmarker_lite.task    # MediaPipe lite pose model (~5.5 MB, faster)
└── reqirenement.txt             # Python dependencies
```

---

## Requirements

- Python 3.8+
- A working webcam
- Good lighting and enough space to move

### Dependencies

Install with pip:

```bash
pip install opencv-python numpy mediapipe pyautogui
```

Or install from the included requirements file:

```bash
pip install -r reqirenement.txt
```

---

## Setup & Running

1. **Clone or download** the project folder.
2. **Install dependencies** (see above).
3. **Run the script:**

```bash
python main.py
```

4. **Stand in front of your webcam** so your full upper body (shoulders, hips, wrists) is visible.
5. **Join both hands together** and hold for ~10 frames — the game will start automatically.
6. **Move your body** to control the game!

Press `Esc` or `Q` to quit.

---

## Tips for Best Performance

- **Lighting** — Make sure you're well-lit from the front. Avoid strong backlighting.
- **Distance** — Stand 1.5–2.5 metres from the webcam so your full torso is in frame.
- **Neutral stance** — Stand still and upright when the game starts so the Y calibration is accurate.
- **Lean, don't step** — Left/right detection tracks your body's horizontal centre, so leaning is enough — you don't need to sidestep.
- **Jump/Crouch** — Raise up on your toes or physically jump for "jump"; squat or bend your knees for "crouch".

---

## Configuration

Key parameters can be tuned inside `main.py`:

| Variable | Default | Description |
|---|---|---|
| `LandmarkSmoother(alpha=0.5)` | `0.5` | EMA smoothing strength (0 = no smoothing, 1 = no lag) |
| `GestureSmoother(window=5)` | `5` | Frames buffered for majority-vote gesture |
| `CooldownTimer(cooldown_sec=0.35)` | `0.35s` | Minimum time between repeated keypresses |
| `min_detection_confidence` | `0.6` | Minimum confidence for pose detection |
| Dead zone (left/right) | 8% of frame width | Prevents left/right flicker near centre |
| Dead zone (jump/crouch) | ±25px / +50px from `MID_Y` | Prevents false jump/crouch triggers |

---

## Troubleshooting

**Camera not detected**
Make sure no other app is using the webcam, then re-run the script.

**Pose not detected / low accuracy**
Improve lighting, move closer to the camera, and ensure your torso is fully visible.

**Actions firing too fast or too slow**
Adjust `cooldown_sec` in `CooldownTimer` (increase to slow down, decrease to speed up).

**Jump/Crouch triggers randomly**
Re-run the script and calibrate from a neutral standing position. Adjust the dead-zone bounds in `checkJumpCrouch()` if needed.

**Model file not found**
Ensure `pose_landmarker_full.task` (or `pose_landmarker_lite.task`) is in the same directory as `main.py`.

---

## License

This project is provided as-is for personal and educational use.
