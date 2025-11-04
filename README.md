# 🏀 Basketball Tracking

> **What to use:** The **most optimized and accurate weights are `best.pt` in the repository’s *Releases* section**. Everything else in this repo is mostly **research material / drafts** and may change.

---

## TL;DR

* Download **`best.pt`** from the **Releases** page of this repo.
* Run with Ultralytics YOLO tracking (BoT-SORT / ByteTrack supported).
* Outputs annotated video.

---

## Features

* **Ball detection & tracking** using Ultralytics YOLO.
* **Multi-object tracking** via **BoT-SORT** / **ByteTrack** (YAML configs supported).
* **Court utilities** (optional scripts) for region/court-aware logic.
* **Simple exports** (videos) for analytics pipelines.

> ⚠️ **Repo status:** scripts, configs, and extra weights here are primarily **experimental**. For production/inference, prefer **`best.pt` from Releases**.

---

## Getting Started

### 1) Environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install ultralytics opencv-python numpy
# If PyTorch is not present, install the build that matches your OS/CUDA
```

### 3) Get the weights

* Go to this repository’s **Releases** page.
* Download **`best.pt`** to your project (e.g., `weights/best.pt`).

> Tip: Keep model files out of Git history; prefer Releases over committing large binaries.

---

## Usage

You can use either the Ultralytics **CLI** or **Python API**.

### A) CLI — quick start

```bash
# Track on a video with BoT-SORT
yolo track model=weights/best.pt source=path/to/video.mp4 tracker=botsort.yaml save=True

# Or with ByteTrack
yolo track model=weights/best.pt source=path/to/video.mp4 tracker=bytetrack.yaml save=True
```

Common flags: `conf=0.25`, `iou=0.5`, `imgsz=1280`, `device=0` (GPU), `show=True/False`.

### B) Python API — programmatic

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")
results = model.track(
    source="path/to/video.mp4",
    tracker="botsort.yaml",   # or "bytetrack.yaml"
    conf=0.25,
    iou=0.5,
    imgsz=1280,
    device=0,
    save=True,
)
```

---

## Outputs

* **Annotated video**: saved to `runs/track/...` by Ultralytics.

---

## Tips & Notes

* Prefer **Releases** for distributing models; avoid pushing large files to Git.
* Tracker behavior can be tuned in `botsort.yaml` / `bytetrack.yaml`.
* If you add player detection later, you can associate ball-to-player via proximity heuristics or re-ID.

---

## Roadmap (nice-to-have)

* [ ] Sample clip + demo GIF in README
* [ ] Document JSON schema for analytics exports
* [ ] Provide `requirements.txt` / pinned versions
* [ ] Minimal notebook for quick experimentation

---

## Contributing

PRs and issues welcome. Because much of the code is experimental, please include a clear description, repro steps, and before/after results where applicable.

## License

No license 
