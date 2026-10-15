# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A real-time webcam detection application. It captures frames, runs object detection
(YOLOv8 via ultralytics) and face detection (OpenCV Haar cascade) on each one, annotates
them, then displays or writes them out.

The core is ~460 lines across four classes in `src/`. The design is the interesting part,
not the volume.

## Commands

```bash
make setup                                        # pip install -r requirements.txt
make run                                          # python main.py
make test                                         # python -m pytest tests/ -v
make format                                       # black src/ tests/ *.py
make lint                                         # flake8 src/ tests/ *.py
make clean                                        # drop __pycache__, logs, build artifacts

python main.py --camera-id 0 --model-path yolov8n.pt --confidence 0.5
python main.py --no-display --save-output out.mp4  # headless, file output
python main.py --log-level DEBUG
```

`tests/` runs without a camera, a model, or a GPU, in about a second. Keep it that way —
that property is what makes the suite worth running on every edit.

## Architecture

`src/smart_detection_app.py` owns the loop; everything else is a collaborator.

```
CameraHandler ──frame──▶ ObjectDetector ─┐
                         FaceDetector    ─┴─▶ draw ─▶ imshow / VideoWriter
```

- `src/camera_handler.py` — capture, release, `get_camera_info()`
- `src/object_detector.py` — YOLOv8 wrapper; CUDA when available, else CPU
- `src/face_detector.py` — Haar cascade wrapper
- `src/smart_detection_app.py` — the loop, FPS logging, cleanup path

**The load-bearing invariant:** both detectors return a list of dicts with the same four
keys — `bbox`, `confidence`, `class_id`, `class_name`. That is why the loop and the drawing
code are detector-agnostic. A new detector should match that contract rather than invent
its own.

One consequence worth knowing before you "fix" it: a Haar cascade computes no score at all,
so `FaceDetector` assigns a fixed `confidence` purely to satisfy the shared contract. That
number is identical on every face and carries no meaning. It is not a calibration bug.

## The `docs/` showcase page

`docs/index.html` is a published GitHub Pages site (https://pengzhao-ai.github.io/webcam-smartDet/).
**`docs/assets/` is generated output — never hand-edit it.** Regenerate from the repo's own
detectors:

```bash
python3 tools/generate_showcase_assets.py                    # everything, ~2 min
python3 tools/generate_showcase_assets.py --skip-benchmark   # figures and charts only
```

`tools/generate_showcase_assets.py` is the evidence for every number the page states. Change
detection behaviour and you must regenerate, or the page will describe code that no longer
exists. `tests/test_showcase_generator.py` pins the values the charts annotate, so drift
fails loudly instead of silently redrawing.

Verify the page by rendering it, not by reading it — a surprising share of layout and chart
defects are invisible in the source:

```bash
python3 ~/.claude/skills/codebase-showcase/assets/verify_showcase.py \
  docs/index.html --expect-repo pengzhao-ai/webcam-smartDet
```

Always pass `--expect-repo`. Without it the harness infers the repo from `git remote -v`,
which is exactly the unreliable source it exists to warn about.

## Traps

- **The canonical repo is `pengzhao-ai/webcam-smartDet`.** Use it in every link and URL.
  `git remote -v` previously held a stale org name that GitHub silently redirects.
- **The MIT claim is wrong and appears in three places** — `README.md:210`, the
  `pyproject.toml` classifier `License :: OSI Approved :: MIT License`, and placeholder
  `Your Name` / `yourusername` metadata. `LICENSE` is **GPL-2.0** and is authoritative.
  `pyproject.toml` points `license = {file = "LICENSE"}` at it, so that file contradicts
  itself. Fixing this is a separate change with its own decision — do not bundle it into
  unrelated work.
- **`README.md:28`'s clone URL is still the placeholder `<your-repo-url>`.**
- **`examples/model_performance_test.py:93` looks for weights in `models/`, but they sit in
  the repo root**, so `compare_all_models()` reports "not found" for every model.
- Weights are gitignored (`*.pt`); ultralytics re-downloads on first use, so a fresh clone
  works but the first run is slow.
- Running the app needs a real camera. The detectors do not — for headless work call
  `detect_objects` / `detect_faces` on a still image directly, which is what the asset
  generator does.

## Conventions

- `black` at line length 88, `flake8` for linting (both configured in `pyproject.toml`)
- Type hints on public functions; docstrings on modules and classes
- Comments explain *why*. The existing code follows this — match it.
- Python 3.8+ (`requires-python = ">=3.8"`), so no `match`, no `X | Y` type unions at
  runtime
