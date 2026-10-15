#!/usr/bin/env python3
"""Generate every figure and data file the docs showcase page uses.

The page claims nothing that is not produced here by running this repository's
own detectors, so this script is the evidence for it. Regenerate with:

    python3 tools/generate_showcase_assets.py

Outputs
-------
docs/assets/pipeline-*.png   annotated detector output, drawn by src/ itself
docs/assets/demo-*.png       the author's captured sessions, downscaled
docs/assets/data.js          var SHOWCASE_DATA = {...} for the charts

The data layer (sweep_cascade, summarize_timings, to_data_js) is deliberately
free of file and model I/O so tests/test_showcase_generator.py can exercise it
without a camera, a model, or a GPU.
"""

import argparse
import json
import logging
import os
import statistics
import sys
import time
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

logger = logging.getLogger("showcase")

DOCS_ASSETS = os.path.join(REPO_ROOT, "docs", "assets")

# --- What the charts are drawn against -------------------------------------

# The frame the cascade sweep is measured on. Deliberately carries no declared
# face count: the cascade returns candidate regions, and on this photo the
# shipped defaults happen to return two regions while only one lands on a real
# face, so any "found N of M faces" claim drawn from a count alone would be
# wrong. The chart therefore plots parameter behaviour, not accuracy.
REFERENCE_IMAGES: List[Dict[str, Any]] = [
    {
        "name": "zidane.jpg",
        "label": "Two people on a touchline",
    },
]

# Rendered for the gallery, with no ground-truth claim attached.
GALLERY_IMAGES: List[Dict[str, Any]] = [
    {
        "name": "bus.jpg",
        "caption": "Object detection on a busier scene: one bus, three people.",
    },
]

# The cascade grid. scale_factor is the pyramid step; min_neighbors is how many
# overlapping hits a region needs before it counts as a face.
SWEEP_SCALE_FACTORS = [1.05, 1.10, 1.15, 1.20, 1.30, 1.50]
SWEEP_MIN_NEIGHBORS = [3, 4, 5, 6, 8]

# The four checkpoints already on disk. yolov8x is not shipped, so it is not
# benchmarked and the page does not mention it.
BENCHMARK_MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]
BENCHMARK_WARMUP = 5
BENCHMARK_REPETITIONS = 20


# --- Data layer (no I/O) ---------------------------------------------------


def mean(values: List[float]) -> float:
    """Arithmetic mean, rounded for display."""
    return round(statistics.fmean(values), 2)


def median(values: List[float]) -> float:
    """Median, rounded for display."""
    return round(statistics.median(values), 2)


def summarize_timings(timings: List[float]) -> Dict[str, float]:
    """Summarise a list of per-frame millisecond timings.

    The page reports the median: a laptop stalls on the occasional frame, and a
    single 1-second outlier would otherwise dominate the headline number.
    """
    return {
        "n": len(timings),
        "median_ms": median(timings),
        "mean_ms": mean(timings),
        "min_ms": round(min(timings), 2),
        "max_ms": round(max(timings), 2),
    }


def build_benchmark_record(
    model: str,
    size_mb: float,
    timings: List[float],
    per_frame_detections: List[int],
    classes_seen: Any,
) -> Dict[str, Any]:
    """Assemble one model's benchmark row.

    Detections are kept per frame rather than averaged: the frames differ in
    how much there is to find, so a single mean would blend two different
    questions into one unreadable number.
    """
    record = summarize_timings(timings)
    record.update(
        {
            "model": model,
            "size_mb": size_mb,
            "per_frame_detections": list(per_frame_detections),
            "detections_total": sum(per_frame_detections),
            "classes": sorted(classes_seen),
            "fps": round(1000.0 / record["median_ms"], 1) if record["median_ms"] else 0.0,
        }
    )
    return record


def fit_least_squares(xs: List[float], ys: List[float]) -> Optional[Dict[str, float]]:
    """Fit ``y = slope * x + intercept`` by ordinary least squares.

    The benchmark chart draws this line, so the coefficients are emitted with the
    data rather than written into the page: a hardcoded fit keeps drawing its old
    line after the measurements move, and a dashed trend line that no longer fits
    its points looks exactly like one that does.

    Returns unrounded values — formatting is the page's concern. Returns None when
    there is nothing to estimate: fewer than two points, or no spread on x. The
    page then omits the line instead of drawing a flat one, which would read as a
    finding rather than as the absence of one.
    """
    if len(xs) < 2 or len(xs) != len(ys):
        return None

    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return None

    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    intercept = mean_y - slope * mean_x

    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    # A perfect fit has ss_res 0 and ss_tot > 0; a constant y has ss_tot 0 and no
    # variance to explain, which counts as fully explained.
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0

    return {"slope": slope, "intercept": intercept, "r2": r2}


def round_fit(fit: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """Round a fit for emission, passing None through untouched."""
    if fit is None:
        return None
    return {
        "slope": round(fit["slope"], 3),
        "intercept": round(fit["intercept"], 3),
        "r2": round(fit["r2"], 4),
    }


def resolve_benchmark(
    fresh: List[Dict[str, Any]], previous: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Pick this run's timings, falling back to the previous run's.

    ``--skip-benchmark`` exists so figures can be regenerated without spending
    minutes re-timing the models. It must not take the timings with it: writing an
    empty benchmark blanks the chart the reader came for.
    """
    return list(fresh) if fresh else list(previous)


def load_existing_data(path: str) -> Dict[str, Any]:
    """Read back a previously generated data.js, for ``--skip-benchmark`` runs."""
    if not os.path.exists(path):
        return {}

    with open(path, encoding="utf-8") as handle:
        text = handle.read()

    prefix = "var SHOWCASE_DATA = "
    if not text.startswith(prefix):
        logger.warning("existing data file has an unexpected shape; ignoring it")
        return {}

    try:
        return json.loads(text[len(prefix) :].strip().rstrip(";"))
    except json.JSONDecodeError:
        logger.warning("could not parse the existing data file; ignoring it")
        return {}


def describe_machine() -> str:
    """Name the CPU as a reader would recognise it.

    platform.processor() reports just "arm" on Apple silicon, which tells a
    reader nothing about the machine a timing came from.
    """
    import platform
    import subprocess

    if platform.system() == "Darwin":
        try:
            brand = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout.strip()
            if brand:
                return brand
        except (OSError, subprocess.SubprocessError):
            pass

    return platform.processor() or platform.machine() or "unknown"


def sweep_cascade(
    detect: Callable[[Any, float, int], int],
    gray: Any,
    scale_factors: List[float],
    min_neighbors: List[int],
) -> List[Dict[str, Any]]:
    """Run the cascade across a grid of parameters.

    `detect(gray, scale_factor, min_neighbors)` returns a region count; passing
    it in keeps OpenCV out of the enumeration logic, which is the part under
    test. The count is deliberately named `detections` and not `faces` — these
    are candidate regions, and only some of them are faces.
    """
    records = []
    for scale_factor in scale_factors:
        for min_neighbors_ in min_neighbors:
            records.append(
                {
                    "scale_factor": scale_factor,
                    "min_neighbors": min_neighbors_,
                    "detections": detect(gray, scale_factor, min_neighbors_),
                }
            )
    return records


def to_data_js(data: Dict[str, Any]) -> str:
    """Emit the payload as an executable assignment.

    Not JSON on purpose: fetch() is blocked for file:// URLs, so a reader who
    clones the repo and opens docs/index.html would get empty charts.
    """
    payload = json.dumps(data, indent=2, sort_keys=False)
    # A literal closer would end the hosting <script> tag mid-string.
    payload = payload.replace("</script>", "<\\/script>")
    return "var SHOWCASE_DATA = " + payload + ";\n"


# --- Inputs ----------------------------------------------------------------


def resolve_reference_path(ref: Dict[str, Any]) -> Optional[str]:
    """Find an input image, preferring one committed to the repo.

    Falls back to the copies that ship inside the ultralytics package, so the
    generator works on a fresh clone without downloading anything.
    """
    local = os.path.join(REPO_ROOT, ref["name"])
    if os.path.exists(local):
        return local

    try:
        import ultralytics
    except ImportError:
        return None

    bundled = os.path.join(os.path.dirname(ultralytics.__file__), "assets", ref["name"])
    return bundled if os.path.exists(bundled) else None


# --- Asset generation ------------------------------------------------------


def write_annotated(frame: Any, detections: List[Dict[str, Any]], draw: Callable, path: str) -> None:
    """Draw real detections onto a frame and save it."""
    import cv2

    for detection in detections:
        frame = draw(frame, [detection])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, frame)
    logger.info("wrote %s (%d detections)", os.path.relpath(path, REPO_ROOT), len(detections))


def downscale_capture(src: str, dest: str, max_width: int = 1100) -> None:
    """Re-encode a captured screenshot to something a page can carry.

    The originals are ~3 MB RGBA screenshots; the page needs them an order of
    magnitude smaller, flattened to RGB.
    """
    import cv2
    from PIL import Image

    image = Image.open(src)
    if image.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        image = image.convert("RGBA")
        background.paste(image, mask=image.split()[-1])
        image = background
    else:
        image = image.convert("RGB")

    if image.width > max_width:
        ratio = max_width / image.width
        image = image.resize((max_width, round(image.height * ratio)), Image.LANCZOS)

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    image.save(dest, "PNG", optimize=True)
    logger.info("wrote %s %dx%d", os.path.relpath(dest, REPO_ROOT), image.width, image.height)


def run_cascade_sweep(face_detector: Any) -> List[Dict[str, Any]]:
    """Sweep the shipped defaults against the reference image."""
    import cv2

    ref = REFERENCE_IMAGES[0]
    path = resolve_reference_path(ref)
    if path is None:
        raise FileNotFoundError(f"reference image {ref['name']} not found")

    gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

    def detect(image: Any, scale_factor: float, min_neighbors: int) -> int:
        return len(
            face_detector.face_cascade.detectMultiScale(
                image,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
        )

    return sweep_cascade(detect, gray, SWEEP_SCALE_FACTORS, SWEEP_MIN_NEIGHBORS)


def run_model_benchmark(detector_cls: Any, frames: List[Any]) -> List[Dict[str, Any]]:
    """Time each checkpoint over the same fixed frames.

    Warm up first (the first few inferences pay for lazy allocation), then take
    one timing per frame per repetition and report the median.
    """
    results = []
    for model_name in BENCHMARK_MODELS:
        model_path = os.path.join(REPO_ROOT, model_name)
        if not os.path.exists(model_path):
            logger.warning("skipping %s (not present)", model_name)
            continue

        detector = detector_cls(model_path=model_path, confidence_threshold=0.5)
        if not detector.load_model():
            logger.warning("skipping %s (failed to load)", model_name)
            continue

        for _ in range(BENCHMARK_WARMUP):
            detector.detect_objects(frames[0])

        timings: List[float] = []
        per_frame: List[int] = [0] * len(frames)
        classes_seen: set = set()
        for _ in range(BENCHMARK_REPETITIONS):
            for index, frame in enumerate(frames):
                started = time.perf_counter()
                detections = detector.detect_objects(frame)
                timings.append((time.perf_counter() - started) * 1000.0)
                per_frame[index] = len(detections)
                classes_seen.update(d["class_name"] for d in detections)

        record = build_benchmark_record(
            model=model_name,
            size_mb=round(os.path.getsize(model_path) / (1024 * 1024), 1),
            timings=timings,
            per_frame_detections=per_frame,
            classes_seen=classes_seen,
        )
        results.append(record)
        logger.info(
            "%s: median %.1f ms (%.1f fps), found %s",
            model_name,
            record["median_ms"],
            record["fps"],
            ", ".join(record["classes"]) or "nothing",
        )

    return results


def build_data(
    cascade_sweep: List[Dict[str, Any]],
    benchmark: List[Dict[str, Any]],
    face_detector: Any,
    object_detector: Any,
    frames: List[Any],
) -> Dict[str, Any]:
    """Assemble everything the page reads."""
    import torch

    reference = REFERENCE_IMAGES[0]
    default_params = cascade_sweep_defaults(face_detector)

    return {
        "meta": {
            "generated_utc": time.strftime("%Y-%m-%d", time.gmtime()),
            "machine": describe_machine(),
            # Read after ultralytics is imported: it pins torch to a single
            # thread by default, so this is the thread count the timings
            # actually ran under, not the machine's core count.
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
            "cascade_repetitions": 1,
            "benchmark_warmup": BENCHMARK_WARMUP,
            "benchmark_repetitions": BENCHMARK_REPETITIONS,
            "benchmark_frames": len(frames),
        },
        "cascade_sweep": cascade_sweep,
        "cascade_reference": {
            "image": reference["name"],
            "label": reference["label"],
            "scale_factors": SWEEP_SCALE_FACTORS,
            "min_neighbors": SWEEP_MIN_NEIGHBORS,
            "default_scale_factor": default_params["scale_factor"],
            "default_min_neighbors": default_params["min_neighbors"],
            # Names what the y-axis counts, so a caption cannot quietly promote
            # candidate regions into confirmed faces.
            "units": "candidate regions returned",
        },
        "model_benchmark": benchmark,
        # Emitted, not hardcoded in the page: see fit_least_squares. rounded here
        # because this is the display boundary.
        "benchmark_fit": round_fit(
            fit_least_squares(
                [record["size_mb"] for record in benchmark],
                [record["median_ms"] for record in benchmark],
            )
        ),
        "detector_contract": {
            "keys": ["bbox", "confidence", "class_id", "class_name"],
        },
        "reference_detections": {
            "persons": len(object_detector.detect_objects(frames[0])),
        },
    }


def cascade_sweep_defaults(face_detector: Any) -> Dict[str, float]:
    """Read the parameters the shipped code actually uses.

    Parsed from source rather than restated here, so the chart's "default"
    marker cannot drift away from src/face_detector.py.
    """
    import re

    source = os.path.join(REPO_ROOT, "src", "face_detector.py")
    with open(source, encoding="utf-8") as handle:
        text = handle.read()

    scale_factor = re.search(r"scaleFactor=([0-9.]+)", text)
    min_neighbors = re.search(r"minNeighbors=([0-9]+)", text)
    if not scale_factor or not min_neighbors:
        raise ValueError("could not read cascade defaults from src/face_detector.py")

    return {
        "scale_factor": float(scale_factor.group(1)),
        "min_neighbors": int(min_neighbors.group(1)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help=(
            "regenerate figures and charts using the timings already in docs/assets/data.js; "
            "the model sweep takes a few minutes"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Ultralytics logs a banner and a per-inference block; both drown out the
    # generator's own output, which is the record of what the page claims.
    try:
        from ultralytics.utils import LOGGER as ultralytics_logger

        ultralytics_logger.setLevel(logging.ERROR)
    except ImportError:
        pass

    import cv2

    from src.face_detector import FaceDetector
    from src.object_detector import ObjectDetector

    os.makedirs(DOCS_ASSETS, exist_ok=True)

    face_detector = FaceDetector()
    object_detector = ObjectDetector(model_path="yolov8n.pt", confidence_threshold=0.5)
    if not object_detector.load_model():
        logger.error("could not load yolov8n.pt; run `make setup` first")
        return 1

    # Reference frame: both detectors' output drawn by the repo's own code.
    reference_path = resolve_reference_path(REFERENCE_IMAGES[0])
    reference_frame = cv2.imread(reference_path)
    write_annotated(
        reference_frame.copy(),
        object_detector.detect_objects(reference_frame),
        object_detector.draw_detections,
        os.path.join(DOCS_ASSETS, "pipeline-zidane-objects.png"),
    )

    # Every frame both detectors see, reused as the benchmark set so the timings
    # and the pictures describe the same workload.
    benchmark_frames = [reference_frame]

    for gallery in GALLERY_IMAGES:
        path = resolve_reference_path(gallery)
        if path is None:
            logger.warning("gallery image %s not found", gallery["name"])
            continue
        frame = cv2.imread(path)
        benchmark_frames.append(frame)
        combined = object_detector.draw_detections(
            frame.copy(), object_detector.detect_objects(frame)
        )
        write_annotated(
            combined,
            face_detector.detect_faces(frame),
            face_detector.draw_faces,
            os.path.join(DOCS_ASSETS, "pipeline-bus.png"),
        )

    # The author's captured sessions, kept as captured.
    for src_name, dest_name in [
        ("demo1.png", "demo-object-detection.png"),
        ("demo2.png", "demo-face-detection.png"),
    ]:
        src_path = os.path.join(REPO_ROOT, src_name)
        if os.path.exists(src_path):
            downscale_capture(src_path, os.path.join(DOCS_ASSETS, dest_name))
        else:
            logger.warning("%s not found; skipping", src_name)

    cascade_sweep = run_cascade_sweep(face_detector)
    logger.info("cascade sweep: %d cells", len(cascade_sweep))

    data_path = os.path.join(DOCS_ASSETS, "data.js")
    benchmark: List[Dict[str, Any]] = []
    if args.skip_benchmark:
        # Carry the previous timings forward rather than writing an empty list:
        # this flag is for regenerating figures quickly, not for discarding the
        # measurements those figures describe.
        previous = load_existing_data(data_path).get("model_benchmark", [])
        benchmark = resolve_benchmark([], previous)
        if benchmark:
            logger.info("benchmark preserved from the existing data file (%d models)", len(benchmark))
        else:
            logger.warning(
                "--skip-benchmark given but no usable data file to preserve timings from; "
                "the benchmark chart will be empty"
            )
    else:
        benchmark = run_model_benchmark(ObjectDetector, benchmark_frames)

    data = build_data(cascade_sweep, benchmark, face_detector, object_detector, benchmark_frames)

    with open(data_path, "w", encoding="utf-8") as handle:
        handle.write(to_data_js(data))
    logger.info("wrote %s", os.path.relpath(data_path, REPO_ROOT))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
