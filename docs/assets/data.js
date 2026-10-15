var SHOWCASE_DATA = {
  "meta": {
    "generated_utc": "2026-10-15",
    "machine": "Apple M3 Max",
    "torch_threads": 1,
    "device": "cpu",
    "cascade_repetitions": 1,
    "benchmark_warmup": 5,
    "benchmark_repetitions": 20,
    "benchmark_frames": 2
  },
  "cascade_sweep": [
    {
      "scale_factor": 1.05,
      "min_neighbors": 3,
      "detections": 3
    },
    {
      "scale_factor": 1.05,
      "min_neighbors": 4,
      "detections": 3
    },
    {
      "scale_factor": 1.05,
      "min_neighbors": 5,
      "detections": 3
    },
    {
      "scale_factor": 1.05,
      "min_neighbors": 6,
      "detections": 2
    },
    {
      "scale_factor": 1.05,
      "min_neighbors": 8,
      "detections": 2
    },
    {
      "scale_factor": 1.1,
      "min_neighbors": 3,
      "detections": 2
    },
    {
      "scale_factor": 1.1,
      "min_neighbors": 4,
      "detections": 2
    },
    {
      "scale_factor": 1.1,
      "min_neighbors": 5,
      "detections": 2
    },
    {
      "scale_factor": 1.1,
      "min_neighbors": 6,
      "detections": 2
    },
    {
      "scale_factor": 1.1,
      "min_neighbors": 8,
      "detections": 2
    },
    {
      "scale_factor": 1.15,
      "min_neighbors": 3,
      "detections": 2
    },
    {
      "scale_factor": 1.15,
      "min_neighbors": 4,
      "detections": 2
    },
    {
      "scale_factor": 1.15,
      "min_neighbors": 5,
      "detections": 2
    },
    {
      "scale_factor": 1.15,
      "min_neighbors": 6,
      "detections": 2
    },
    {
      "scale_factor": 1.15,
      "min_neighbors": 8,
      "detections": 2
    },
    {
      "scale_factor": 1.2,
      "min_neighbors": 3,
      "detections": 1
    },
    {
      "scale_factor": 1.2,
      "min_neighbors": 4,
      "detections": 1
    },
    {
      "scale_factor": 1.2,
      "min_neighbors": 5,
      "detections": 1
    },
    {
      "scale_factor": 1.2,
      "min_neighbors": 6,
      "detections": 1
    },
    {
      "scale_factor": 1.2,
      "min_neighbors": 8,
      "detections": 1
    },
    {
      "scale_factor": 1.3,
      "min_neighbors": 3,
      "detections": 2
    },
    {
      "scale_factor": 1.3,
      "min_neighbors": 4,
      "detections": 2
    },
    {
      "scale_factor": 1.3,
      "min_neighbors": 5,
      "detections": 1
    },
    {
      "scale_factor": 1.3,
      "min_neighbors": 6,
      "detections": 1
    },
    {
      "scale_factor": 1.3,
      "min_neighbors": 8,
      "detections": 1
    },
    {
      "scale_factor": 1.5,
      "min_neighbors": 3,
      "detections": 1
    },
    {
      "scale_factor": 1.5,
      "min_neighbors": 4,
      "detections": 1
    },
    {
      "scale_factor": 1.5,
      "min_neighbors": 5,
      "detections": 1
    },
    {
      "scale_factor": 1.5,
      "min_neighbors": 6,
      "detections": 1
    },
    {
      "scale_factor": 1.5,
      "min_neighbors": 8,
      "detections": 0
    }
  ],
  "cascade_reference": {
    "image": "zidane.jpg",
    "label": "Two people on a touchline",
    "scale_factors": [
      1.05,
      1.1,
      1.15,
      1.2,
      1.3,
      1.5
    ],
    "min_neighbors": [
      3,
      4,
      5,
      6,
      8
    ],
    "default_scale_factor": 1.1,
    "default_min_neighbors": 5,
    "units": "candidate regions returned"
  },
  "model_benchmark": [
    {
      "n": 40,
      "median_ms": 34.23,
      "mean_ms": 34.32,
      "min_ms": 29.42,
      "max_ms": 39.36,
      "model": "yolov8n.pt",
      "size_mb": 6.2,
      "per_frame_detections": [
        2,
        4
      ],
      "detections_total": 6,
      "classes": [
        "bus",
        "person"
      ],
      "fps": 29.2
    },
    {
      "n": 40,
      "median_ms": 68.73,
      "mean_ms": 68.61,
      "min_ms": 60.64,
      "max_ms": 76.95,
      "model": "yolov8s.pt",
      "size_mb": 21.5,
      "per_frame_detections": [
        3,
        5
      ],
      "detections_total": 8,
      "classes": [
        "bus",
        "person",
        "tie"
      ],
      "fps": 14.5
    },
    {
      "n": 40,
      "median_ms": 122.4,
      "mean_ms": 122.72,
      "min_ms": 108.35,
      "max_ms": 139.6,
      "model": "yolov8m.pt",
      "size_mb": 49.7,
      "per_frame_detections": [
        4,
        5
      ],
      "detections_total": 9,
      "classes": [
        "bus",
        "person",
        "tie"
      ],
      "fps": 8.2
    },
    {
      "n": 40,
      "median_ms": 196.2,
      "mean_ms": 196.13,
      "min_ms": 172.9,
      "max_ms": 220.15,
      "model": "yolov8l.pt",
      "size_mb": 83.7,
      "per_frame_detections": [
        3,
        5
      ],
      "detections_total": 8,
      "classes": [
        "bus",
        "person",
        "tie"
      ],
      "fps": 5.1
    }
  ],
  "benchmark_fit": {
    "slope": 2.069,
    "intercept": 22.063,
    "r2": 0.9992
  },
  "detector_contract": {
    "keys": [
      "bbox",
      "confidence",
      "class_id",
      "class_name"
    ]
  },
  "reference_detections": {
    "persons": 2
  }
};
