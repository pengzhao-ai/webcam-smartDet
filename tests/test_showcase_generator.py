"""Tests for the showcase asset generator's data layer.

The generator writes the figures and the data file the docs page reads, so a bug
here shows up as a wrong number on a public page rather than as an exception.
These tests cover the parts that can be wrong silently: the sweep grid, the
statistic choice, and the shape of the emitted JavaScript.
"""

import importlib.util
import json
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATOR_PATH = os.path.join(REPO_ROOT, "tools", "generate_showcase_assets.py")

sys.path.insert(0, REPO_ROOT)


def load_generator():
    """Import the generator module from its path (tools/ is not a package)."""
    spec = importlib.util.spec_from_file_location("generate_showcase_assets", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCascadeSweep(unittest.TestCase):
    """The sweep that produces the cascade-sensitivity chart."""

    @classmethod
    def setUpClass(cls):
        cls.gen = load_generator()

    def test_sweep_covers_every_combination_in_the_grid(self):
        """Every (scale_factor, min_neighbors) pair gets exactly one record."""
        scale_factors = [1.05, 1.10, 1.20]
        min_neighbors = [3, 5, 8]
        seen = []

        def fake_detect(gray, scale_factor, min_neighbors_):
            seen.append((scale_factor, min_neighbors_))
            return 0

        records = self.gen.sweep_cascade(fake_detect, None, scale_factors, min_neighbors)

        self.assertEqual(
            sorted(seen), sorted((sf, mn) for sf in scale_factors for mn in min_neighbors)
        )
        self.assertEqual(len(records), len(scale_factors) * len(min_neighbors))

    def test_sweep_records_echo_the_parameters_and_the_detected_count(self):
        """Each record is self-describing, so the chart needs no positional assumptions."""

        def fake_detect(gray, scale_factor, min_neighbors):
            return 7

        records = self.gen.sweep_cascade(fake_detect, None, [1.10], [5])

        self.assertEqual(
            records[0],
            {"scale_factor": 1.10, "min_neighbors": 5, "detections": 7},
        )

    def test_sweep_survives_a_detector_that_finds_nothing(self):
        """A zero-detection cell is data, not an error — it must reach the chart."""

        def fake_detect(gray, scale_factor, min_neighbors):
            return 0

        records = self.gen.sweep_cascade(fake_detect, None, [1.50], [8])

        self.assertEqual(records[0]["detections"], 0)

    def test_sweep_result_is_never_labelled_as_a_face_count(self):
        """The cascade returns candidate regions; only some are faces.

        A key called "faces" invites a caption claiming the chart tracks
        accuracy, which this measurement cannot support.
        """
        records = self.gen.sweep_cascade(lambda g, s, m: 2, None, [1.10], [5])

        self.assertNotIn("faces", records[0])


class TestStatistics(unittest.TestCase):
    """Timing summary. The page states which statistic it reports, so it must be that one."""

    @classmethod
    def setUpClass(cls):
        cls.gen = load_generator()

    def test_median_resists_an_outlier_that_drags_the_mean(self):
        """A single stalled frame is common on a laptop; it must not move the headline."""
        timings = [10.0, 10.0, 10.0, 10.0, 1000.0]

        self.assertEqual(self.gen.median(timings), 10.0)
        self.assertGreater(self.gen.mean(timings), 100.0)

    def test_median_of_an_even_count_averages_the_middle_pair(self):
        self.assertEqual(self.gen.median([1.0, 2.0, 3.0, 4.0]), 2.5)

    def test_timing_summary_reports_every_advertised_field(self):
        summary = self.gen.summarize_timings([4.0, 2.0, 6.0])

        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["median_ms"], 4.0)
        self.assertEqual(summary["min_ms"], 2.0)
        self.assertEqual(summary["max_ms"], 6.0)

    def test_timing_summary_rounds_every_figure_it_reports(self):
        """Raw perf_counter values carry 15 digits; the page quotes two."""
        summary = self.gen.summarize_timings([29.150041999855603, 29.906333000326413])

        for field in ("median_ms", "mean_ms", "min_ms", "max_ms"):
            with self.subTest(field=field):
                self.assertEqual(summary[field], round(summary[field], 2))


class TestBenchmarkRecord(unittest.TestCase):
    """The per-model record is what the benchmark chart and table both read."""

    @classmethod
    def setUpClass(cls):
        cls.gen = load_generator()

    def test_record_keeps_detections_per_frame_so_a_mixed_set_stays_readable(self):
        record = self.gen.build_benchmark_record(
            model="yolov8n.pt",
            size_mb=6.2,
            timings=[30.0, 30.0],
            per_frame_detections=[2, 4],
            classes_seen={"person", "bus"},
        )

        self.assertEqual(record["per_frame_detections"], [2, 4])
        self.assertEqual(record["detections_total"], 6)
        self.assertEqual(record["classes"], ["bus", "person"])

    def test_record_derives_throughput_from_the_median_not_the_mean(self):
        """One stalled frame must not move the fps the page advertises."""
        record = self.gen.build_benchmark_record(
            model="yolov8n.pt",
            size_mb=6.2,
            timings=[20.0, 20.0, 20.0, 20.0, 2000.0],
            per_frame_detections=[2],
            classes_seen={"person"},
        )

        self.assertEqual(record["median_ms"], 20.0)
        self.assertEqual(record["fps"], 50.0)


class TestDataJsEmission(unittest.TestCase):
    """The page loads this file with a <script> tag, never fetch() — see gotcha 6."""

    @classmethod
    def setUpClass(cls):
        cls.gen = load_generator()

    def test_emitted_file_is_a_var_assignment_not_a_bare_json_payload(self):
        text = self.gen.to_data_js({"a": 1})

        self.assertTrue(text.startswith("var SHOWCASE_DATA = "))
        self.assertTrue(text.rstrip().endswith(";"))

    def test_emitted_payload_round_trips_as_json(self):
        data = {"cascade_sweep": [{"scale_factor": 1.1, "faces": 2}]}
        text = self.gen.to_data_js(data)

        payload = text[len("var SHOWCASE_DATA = ") :].rstrip().rstrip(";")
        self.assertEqual(json.loads(payload), data)

    def test_emitted_file_cannot_terminate_the_hosting_script_tag(self):
        """A '</script>' inside a string would break the page when inlined."""
        text = self.gen.to_data_js({"note": "a </script> sequence"})

        self.assertNotIn("</script>", text)


class TestLeastSquares(unittest.TestCase):
    """The benchmark chart draws a fitted trend line, so the coefficients it draws
    must come from the data. Hardcoding them in the page lets the line drift away
    from the points the moment the benchmark is regenerated on another machine —
    and a dashed line that no longer fits is indistinguishable from one that does.
    """

    @classmethod
    def setUpClass(cls):
        cls.gen = load_generator()

    def test_recovers_a_line_it_was_given_exactly(self):
        xs = [6.2, 21.5, 49.7, 83.7]
        ys = [22.06 + 2.07 * x for x in xs]

        fit = self.gen.fit_least_squares(xs, ys)

        self.assertAlmostEqual(fit["slope"], 2.07, places=6)
        self.assertAlmostEqual(fit["intercept"], 22.06, places=6)
        self.assertAlmostEqual(fit["r2"], 1.0, places=9)

    def test_matches_a_hand_computed_fit(self):
        """x=[1,2,3], y=[1,2,4]: slope 1.5, intercept -2/3, r^2 27/28."""
        fit = self.gen.fit_least_squares([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])

        self.assertAlmostEqual(fit["slope"], 1.5, places=9)
        self.assertAlmostEqual(fit["intercept"], -2.0 / 3.0, places=9)
        self.assertAlmostEqual(fit["r2"], 27.0 / 28.0, places=9)

    def test_r2_falls_between_zero_and_one_for_noisy_input(self):
        fit = self.gen.fit_least_squares([1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 2.0, 9.0])

        self.assertGreater(fit["r2"], 0.0)
        self.assertLess(fit["r2"], 1.0)

    def test_refuses_a_degenerate_axis_rather_than_dividing_by_zero(self):
        """One checkpoint, or several of the same size, has no slope to estimate."""
        self.assertIsNone(self.gen.fit_least_squares([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]))
        self.assertIsNone(self.gen.fit_least_squares([1.0], [1.0]))
        self.assertIsNone(self.gen.fit_least_squares([], []))

    def test_display_rounding_leaves_a_missing_fit_alone(self):
        """None means "no line to draw" and must survive to the JSON as null."""
        self.assertIsNone(self.gen.round_fit(None))

    def test_display_rounding_keeps_enough_precision_for_the_drawn_line(self):
        fit = self.gen.round_fit(
            {"slope": 2.0694321, "intercept": 22.0604321, "r2": 0.99991234}
        )

        self.assertEqual(fit["slope"], 2.069)
        self.assertEqual(fit["intercept"], 22.06)
        self.assertEqual(fit["r2"], 0.9999)


class TestBenchmarkPreservation(unittest.TestCase):
    """`--skip-benchmark` exists so figures can be regenerated without a few minutes
    of timing. It must not take the timings with it: writing an empty benchmark
    blanks the chart the reader came for.
    """

    @classmethod
    def setUpClass(cls):
        cls.gen = load_generator()

    def test_fresh_results_win_when_this_run_produced_them(self):
        fresh = [{"model": "yolov8n.pt", "median_ms": 1.0}]

        self.assertEqual(self.gen.resolve_benchmark(fresh, [{"model": "stale"}]), fresh)

    def test_previous_results_survive_a_skipped_run(self):
        previous = [{"model": "yolov8n.pt", "median_ms": 34.23}]

        self.assertEqual(self.gen.resolve_benchmark([], previous), previous)

    def test_both_empty_is_an_empty_benchmark_not_an_error(self):
        self.assertEqual(self.gen.resolve_benchmark([], []), [])


class TestCascadeReference(unittest.TestCase):
    """The chart marks where the shipped defaults sit, so that marker must be real.

    Note this class makes no accuracy claim. A Haar cascade returns candidate
    regions, and on the reference photo the shipped defaults return one region
    on a real face and one in the crowd behind — equal to the number of people
    in the frame purely by coincidence. What is checked here is that the chart
    describes the code's behaviour, nothing more.
    """

    # Pinned so that moving the cascade parameters shows up as a failing test
    # rather than as a chart that silently redraws around a stale marker.
    EXPECTED_DEFAULT_SCALE_FACTOR = 1.1
    EXPECTED_DEFAULT_MIN_NEIGHBORS = 5
    EXPECTED_REGIONS_ON_REFERENCE = 2

    @classmethod
    def setUpClass(cls):
        cls.gen = load_generator()

    def test_defaults_are_read_from_the_shipped_detector(self):
        defaults = self.gen.cascade_sweep_defaults(None)

        self.assertEqual(defaults["scale_factor"], self.EXPECTED_DEFAULT_SCALE_FACTOR)
        self.assertEqual(defaults["min_neighbors"], self.EXPECTED_DEFAULT_MIN_NEIGHBORS)

    def test_the_defaults_fall_inside_the_swept_grid(self):
        """Otherwise the chart has no cell to mark as the shipped setting."""
        defaults = self.gen.cascade_sweep_defaults(None)

        self.assertIn(defaults["scale_factor"], self.gen.SWEEP_SCALE_FACTORS)
        self.assertIn(defaults["min_neighbors"], self.gen.SWEEP_MIN_NEIGHBORS)

    def test_reference_image_describes_itself_without_claiming_accuracy(self):
        for ref in self.gen.REFERENCE_IMAGES:
            with self.subTest(image=ref["name"]):
                self.assertIn("label", ref)
                self.assertNotIn(
                    "faces",
                    ref,
                    "a declared face count on the input invites a caption the "
                    "measurement cannot support",
                )

    @unittest.skipUnless(
        os.path.exists(os.path.join(REPO_ROOT, "yolov8n.pt")),
        "yolov8n.pt not present; run `make setup` to fetch models",
    )
    def test_default_parameters_return_a_stable_region_count(self):
        """Behaviour guard, not an accuracy claim — see the class docstring."""
        import cv2

        from src.face_detector import FaceDetector

        ref = self.gen.REFERENCE_IMAGES[0]
        path = self.gen.resolve_reference_path(ref)
        self.assertIsNotNone(path, f"reference image {ref['name']} not found")

        found = FaceDetector().detect_faces(cv2.imread(path))

        self.assertEqual(len(found), self.EXPECTED_REGIONS_ON_REFERENCE)


if __name__ == "__main__":
    unittest.main()
