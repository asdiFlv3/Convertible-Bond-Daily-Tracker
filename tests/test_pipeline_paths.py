"""Tests for code-derived pipeline paths and current-run manifests."""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_paths import (
    CURRENT_RUN_FILENAME,
    PipelineManifestError,
    PipelinePaths,
    build_run_manifest,
    load_current_run,
    normalize_bond_codes,
    run_id_for_bond_codes,
    save_run_manifest,
    validate_exact_bond_codes,
)


class PipelinePathTests(unittest.TestCase):
    """Verify stable, isolated, Windows-safe paths for each code set."""

    def test_same_code_set_has_same_run_id_regardless_of_order(self) -> None:
        first = run_id_for_bond_codes(["123117.SZ", "118058.SH"])
        second = run_id_for_bond_codes([" 118058.sh ", "123117.sz"])

        self.assertEqual(first, second)
        self.assertRegex(first, re.compile(r"^[A-Z0-9_]+$"))
        self.assertIn("123117_SZ", first)
        self.assertIn("118058_SH", first)

    def test_replacing_one_code_changes_every_stage_directory(self) -> None:
        old = PipelinePaths.from_bond_codes(
            ["123117.SZ", "118058.SH"]
        )
        new = PipelinePaths.from_bond_codes(
            ["123117.SZ", "123255.SZ"]
        )

        self.assertNotEqual(old.run_id, new.run_id)
        self.assertNotEqual(old.batch_dir, new.batch_dir)
        self.assertNotEqual(old.iv_dir, new.iv_dir)
        self.assertNotEqual(old.gap_dir, new.gap_dir)
        self.assertNotEqual(old.figures_dir, new.figures_dir)

    def test_long_code_list_uses_bounded_stable_identifier(self) -> None:
        codes = [f"{index:06d}.SZ" for index in range(30)]
        first = run_id_for_bond_codes(codes)
        second = run_id_for_bond_codes(reversed(codes))

        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 64)
        self.assertRegex(first, re.compile(r"^30_BONDS_[A-F0-9]{12}$"))

    def test_readable_run_id_switches_to_hash_after_64_characters(self) -> None:
        six_codes = [f"{index:06d}.SZ" for index in range(6)]
        seven_codes = [f"{index:06d}.SZ" for index in range(7)]

        self.assertEqual(len(run_id_for_bond_codes(six_codes)), 64)
        self.assertRegex(
            run_id_for_bond_codes(seven_codes),
            re.compile(r"^7_BONDS_[A-F0-9]{12}$"),
        )

    def test_empty_duplicate_and_path_like_codes_are_rejected(self) -> None:
        invalid_sets = (
            [],
            ["123117.SZ", " 123117.sz "],
            ["../123117.SZ"],
            [r"folder\123117.SZ"],
            ["123117"],
        )
        for codes in invalid_sets:
            with self.subTest(codes=codes):
                with self.assertRaises(ValueError):
                    normalize_bond_codes(codes)

    def test_exact_code_validation_reports_cross_run_mismatch(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"actual=.*118058\.SH.*expected=.*123255\.SZ",
        ):
            validate_exact_bond_codes(
                ["123117.SZ", "118058.SH"],
                ["123117.SZ", "123255.SZ"],
                "daily.csv",
            )


class RunManifestTests(unittest.TestCase):
    """Verify publication, round-trip validation, and failure behavior."""

    def test_manifest_round_trip_preserves_membership_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "output"
            paths = PipelinePaths.from_bond_codes(
                ["123117.SZ", "118058.SH"],
                root,
            )
            manifest = build_run_manifest(
                paths,
                ["123117.SZ", "118058.SH"],
                ["123117.SZ"],
                ["118058.SH"],
            )
            save_run_manifest(manifest)

            loaded = load_current_run(root)

            self.assertEqual(loaded.paths, paths)
            self.assertEqual(
                loaded.requested_bond_codes,
                ("118058.SH", "123117.SZ"),
            )
            self.assertEqual(loaded.successful_bond_codes, ("123117.SZ",))
            self.assertEqual(loaded.failed_bond_codes, ("118058.SH",))

    def test_all_failed_run_does_not_replace_previous_current_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "output"
            usable_paths = PipelinePaths.from_bond_codes(["123117.SZ"], root)
            usable = build_run_manifest(
                usable_paths,
                ["123117.SZ"],
                ["123117.SZ"],
                [],
            )
            save_run_manifest(usable)
            failed_paths = PipelinePaths.from_bond_codes(["118058.SH"], root)
            failed = build_run_manifest(
                failed_paths,
                ["118058.SH"],
                [],
                ["118058.SH"],
            )
            save_run_manifest(failed)

            current = load_current_run(root)

            self.assertEqual(current.paths.run_id, usable_paths.run_id)
            self.assertTrue(failed_paths.run_manifest_path.is_file())

    def test_missing_or_malformed_current_pointer_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "output"
            with self.assertRaisesRegex(
                PipelineManifestError,
                "run batch_compare_crr.py first",
            ):
                load_current_run(root)

            root.mkdir(parents=True)
            (root / CURRENT_RUN_FILENAME).write_text("not-json", encoding="utf-8")
            with self.assertRaisesRegex(
                PipelineManifestError,
                "not valid JSON",
            ):
                load_current_run(root)

    def test_current_pointer_cannot_escape_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "output"
            root.mkdir(parents=True)
            pointer = {
                "schema_version": 1,
                "run_id": "123117_SZ",
                "manifest_path": "../outside.json",
            }
            (root / CURRENT_RUN_FILENAME).write_text(
                json.dumps(pointer),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                PipelineManifestError,
                "escapes output root",
            ):
                load_current_run(root)


if __name__ == "__main__":
    unittest.main()
