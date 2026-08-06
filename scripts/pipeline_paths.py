"""Shared run paths and manifests for the batch analysis pipeline.

The Wind-backed batch is the only source of bond codes.  Each requested code
set gets an isolated run directory, and offline stages discover the latest
usable batch through ``output/current_run.json``.  This prevents IV, gap, and
plotting jobs from silently mixing artifacts from different batches.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Iterable


DEFAULT_OUTPUT_ROOT = Path("output")
CURRENT_RUN_FILENAME = "current_run.json"
RUN_MANIFEST_FILENAME = "run_manifest.json"
MANIFEST_SCHEMA_VERSION = 1
_BOND_CODE_PATTERN = re.compile(r"^[A-Z0-9]+\.[A-Z0-9]+$")
_RUN_ID_PATTERN = re.compile(r"^[A-Z0-9_]+$")
_MAX_READABLE_RUN_ID_LENGTH = 64


class PipelineManifestError(ValueError):
    """Raised when a run pointer or manifest cannot be trusted."""


def normalize_bond_codes(codes: Iterable[str]) -> tuple[str, ...]:
    """Return a validated, unique, order-independent bond-code tuple."""

    normalized = tuple(sorted(str(code).strip().upper() for code in codes))
    if not normalized:
        raise ValueError("bond codes cannot be empty")
    invalid = [
        code
        for code in normalized
        if not code or not _BOND_CODE_PATTERN.fullmatch(code)
    ]
    if invalid:
        raise ValueError(
            "bond codes must look like '123117.SZ' and contain no path "
            f"characters: {invalid}"
        )
    duplicated = sorted(
        {code for code in normalized if normalized.count(code) > 1}
    )
    if duplicated:
        raise ValueError(f"bond codes contain duplicates: {duplicated}")
    return normalized


def run_id_for_bond_codes(codes: Iterable[str]) -> str:
    """Build a stable Windows-safe identifier for one requested code set."""

    normalized = normalize_bond_codes(codes)
    readable = "__".join(code.replace(".", "_") for code in normalized)
    if len(readable) <= _MAX_READABLE_RUN_ID_LENGTH:
        return readable
    digest = sha256("|".join(normalized).encode("utf-8")).hexdigest()[:12]
    return f"{len(normalized)}_BONDS_{digest.upper()}"


@dataclass(frozen=True)
class PipelinePaths:
    """All file and directory locations belonging to one batch run."""

    output_root: Path
    run_id: str

    @classmethod
    def from_bond_codes(
        cls,
        codes: Iterable[str],
        output_root: Path = DEFAULT_OUTPUT_ROOT,
    ) -> "PipelinePaths":
        """Create paths derived only from the requested bond-code set."""

        return cls(Path(output_root), run_id_for_bond_codes(codes))

    @classmethod
    def from_run_id(
        cls,
        run_id: str,
        output_root: Path = DEFAULT_OUTPUT_ROOT,
    ) -> "PipelinePaths":
        """Create paths for a validated run identifier."""

        normalized = str(run_id).strip().upper()
        if not normalized or not _RUN_ID_PATTERN.fullmatch(normalized):
            raise PipelineManifestError(f"invalid run_id: {run_id!r}")
        return cls(Path(output_root), normalized)

    @property
    def runs_dir(self) -> Path:
        return self.output_root / "runs"

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / self.run_id

    @property
    def batch_dir(self) -> Path:
        return self.run_dir / "batch"

    @property
    def iv_dir(self) -> Path:
        return self.run_dir / "implied_volatility"

    @property
    def gap_dir(self) -> Path:
        return self.run_dir / "gap_correction"

    @property
    def figures_dir(self) -> Path:
        return self.run_dir / "figures"

    @property
    def run_manifest_path(self) -> Path:
        return self.run_dir / RUN_MANIFEST_FILENAME

    @property
    def current_run_path(self) -> Path:
        return self.output_root / CURRENT_RUN_FILENAME

    @property
    def batch_daily_csv(self) -> Path:
        return self.batch_dir / "daily_tracking_all.csv"

    @property
    def batch_summary_csv(self) -> Path:
        return self.batch_dir / "comparison_summary.csv"

    @property
    def iv_daily_csv(self) -> Path:
        return self.iv_dir / "implied_volatility_backtest_daily.csv"

    @property
    def iv_summary_csv(self) -> Path:
        return self.iv_dir / "implied_volatility_backtest_summary.csv"

    @property
    def iv_selected_csv(self) -> Path:
        return self.iv_dir / "implied_volatility_selected_models.csv"

    @property
    def iv_calibration_csv(self) -> Path:
        return self.iv_dir / "implied_volatility_calibration_diagnostics.csv"

    @property
    def iv_config_csv(self) -> Path:
        return self.iv_dir / "implied_volatility_backtest_config.csv"

    @property
    def gap_head_to_head_csv(self) -> Path:
        return self.gap_dir / "iv_vs_gap_head_to_head.csv"

    @property
    def gap_headline_csv(self) -> Path:
        return self.gap_dir / "iv_vs_gap_headline.csv"


@dataclass(frozen=True)
class RunManifest:
    """Validated bond membership and paths for one saved batch run."""

    paths: PipelinePaths
    requested_bond_codes: tuple[str, ...]
    successful_bond_codes: tuple[str, ...]
    failed_bond_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a portable representation with output-root-relative paths."""

        root = self.paths.output_root

        def relative(path: Path) -> str:
            return path.relative_to(root).as_posix()

        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": self.paths.run_id,
            "requested_bond_codes": list(self.requested_bond_codes),
            "successful_bond_codes": list(self.successful_bond_codes),
            "failed_bond_codes": list(self.failed_bond_codes),
            "paths": {
                "run_dir": relative(self.paths.run_dir),
                "batch_dir": relative(self.paths.batch_dir),
                "iv_dir": relative(self.paths.iv_dir),
                "gap_dir": relative(self.paths.gap_dir),
                "figures_dir": relative(self.paths.figures_dir),
            },
        }


def build_run_manifest(
    paths: PipelinePaths,
    requested_bond_codes: Iterable[str],
    successful_bond_codes: Iterable[str],
    failed_bond_codes: Iterable[str],
) -> RunManifest:
    """Validate and build the manifest saved after a batch attempt."""

    requested = normalize_bond_codes(requested_bond_codes)
    successful_values = tuple(successful_bond_codes)
    failed_values = tuple(failed_bond_codes)
    successful = (
        normalize_bond_codes(successful_values)
        if successful_values
        else ()
    )
    failed = (
        normalize_bond_codes(failed_values)
        if failed_values
        else ()
    )
    expected_run_id = run_id_for_bond_codes(requested)
    if paths.run_id != expected_run_id:
        raise ValueError(
            f"run_id {paths.run_id!r} does not match requested codes "
            f"({expected_run_id!r})"
        )
    successful_set = set(successful)
    failed_set = set(failed)
    requested_set = set(requested)
    if successful_set & failed_set:
        raise ValueError("successful and failed bond codes must be disjoint")
    if successful_set | failed_set != requested_set:
        raise ValueError(
            "successful and failed bond codes must partition requested codes"
        )
    return RunManifest(paths, requested, successful, failed)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """Replace one JSON file only after its complete contents are on disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_run_manifest(manifest: RunManifest) -> None:
    """Save a run manifest and update current only for a usable batch."""

    _write_json_atomic(manifest.paths.run_manifest_path, manifest.to_dict())
    if manifest.successful_bond_codes:
        pointer = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": manifest.paths.run_id,
            "manifest_path": manifest.paths.run_manifest_path.relative_to(
                manifest.paths.output_root
            ).as_posix(),
        }
        _write_json_atomic(manifest.paths.current_run_path, pointer)


def _read_json(path: Path, description: str) -> dict[str, object]:
    """Read one JSON object and provide a stage-aware error."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PipelineManifestError(
            f"{description} not found at {path}; run batch_compare_crr.py first"
        ) from exc
    except json.JSONDecodeError as exc:
        raise PipelineManifestError(
            f"{description} is not valid JSON at {path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise PipelineManifestError(
            f"{description} must contain a JSON object: {path}"
        )
    return raw


def _resolve_inside_output_root(output_root: Path, relative: object) -> Path:
    """Resolve a manifest path without allowing it to escape output_root."""

    if not isinstance(relative, str) or not relative.strip():
        raise PipelineManifestError("manifest_path must be a non-empty string")
    root = output_root.resolve()
    candidate = (output_root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise PipelineManifestError(
            f"manifest path escapes output root {output_root}: {relative!r}"
        )
    return candidate


def load_run_manifest(
    manifest_path: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> RunManifest:
    """Load and fully validate a saved run manifest."""

    raw = _read_json(Path(manifest_path), "run manifest")
    if raw.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise PipelineManifestError(
            f"unsupported run manifest schema: {raw.get('schema_version')!r}"
        )
    run_id = raw.get("run_id")
    paths = PipelinePaths.from_run_id(str(run_id), Path(output_root))
    try:
        requested = normalize_bond_codes(raw["requested_bond_codes"])
        successful_raw = tuple(raw["successful_bond_codes"])
        failed_raw = tuple(raw["failed_bond_codes"])
        successful = (
            normalize_bond_codes(successful_raw) if successful_raw else ()
        )
        failed = normalize_bond_codes(failed_raw) if failed_raw else ()
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineManifestError(f"invalid bond membership in manifest: {exc}") from exc
    try:
        manifest = build_run_manifest(
            paths,
            requested,
            successful,
            failed,
        )
    except ValueError as exc:
        raise PipelineManifestError(str(exc)) from exc
    expected_paths = manifest.to_dict()["paths"]
    if raw.get("paths") != expected_paths:
        raise PipelineManifestError(
            "manifest paths do not match the run_id and output root"
        )
    return manifest


def load_current_run(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> RunManifest:
    """Load the latest usable batch run selected by current_run.json."""

    root = Path(output_root)
    pointer_path = root / CURRENT_RUN_FILENAME
    pointer = _read_json(pointer_path, "current run pointer")
    if pointer.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise PipelineManifestError(
            f"unsupported current run schema: {pointer.get('schema_version')!r}"
        )
    manifest_path = _resolve_inside_output_root(
        root,
        pointer.get("manifest_path"),
    )
    manifest = load_run_manifest(manifest_path, root)
    if pointer.get("run_id") != manifest.paths.run_id:
        raise PipelineManifestError(
            "current run pointer and run manifest contain different run_id values"
        )
    if not manifest.successful_bond_codes:
        raise PipelineManifestError(
            f"current run {manifest.paths.run_id} has no successful bonds"
        )
    return manifest


def validate_exact_bond_codes(
    actual_codes: Iterable[str],
    expected_codes: Iterable[str],
    source: str,
) -> None:
    """Require one saved table to contain exactly the successful batch bonds."""

    try:
        actual = normalize_bond_codes(actual_codes)
    except ValueError as exc:
        raise ValueError(f"{source} has invalid bond codes: {exc}") from exc
    expected = normalize_bond_codes(expected_codes)
    if actual != expected:
        raise ValueError(
            f"{source} bond codes do not match the current run: "
            f"actual={list(actual)}, expected={list(expected)}"
        )


def require_artifacts(paths: Iterable[Path], stage: str, run_id: str) -> None:
    """Fail before computation when an upstream stage is incomplete."""

    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{stage} artifacts are incomplete for run {run_id}: {missing}"
        )
