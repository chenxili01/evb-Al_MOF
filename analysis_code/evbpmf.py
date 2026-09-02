#!/usr/bin/env python3
"""Run EVB PMF reconstruction from a JSON benchmark/path config.

Supports loading DFT reference targets (barrier, free_energy, reverse_isfs)
from an external Python-style file via --target-file, or inline via the
``dft_benchmark`` section of the JSON config.

Typical usage::

    python evbpmf.py evbpmf_config.json
    python evbpmf.py evbpmf_config.json --target-file ref_target_def2svp.txt
    python evbpmf.py evbpmf_config.json --preflight-only
    python evbpmf.py evbpmf_config.json --system sys01_kneutral_tip3p --site site01_res1109
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HELPER_CANDIDATES = [
    SCRIPT_DIR / "cluster_evb_tools",
    SCRIPT_DIR / "../cluster_evb_tools",
    SCRIPT_DIR / "../harmonic_collection/code/cluster_evb_tools",
    SCRIPT_DIR / "code/cluster_evb_tools",
    Path.cwd() / "code/cluster_evb_tools",
]


def _resolve_path(path: str | os.PathLike[str], base: Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value.resolve()
    return (base / value).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Config JSON must contain an object at the top level.")
    return payload


def _setup_helpers(config: dict[str, Any], config_path: Path) -> Path:
    candidates = []
    if config.get("helper_path"):
        candidates.append(_resolve_path(config["helper_path"], config_path.parent))
    candidates.extend(path.resolve() for path in DEFAULT_HELPER_CANDIDATES)
    for candidate in candidates:
        if (candidate / "mof_evb_demo_methods.py").is_file() and (
            candidate / "evb_tools" / "evb_utils.py"
        ).is_file():
            sys.path.insert(0, str(candidate))
            return candidate
    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not find cluster_evb_tools helper code. Set `helper_path` in the "
        "JSON config. Searched:\n" + searched
    )


def _benchmark_lookup(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = config.get("dft_benchmark")
    if raw is None:
        raw = config.get("dft_benchmarks")
    if raw is None:
        raise ValueError("Config must define `dft_benchmark`.")

    if isinstance(raw, dict):
        return {str(name): dict(value) for name, value in raw.items()}
    if isinstance(raw, list):
        out = {}
        for item in raw:
            if not isinstance(item, dict) or "reaction" not in item:
                raise ValueError("Each dft_benchmark list entry must contain `reaction`.")
            out[str(item["reaction"])] = dict(item)
        return out
    raise ValueError("`dft_benchmark` must be either an object or a list.")


def _load_reference_targets(
    path: Path | None,
) -> list[tuple[float, float, bool]] | None:
    """Load REFERENCE_TARGETS from a Python-style target file.

    The file must define a module-level ``REFERENCE_TARGETS`` list where each
    entry is a 3-tuple of ``(barrier_kcal, free_energy_kcal, reverse_isfs)``.

    Returns ``None`` when *path* is ``None``, so callers can fall back to the
    JSON config's ``dft_benchmark`` section.
    """
    if path is None:
        return None

    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "REFERENCE_TARGETS"
                for target in node.targets
            )
        ):
            targets = ast.literal_eval(node.value)
            break
    else:
        raise ValueError(f"{path} does not define REFERENCE_TARGETS")

    if not isinstance(targets, list) or not targets:
        raise ValueError(f"{path}: REFERENCE_TARGETS must be a non-empty list.")

    normalized: list[tuple[float, float, bool]] = []
    for index, target in enumerate(targets, start=1):
        if not isinstance(target, (list, tuple)) or len(target) != 3:
            raise ValueError(
                f"{path}: entry {index} must be "
                "(barrier_kcal, free_energy_kcal, reverse_isfs)"
            )
        barrier, free_energy, reverse_isfs = target
        if not isinstance(reverse_isfs, bool):
            raise ValueError(
                f"{path}: entry {index} reverse_isfs must be boolean, "
                f"got {type(reverse_isfs).__name__}"
            )
        normalized.append((float(barrier), float(free_energy), reverse_isfs))
    return normalized


def _merge_benchmarks(
    config_benchmarks: dict[str, dict[str, Any]],
    reaction_names: list[str],
    external_targets: list[tuple[float, float, bool]] | None,
) -> dict[str, dict[str, Any]]:
    """Merge JSON-configured benchmarks with external target-file values.

    External targets (when provided) are matched to reactions **by position**
    (the order they appear in the ``reactions`` list).  Entries in the JSON
    ``dft_benchmark`` object act as overrides — if a reaction name already
    has a benchmark from the JSON config the external value is ignored for
    that reaction.
    """
    merged = dict(config_benchmarks)

    if external_targets is None:
        return merged

    for index, name in enumerate(reaction_names):
        if index >= len(external_targets):
            raise ValueError(
                f"Reaction {index + 1} ('{name}') has no matching entry in the "
                f"target file ({len(external_targets)} entries provided)."
            )
        # JSON config values take precedence over target-file values
        if name not in merged:
            barrier, free_energy, reverse_isfs = external_targets[index]
            merged[name] = {
                "barrier_kcal": barrier,
                "free_energy_kcal": free_energy,
                "reverse_isfs": reverse_isfs,
            }

    return merged


def _reaction_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    reactions = config.get("reactions")
    if not isinstance(reactions, list) or not reactions:
        raise ValueError("Config must define a non-empty `reactions` list.")
    for entry in reactions:
        if not isinstance(entry, dict):
            raise ValueError("Each reaction entry must be an object.")
        for key in ("name", "reaction_path", "reference_path"):
            if key not in entry:
                raise ValueError(f"Reaction entry is missing `{key}`.")
    return reactions


def _common_root(paths: list[Path]) -> Path:
    return Path(os.path.commonpath([str(path) for path in paths])).resolve()


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError as exc:
        raise ValueError(f"{path} is not under data_root {root}") from exc


def _base_presets(central: bool = False) -> list[dict[str, Any]]:
    baseline = dict(
        state_min_prominence=1.0,
        state_min_basin_width=2.0,
        state_edge_exclusion_bins=2,
    )
    relaxed = dict(
        state_min_prominence=0.0,
        state_min_basin_width=0.0,
        state_edge_exclusion_bins=2,
    )
    presets = [
        dict(name="baseline", step=1, bins=50, minpts=30, **baseline),
        dict(name="relaxed_topology", step=1, bins=50, minpts=15, **relaxed),
        dict(name="low_minpts", step=1, bins=60, minpts=5, **relaxed),
    ]
    if central:
        for preset in presets:
            preset.update(
                {
                    "central_window_kj": [1500.0, 3000.0, 5000.0, 8000.0],
                    "central_window_max_y_range_kcal": 20.0,
                    "require_central_window": True,
                }
            )
    return presets


def _preflight(data_root: Path, configs: list, configs_meta: dict[str, Any]) -> dict[str, Any]:
    """Validate that all reaction data paths exist and contain expected XVG files."""
    from mof_evb_demo_methods import discover_active_systems

    system_filter = configs_meta.get("system_filter", [])
    site_filter = configs_meta.get("site_filter", [])

    report: dict[str, Any] = {
        "data_root": str(data_root),
        "reactions": [],
    }

    total_sites = 0
    total_refs = 0
    total_xvgs = 0

    for cfg in configs:
        reaction_info: dict[str, Any] = {
            "name": cfg.name,
            "collection_dir": cfg.collection_dir,
            "reference_pair": cfg.reference_pair,
        }
        try:
            systems = discover_active_systems(data_root, cfg.collection_dir)
        except FileNotFoundError as exc:
            reaction_info["status"] = "missing"
            reaction_info["error"] = str(exc)
            report["reactions"].append(reaction_info)
            continue

        # Apply system/site filters
        if system_filter:
            systems = {
                sys_name: sites
                for sys_name, sites in systems.items()
                if sys_name in system_filter
            }
        if site_filter and systems:
            first_sys = next(iter(systems))
            systems = {
                sys_name: [s for s in sites if s in site_filter]
                for sys_name, sites in systems.items()
            }

        site_count = sum(len(sites) for sites in systems.values())
        total_sites += site_count
        reaction_info["systems"] = {
            sys_name: {"sites": sites, "site_count": len(sites)}
            for sys_name, sites in systems.items()
        }

        # Check reference pair
        ref_path = data_root / cfg.reference_pair[0]
        if ref_path.is_dir():
            ref_count = sum(1 for _ in ref_path.glob("*.xvg"))
            reaction_info["reference"] = {
                "path": str(ref_path),
                "xvg_count": ref_count,
            }
            total_refs += 1
        else:
            reaction_info["reference"] = {
                "path": str(ref_path),
                "status": "missing",
            }

        # Count XVG files
        xvg_total = 0
        for sys_name, sites in systems.items():
            for site in sites:
                for side in ("potentialA", "potentialB"):
                    pair_dir = data_root / cfg.collection_dir / sys_name / site / side
                    if pair_dir.is_dir():
                        xvg_total += sum(1 for _ in pair_dir.glob("*.xvg"))
        total_xvgs += xvg_total
        reaction_info["xvg_files"] = xvg_total
        reaction_info["site_pairs"] = site_count * 2  # A + B
        reaction_info["status"] = "ok"
        report["reactions"].append(reaction_info)

    report["totals"] = {
        "site_pairs": total_sites * 2,
        "reference_pairs": total_refs * 2,
        "xvg_files": total_xvgs,
    }
    return report


def _build_configs(
    config: dict[str, Any],
    config_path: Path,
    external_targets: list[tuple[float, float, bool]] | None = None,
    system_filter: list[str] | None = None,
    site_filter: list[str] | None = None,
):
    from mof_evb_demo_methods import ReactionConfig

    config_benchmarks = _benchmark_lookup(config)
    entries = _reaction_entries(config)
    benchmarks = _merge_benchmarks(
        config_benchmarks,
        [str(entry["name"]) for entry in entries],
        external_targets,
    )
    data_root = (
        _resolve_path(config["data_root"], config_path.parent)
        if config.get("data_root")
        else _common_root(
            [
                _resolve_path(entry["reaction_path"], config_path.parent)
                for entry in entries
            ]
            + [
                _resolve_path(entry["reference_path"], config_path.parent)
                for entry in entries
            ]
        )
    )

    configs: list[ReactionConfig] = []
    for entry in entries:
        name = str(entry["name"])
        benchmark = benchmarks.get(name) or benchmarks.get(str(entry.get("benchmark")))
        if benchmark is None:
            raise ValueError(f"No DFT benchmark found for reaction `{name}`.")
        reaction_path = _resolve_path(entry["reaction_path"], config_path.parent)
        reference_path = _resolve_path(entry["reference_path"], config_path.parent)
        central = bool(entry.get("central_window", False))
        presets = _base_presets(central=central)
        if entry.get("resolution_presets"):
            presets = entry["resolution_presets"]
        if entry.get("require_rs_basin"):
            for preset in presets:
                preset["require_rs_basin"] = True

        configs.append(
            ReactionConfig(
                name=name,
                label=str(entry.get("label", name)),
                collection_dir=_relative_to_root(reaction_path, data_root),
                reference_pair=[
                    str(Path(_relative_to_root(reference_path, data_root)) / "potentialA"),
                    str(Path(_relative_to_root(reference_path, data_root)) / "potentialB"),
                ],
                barrier_kcal=float(benchmark["barrier_kcal"]),
                free_energy_kcal=float(
                    benchmark.get("free_energy_kcal", benchmark.get("dg_kcal"))
                ),
                reverse_isfs=bool(
                    benchmark.get("reverse_isfs", entry.get("reverse_isfs", False))
                ),
                resolution_presets=presets,
                parameter_mode=str(entry.get("parameter_mode", "fit_reference")),
                fit_max_iterations=int(entry.get("fit_max_iterations", 2000)),
                manual_alpha_kj=entry.get("manual_alpha_kj"),
                manual_h12_kj=entry.get("manual_h12_kj"),
                use_stable_boltzmann=bool(entry.get("use_stable_boltzmann", True)),
                plot_state_padding_fraction=float(
                    entry.get("plot_state_padding_fraction", 0.10)
                ),
                barrierless_barrier_tolerance_kcal=float(
                    entry.get("barrierless_barrier_tolerance_kcal", 0.10)
                ),
                barrierless_free_energy_tolerance_kcal=float(
                    entry.get("barrierless_free_energy_tolerance_kcal", 0.05)
                ),
            )
        )

    # Attach filters for use during preflight / run_all_reactions
    configs_meta = {
        "system_filter": system_filter or [],
        "site_filter": site_filter or [],
    }
    return data_root, configs, configs_meta


def _write_manifest(path: Path, config_path: Path, helper_path: Path, data_root: Path, configs) -> None:
    path.write_text(
        json.dumps(
            {
                "config": config_path.name,
                "helper_path": helper_path.name,
                "data_root": data_root.name,
                "reactions": [
                    {
                        "name": cfg.name,
                        "collection_dir": cfg.collection_dir,
                        "reference_pair": cfg.reference_pair,
                        "barrier_kcal": cfg.barrier_kcal,
                        "free_energy_kcal": cfg.free_energy_kcal,
                        "reverse_isfs": cfg.reverse_isfs,
                        "parameter_mode": cfg.parameter_mode,
                        "fit_max_iterations": cfg.fit_max_iterations,
                        "resolution_presets": cfg.resolution_presets,
                    }
                    for cfg in configs
                ],
            },
            indent=2,
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run EVB PMF reconstruction from a JSON benchmark/path config."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="JSON config with DFT benchmarks and paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output directory from JSON config.",
    )
    parser.add_argument(
        "--target-file",
        type=Path,
        default=None,
        help=(
            "Python-style file containing a REFERENCE_TARGETS list of "
            "(barrier_kcal, free_energy_kcal, reverse_isfs) tuples. "
            "Values are matched to reactions by position in the reactions list. "
            "JSON dft_benchmark entries take precedence over target-file values."
        ),
    )
    parser.add_argument(
        "--reaction",
        action="append",
        default=[],
        dest="reactions_filter",
        help="Limit processing to one or more reaction names (can repeat).",
    )
    parser.add_argument(
        "--system",
        action="append",
        default=[],
        dest="systems",
        help="Limit processing to one or more system IDs (can repeat).",
    )
    parser.add_argument(
        "--site",
        action="append",
        default=[],
        dest="sites",
        help="Limit processing to one or more site IDs (can repeat).",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate data paths and XVG file counts without running EVB.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = _load_json(config_path)

    # Load external DFT reference targets if provided
    external_targets = _load_reference_targets(
        args.target_file.resolve() if args.target_file else None
    )

    helper_path = _setup_helpers(config, config_path)

    from mof_evb_demo_methods import run_all_reactions

    data_root, configs, configs_meta = _build_configs(
        config,
        config_path,
        external_targets=external_targets,
        system_filter=args.systems,
        site_filter=args.sites,
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else _resolve_path(
            config.get("output_dir", "evbpmf_result"), config_path.parent
        )
    )

    # --- Apply reaction filter -----------------------------------------------
    if args.reactions_filter:
        configs = [c for c in configs if c.name in args.reactions_filter]
        if not configs:
            print(
                f"No reactions matched filter: {args.reactions_filter}",
                file=sys.stderr,
            )
            return 1

    # --- Preflight (only when --preflight-only is set) -----------------------
    if args.preflight_only:
        preflight_report = _preflight(data_root, configs, configs_meta)
        print("=== Preflight ===")
        print(json.dumps(preflight_report, indent=2, sort_keys=True))
        if external_targets is not None:
            print(
                "\nReference targets (from file):",
                json.dumps(
                    [
                        {"barrier_kcal": b, "free_energy_kcal": dg, "reverse_isfs": r}
                        for b, dg, r in external_targets
                    ],
                    indent=2,
                ),
            )
        else:
            print("\nReference targets: sourced from JSON dft_benchmark section.")
        return 0

    # --- Run -----------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(
        output_dir / "evbpmf_manifest.json",
        config_path,
        helper_path,
        data_root,
        configs,
    )

    if args.target_file is not None:
        (output_dir / "reference_targets_source.txt").write_text(
            args.target_file.resolve().read_text()
        )

    raw_profile_dir = output_dir / "raw_evb_profiles"
    previous_profile_artifact_dir = os.environ.get("EVB_PROFILE_ARTIFACT_DIR")
    os.environ["EVB_PROFILE_ARTIFACT_DIR"] = str(raw_profile_dir)
    try:
        run_all_reactions(data_root, configs, output_dir)
    finally:
        if previous_profile_artifact_dir is None:
            os.environ.pop("EVB_PROFILE_ARTIFACT_DIR", None)
        else:
            os.environ["EVB_PROFILE_ARTIFACT_DIR"] = previous_profile_artifact_dir
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
