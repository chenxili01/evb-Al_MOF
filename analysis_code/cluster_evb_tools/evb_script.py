#!/usr/bin/env python3
"""Recalculate the harmonic_0606 EVB profiles using updated reference energies."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

TOOL_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path.cwd()
sys.path.insert(0, str(TOOL_ROOT))

from mof_evb_demo_methods import default_reaction_configs, run_all_reactions  # noqa: E402


EXPECTED_SYSTEMS = [
    "sys01_kneutral_tip3p",
    "sys02_khco3_co2_tip3p",
    "sys03_khco3_co2_tip3p_graphene",
    "sys04_khco3_co2_tip3p_graphene_weakEy",
]
EXPECTED_SITES = [
    "site01_res1109",
    "site02_res1198",
    "site03_res1122",
    "site04_res1085",
    "site05_res1171",
    "site06_res1145",
    "site07_res1179",
    "site08_res1154",
]
EXPECTED_WINDOWS = set(range(21))
REFERENCE_TARGETS = [
    (11.44, 0.17, True),
    (5.19, -34.00, False),
    (0.00, -7.62, False),
    (3.58, -13.40, False),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recalculate harmonic_0606 EVB profiles with new reference energies."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Directory containing reaction1..reaction4 and ref_reaction1..ref_reaction4. "
            "Default: current working directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT / "evb_result",
    )
    parser.add_argument(
        "--target-file",
        type=Path,
        help="Python-style file containing a four-entry REFERENCE_TARGETS list.",
    )
    parser.add_argument("--system", action="append", default=[], help="Limit preflight to one or more system ids.")
    parser.add_argument("--site", action="append", default=[], help="Limit preflight to one or more site ids.")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def window_indices(directory: Path, side: str) -> set[int]:
    suffix = f"potential_recal{side}.xvg"
    indices = set()
    for path in directory.glob(f"*{suffix}"):
        prefix = path.name.removesuffix(suffix)
        if prefix.isdigit():
            indices.add(int(prefix))
    return indices


def validate_pair(pair_root: Path) -> list[str]:
    problems = []
    for side in ("A", "B"):
        directory = pair_root / f"potential{side}"
        indices = window_indices(directory, side) if directory.is_dir() else set()
        if indices != EXPECTED_WINDOWS:
            problems.append(
                f"{directory}: expected windows 0-20; "
                f"missing={sorted(EXPECTED_WINDOWS - indices)}, "
                f"extra={sorted(indices - EXPECTED_WINDOWS)}"
            )
    return problems


def count_pair_xvgs(pair_root: Path) -> int:
    return sum(
        1
        for side in ("A", "B")
        for _ in (pair_root / f"potential{side}").glob("*.xvg")
    )


def validate_batch(
    data_root: Path,
    systems: list[str] | None = None,
    sites: list[str] | None = None,
) -> dict[str, int]:
    selected_systems = list(systems or EXPECTED_SYSTEMS)
    selected_sites = list(sites or EXPECTED_SITES)
    unknown_systems = sorted(set(selected_systems) - set(EXPECTED_SYSTEMS))
    unknown_sites = sorted(set(selected_sites) - set(EXPECTED_SITES))
    if unknown_systems:
        raise RuntimeError(f"Unknown system filter(s): {unknown_systems}")
    if unknown_sites:
        raise RuntimeError(f"Unknown site filter(s): {unknown_sites}")

    problems = []
    site_pairs = 0
    reference_pairs = 0
    xvg_files = 0
    for reaction_number in range(1, 5):
        reaction_root = data_root / f"reaction{reaction_number}"
        for system in selected_systems:
            for site in selected_sites:
                pair_root = reaction_root / system / site
                pair_problems = validate_pair(pair_root)
                if pair_problems:
                    problems.extend(pair_problems)
                else:
                    site_pairs += 1
                    xvg_files += count_pair_xvgs(pair_root)

        pair_root = data_root / f"ref_reaction{reaction_number}"
        pair_problems = validate_pair(pair_root)
        if pair_problems:
            problems.extend(pair_problems)
        else:
            reference_pairs += 1
            xvg_files += count_pair_xvgs(pair_root)

    if problems:
        raise RuntimeError("EVB batch preflight failed:\n" + "\n".join(problems))

    expected_xvgs = (site_pairs + reference_pairs) * 2 * len(EXPECTED_WINDOWS)
    if xvg_files != expected_xvgs:
        raise RuntimeError(
            f"EVB batch preflight found {xvg_files} XVGs; expected {expected_xvgs}."
        )
    return {
        "site_pairs": site_pairs,
        "reference_pairs": reference_pairs,
        "xvg_files": xvg_files,
        "systems": selected_systems,
        "sites": selected_sites,
    }


def load_reference_targets(path: Path | None):
    if path is None:
        return REFERENCE_TARGETS

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

    if len(targets) != 4:
        raise ValueError(f"{path}: expected four reference targets, found {len(targets)}")

    normalized = []
    for index, target in enumerate(targets, start=1):
        if not isinstance(target, (list, tuple)) or len(target) != 3:
            raise ValueError(
                f"{path}: reaction {index} target must be "
                "(barrier_kcal, free_energy_kcal, reverse_isfs)"
            )
        barrier, free_energy, reverse_isfs = target
        if not isinstance(reverse_isfs, bool):
            raise ValueError(f"{path}: reaction {index} reverse_isfs must be boolean")
        normalized.append((float(barrier), float(free_energy), reverse_isfs))
    return normalized


def batch_configs(data_root: Path, reference_targets=REFERENCE_TARGETS):
    configs = []
    for reaction_number, (config, target) in enumerate(
        zip(default_reaction_configs(), reference_targets), start=1
    ):
        barrier_kcal, free_energy_kcal, reverse_isfs = target
        configs.append(
            replace(
                config,
                collection_dir=f"reaction{reaction_number}",
                reference_pair=[
                    str(Path(f"ref_reaction{reaction_number}") / "potentialA"),
                    str(Path(f"ref_reaction{reaction_number}") / "potentialB"),
                ],
                barrier_kcal=barrier_kcal,
                free_energy_kcal=free_energy_kcal,
                reverse_isfs=reverse_isfs,
                parameter_mode=(
                    "fit_barrierless_reference"
                    if reaction_number == 3 and barrier_kcal <= 0.5
                    else "fit_reference"
                ),
                manual_alpha_kj=None,
                manual_h12_kj=None,
                plot_state_padding_fraction=0.10,
            )
        )
    return configs


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_results(rows, dimension):
    grouped = defaultdict(list)
    for row in rows:
        if row["status"] == "success":
            grouped[(row["reaction"], row[dimension])].append(row)

    output = []
    for (reaction, value), group in sorted(grouped.items()):
        ts = np.asarray([row["ts_kcal"] for row in group], dtype=float)
        dg = np.asarray([row["dg_kcal"] for row in group], dtype=float)
        output.append(
            {
                "reaction": reaction,
                dimension: value,
                "count": len(group),
                "ts_mean_kcal": float(np.mean(ts)),
                "ts_std_kcal": float(np.std(ts)),
                "ts_min_kcal": float(np.min(ts)),
                "ts_max_kcal": float(np.max(ts)),
                "dg_mean_kcal": float(np.mean(dg)),
                "dg_std_kcal": float(np.std(dg)),
                "dg_min_kcal": float(np.min(dg)),
                "dg_max_kcal": float(np.max(dg)),
            }
        )
    return output


def write_analysis(combined, output_dir: Path) -> None:
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    def kj_to_kcal(value):
        return None if value is None else value / 4.184

    def format_optional(value, digits: int = 4) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    fit_rows = []
    confidence_rows = []
    for summary in combined["summaries"]:
        config = summary["config"]
        fit = summary["fit"]
        results = summary["results"]
        successes = [row for row in results if row["status"] == "success"]
        confidence_counts = Counter(row["confidence"] for row in successes)
        fallback_counts = Counter(
            fallback for row in successes for fallback in row.get("fallbacks_used", [])
        )
        fit_rows.append(
            {
                "reaction": config["name"],
                "target_barrier_kcal": config["barrier_kcal"],
                "target_free_energy_kcal": config["free_energy_kcal"],
                "reverse_isfs": config["reverse_isfs"],
                "alpha_kj": fit.get("alpha"),
                "H12_kj": fit.get("H12"),
                "parameter_source": fit.get("parameter_source"),
                "fit_converged": fit.get("fit_converged"),
                "fit_iterations": fit.get("fit_iterations"),
                "fit_barrier_kcal": kj_to_kcal(fit.get("fit_barrier_kj")),
                "fit_free_energy_kcal": kj_to_kcal(fit.get("fit_free_energy_kj")),
                "barrier_residual_kj": fit.get("barrier_residual_kj"),
                "free_energy_residual_kj": fit.get("free_energy_residual_kj"),
                "coordinate_center_kj": fit.get("coordinate_center_kj"),
                "within_residual_tolerance": fit.get("within_residual_tolerance"),
                "coordinate_centered": fit.get("coordinate_centered"),
                "successful_mappings": len(successes),
                "failed_mappings": len(results) - len(successes),
            }
        )
        confidence_rows.append(
            {
                "reaction": config["name"],
                "high_confidence": confidence_counts.get("high", 0),
                "medium_confidence": confidence_counts.get("medium", 0),
                "low_confidence": confidence_counts.get("low", 0),
                "fallback_profiles": sum(
                    bool(row.get("fallbacks_used")) for row in successes
                ),
                "fallback_counts": "|".join(
                    f"{name}:{count}" for name, count in sorted(fallback_counts.items())
                ),
            }
        )

    fit_fields = list(fit_rows[0])
    write_csv(analysis_dir / "reaction_fit_quality.csv", fit_rows, fit_fields)
    write_csv(
        analysis_dir / "mapping_quality.csv",
        confidence_rows,
        list(confidence_rows[0]),
    )

    overlap_rows = []
    failed_rows = []
    for summary in combined["summaries"]:
        config = summary["config"]
        fit = summary["fit"]
        overlap = summary.get("reference_overlap")
        weak_frames = None
        overlap_warning = None
        if overlap:
            weak_frames = len(overlap.get("weak_overlap_frames", []))
            overlap_warning = overlap.get("warning")
        overlap_rows.append(
            {
                "reaction": config["name"],
                "parameter_source": fit.get("parameter_source"),
                "barrier_residual_kj": fit.get("barrier_residual_kj"),
                "free_energy_residual_kj": fit.get("free_energy_residual_kj"),
                "coordinate_center_kj": fit.get("coordinate_center_kj"),
                "within_residual_tolerance": fit.get("within_residual_tolerance"),
                "coordinate_centered": fit.get("coordinate_centered"),
                "reference_weak_overlap_frames": weak_frames,
                "reference_overlap_warning": overlap_warning,
                "fit_error": fit.get("fit_error"),
                "fit_message": fit.get("fit_message"),
            }
        )
        for row in summary["results"]:
            if row["status"] != "success":
                failed_rows.append(row)
    write_csv(
        analysis_dir / "overlap_quality.csv",
        overlap_rows,
        list(overlap_rows[0]),
    )
    write_csv(
        analysis_dir / "failed_cases.csv",
        failed_rows,
        [
            "reaction",
            "reaction_label",
            "system",
            "site",
            "status",
            "error",
            "failure_code",
            "message",
        ],
    )

    aggregate_fields = [
        "reaction",
        "dimension",
        "count",
        "ts_mean_kcal",
        "ts_std_kcal",
        "ts_min_kcal",
        "ts_max_kcal",
        "dg_mean_kcal",
        "dg_std_kcal",
        "dg_min_kcal",
        "dg_max_kcal",
    ]
    for dimension, filename in (
        ("system", "system_summary.csv"),
        ("site", "site_summary.csv"),
    ):
        aggregate_rows = aggregate_results(combined["results"], dimension)
        write_csv(
            analysis_dir / filename,
            aggregate_rows,
            [dimension if field == "dimension" else field for field in aggregate_fields],
        )

    ranked = []
    by_reaction = defaultdict(list)
    for row in combined["results"]:
        if row["status"] == "success":
            by_reaction[row["reaction"]].append(row)
    for reaction, rows in sorted(by_reaction.items()):
        for rank, row in enumerate(sorted(rows, key=lambda item: item["ts_kcal"]), start=1):
            ranked.append(
                {
                    "reaction": reaction,
                    "barrier_rank": rank,
                    "system": row["system"],
                    "site": row["site"],
                    "ts_kcal": row["ts_kcal"],
                    "dg_kcal": row["dg_kcal"],
                    "confidence": row["confidence"],
                    "assignment_mode": row["assignment_mode"],
                    "fallbacks_used": "|".join(row.get("fallbacks_used", [])),
                }
            )
    write_csv(
        analysis_dir / "site_results_ranked.csv",
        ranked,
        list(ranked[0]) if ranked else [
            "reaction",
            "barrier_rank",
            "system",
            "site",
            "ts_kcal",
            "dg_kcal",
            "confidence",
            "assignment_mode",
            "fallbacks_used",
        ],
    )

    lines = [
        "# New-reference EVB analysis",
        "",
        "All energies below are in kcal/mol unless noted otherwise.",
        "",
        "## Fit and mapping quality",
        "",
    ]
    for fit_row, quality_row in zip(fit_rows, confidence_rows):
        lines.append(
            f"- **{fit_row['reaction']}**: target "
            f"{fit_row['target_barrier_kcal']:.2f}/"
            f"{fit_row['target_free_energy_kcal']:.2f}; "
            f"fit residuals {format_optional(fit_row['barrier_residual_kj'])}/"
            f"{format_optional(fit_row['free_energy_residual_kj'])} kJ/mol; "
            f"{fit_row['successful_mappings']} successful and "
            f"{fit_row['failed_mappings']} failed mappings; confidence "
            f"H/M/L={quality_row['high_confidence']}/"
            f"{quality_row['medium_confidence']}/"
            f"{quality_row['low_confidence']}."
        )

    lines.extend(["", "## Barrier extremes", ""])
    for reaction, rows in sorted(by_reaction.items()):
        ordered = sorted(rows, key=lambda item: item["ts_kcal"])
        lowest = ordered[0]
        highest = ordered[-1]
        lines.append(
            f"- **{reaction}**: lowest {lowest['ts_kcal']:.2f} at "
            f"{lowest['system']}/{lowest['site']}; highest "
            f"{highest['ts_kcal']:.2f} at {highest['system']}/{highest['site']}."
        )

    lines.extend(
        [
            "",
            "## Figure range",
            "",
            "EVB comparison plots show the detected reactant-to-product state "
            "envelope with 10% coordinate padding. Full profile arrays remain "
            "available in the detailed JSON.",
            "",
        ]
    )
    (analysis_dir / "new_reference_analysis.md").write_text("\n".join(lines))

def print_run_summary(combined) -> None:
    print("\nEVB batch summary")
    for summary in combined["summaries"]:
        fit = summary["fit"]
        rows = summary["results"]
        successes = sum(row["status"] == "success" for row in rows)
        print(
            f"- {summary['config']['name']}: alpha={fit['alpha']:.6f}, "
            f"H12={fit['H12']:.6f}, converged={fit.get('fit_converged')}, "
            f"iterations={fit.get('fit_iterations')}, "
            f"residuals=({fit['barrier_residual_kj']:.6f}, "
            f"{fit['free_energy_residual_kj']:.6f}) kJ/mol, "
            f"sites={successes} success/{len(rows) - successes} failed"
        )


def main() -> int:
    args = build_parser().parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    target_file = args.target_file.resolve() if args.target_file else None
    reference_targets = load_reference_targets(target_file)
    preflight = validate_batch(data_root, systems=args.system, sites=args.site)
    if args.preflight_only:
        print("Preflight:", json.dumps(preflight, sort_keys=True))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    run_all_reactions(
        data_root,
        batch_configs(data_root, reference_targets),
        output_dir,
    )
    if target_file is not None:
        (output_dir / "reference_targets_source.txt").write_text(
            target_file.read_text()
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
