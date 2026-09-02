"""Shared helpers for the four-reaction MOF EVB demonstration."""

from __future__ import annotations

import csv
import json
import math as m
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover - cluster environments should provide scipy
    minimize = None

import evb_tools as cxt
import evb_tools.evb_utils as evb_utils
from evb_tools.evb_utils import EvbMappingError

KCAL_TO_KJ = 4.184
FIT_RESIDUAL_TOLERANCE_KJ = 5.0
COORDINATE_CENTER_TOLERANCE_KJ = 30.0
COORDINATE_CENTER_WEIGHT = 0.02
AGGRESSIVE_MAXITER = 45
PMF_Y_RANGE_CAUTION_KCAL = 20.0


@dataclass(frozen=True)
class ReactionConfig:
    name: str
    label: str
    collection_dir: str
    reference_pair: list[str]
    barrier_kcal: float
    free_energy_kcal: float
    reverse_isfs: bool
    resolution_presets: list[dict[str, Any]]
    parameter_mode: str
    fit_max_iterations: int = 2000
    manual_alpha_kj: float | None = None
    manual_h12_kj: float | None = None
    use_stable_boltzmann: bool = False
    plot_state_padding_fraction: float | None = None
    barrierless_barrier_tolerance_kcal: float = 0.10
    barrierless_free_energy_tolerance_kcal: float = 0.05


def stable_log_mean_exp(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise EvbMappingError("Cannot average an empty Boltzmann sample.")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise EvbMappingError("Boltzmann sample contains no finite values.")
    vmax = float(np.max(finite))
    return vmax + m.log(float(np.mean(np.exp(finite - vmax))))


def stable_gofl(feps, frames, N, kT, dl):
    dgf = []
    dgb = []
    dg_lambda = [0.0]
    log_n = m.log(N)
    for frame in range(frames - 1):
        forward_logsum = stable_log_mean_exp(feps[frame][2] * dl / kT) + log_n
        backward_logsum = stable_log_mean_exp(-feps[frame + 1][2] * dl / kT) + log_n
        dgf.append(-kT * (forward_logsum - log_n))
        dgb.append(kT * (backward_logsum - log_n))
        dg_lambda.append(dg_lambda[-1] + 0.5 * (dgf[frame] + dgb[frame]))
    lc = [dl * frame for frame in range(frames)]
    lc.reverse()
    return dg_lambda, lc


def stable_gofx(dg_lambda, free_en, bins, frames, minpts, kT, x, N):
    del N
    dx = []
    g1 = []
    g2 = []
    gg = []
    for nbin in range(bins):
        dga = []
        dgb = []
        dgg = []
        weights = []
        bin_coordinate = None
        for frame in range(frames):
            if len(free_en[nbin][frame][0]) < minpts:
                continue
            e1_values = np.asarray(free_en[nbin][frame][0], dtype=float)
            e2_values = np.asarray(free_en[nbin][frame][1], dtype=float)
            gg_values = np.asarray(free_en[nbin][frame][2], dtype=float)
            npts = len(e1_values)
            avv1 = float(np.mean(e1_values))
            avv2 = float(np.mean(e2_values))
            avvgg = float(np.mean(gg_values))
            dga.append(dg_lambda[frame] - kT * stable_log_mean_exp(-(e1_values - avv1) / kT) + avv1)
            dgb.append(dg_lambda[frame] - kT * stable_log_mean_exp(-(e2_values - avv2) / kT) + avv2)
            dgg.append(dg_lambda[frame] - kT * stable_log_mean_exp(-(gg_values - avvgg) / kT) + avvgg)
            weights.append(npts)
            if bin_coordinate is None:
                bin_coordinate = x[nbin]
        if weights:
            dx.append(bin_coordinate)
            g1.append(np.average(dga, weights=weights))
            g2.append(np.average(dgb, weights=weights))
            gg.append(np.average(dgg, weights=weights))
    dx = np.asarray(dx, dtype=float)
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    gg = np.asarray(gg, dtype=float)
    if len(x) > 1 and dx.size:
        dx = dx - (x[1] - x[0]) / 2
    return dx, g1, g2, gg


ORIGINAL_GOFL = evb_utils.gofl
ORIGINAL_GOFX = evb_utils.gofx


def configure_boltzmann_helpers(use_stable: bool):
    evb_utils.gofl = stable_gofl if use_stable else ORIGINAL_GOFL
    evb_utils.gofx = stable_gofx if use_stable else ORIGINAL_GOFX


def build_evb(config: ReactionConfig, preset: dict[str, Any]):
    evb = cxt.Evb()
    evb.verbose = False
    evb.barrier = config.barrier_kcal * KCAL_TO_KJ
    evb.free_energy = config.free_energy_kcal * KCAL_TO_KJ
    evb.reverse_isfs = config.reverse_isfs

    evb.step = preset["step"]
    evb.bins = preset["bins"]
    evb.minpts = preset["minpts"]

    evb.profile_secondary_smoothing = False
    evb.profile_secondary_smoothing_for_fitting = False
    evb.profile_smoothing_gaussian_sigma = 1.2
    evb.profile_smoothing_savgol_window = 5
    evb.profile_smoothing_savgol_polyorder = 3

    evb.state_detection_enabled = True
    evb.state_assignment_policy = "robust"
    evb.state_smoothing_gaussian_sigma = 1.2
    evb.state_smoothing_savgol_window = 9
    evb.state_smoothing_savgol_polyorder = 3
    evb.state_min_prominence = preset["state_min_prominence"]
    evb.state_min_basin_width = preset["state_min_basin_width"]
    evb.state_edge_exclusion_bins = preset["state_edge_exclusion_bins"]
    evb.state_auto_retry_resolution = True
    if "state_auto_retry_resolution" in preset:
        evb.state_auto_retry_resolution = bool(preset["state_auto_retry_resolution"])
    return evb


def discover_active_systems(root: Path, collection_dir: str):
    base = root / collection_dir
    if not base.is_dir():
        raise FileNotFoundError(f"Collection directory not found: {base}")
    systems = {
        sysdir.name: sorted(
            p.name
            for p in sysdir.iterdir()
            if p.is_dir() and (p / "potentialA").is_dir() and (p / "potentialB").is_dir()
        )
        for sysdir in sorted(base.iterdir())
        if sysdir.is_dir()
    }
    if not systems:
        raise FileNotFoundError(f"No systems with potentialA/potentialB found in {base}")
    return systems


def map_pair_with_presets(config: ReactionConfig, preset_evbs, pair, rs_hint_x_kj=None):
    last_error = None
    for preset in config.resolution_presets:
        evb = preset_evbs[preset["name"]]
        try:
            data = evb.evb_systems({0: {"AB_pairs": [pair]}}, plot=False, kcal=False)
        except EvbMappingError as exc:
            last_error = exc
            continue

        meta = data[0]["state_detection"][0]
        raw = np.asarray(meta["raw_pmf"], dtype=float)
        smooth = np.asarray(meta["smoothed_pmf"], dtype=float)
        states = {
            name: {
                "x_kj": float(state["x"]),
                "raw_y_kj": float(state["raw_y"]),
                "source": state["source"],
                "penalties": list(state["penalties"]),
                "accepted_strictly": bool(state["accepted_strictly"]),
            }
            for name, state in meta["states"].items()
        }
        coordinate_center_kj = (
            states.get("TS", {}).get("x_kj")
            if "TS" in states
            else _coordinate_center_from_state_detection(meta)
        )
        profile = {
            "x_kj": smooth[:, 0].tolist(),
            "raw_kj": raw[:, 1].tolist(),
            "smooth_kj": smooth[:, 1].tolist(),
            "states": states,
        }
        central_window_mapping = False
        central_window_failure = None
        central_window_info = None
        if preset.get("central_window_kj") is not None:
            central_window_mapping, central_window_info = _apply_central_window_mapping(
                profile,
                preset["central_window_kj"],
                y_range_limit_kcal=preset.get("central_window_max_y_range_kcal"),
                rs_hint_x_kj=rs_hint_x_kj,
            )
            if not central_window_mapping and preset.get("require_central_window"):
                central_window_failure = central_window_info
                last_error = EvbMappingError(
                    f"Central-window mapping failed: {central_window_failure}"
                )
                continue
        repaired_rs = False
        if preset.get("require_rs_basin") and not _rs_is_local_basin(profile):
            repaired_rs = _repair_rs_to_left_basin(profile)
            if not repaired_rs or not _rs_is_local_basin(profile):
                last_error = EvbMappingError(
                    "Mapped RS is not a local left-side basin and no repairable left basin was found."
                )
                continue
        repaired_ts = _profile_state_delta_kj(profile, "TS")
        repaired_dg = _profile_state_delta_kj(profile, "PS")
        ts_kj = float(data[0]["ts_energy"][0]) if repaired_ts is None else repaired_ts
        dg_kj = float(data[0]["dg_energy"][0]) if repaired_dg is None else repaired_dg
        coordinate_center_kj = (
            profile["states"].get("TS", {}).get("x_kj")
            if "TS" in profile["states"]
            else coordinate_center_kj
        )
        y_range_kcal = _pmf_y_range_kcal(profile)
        warnings = list(meta["warnings"])
        fallbacks_used = list(meta["fallbacks_used"])
        if central_window_mapping:
            warnings.append(
                f"Applied central-window state mapping within +/-{float(central_window_info['half_width_kj']):.1f} kJ/mol around raw e1-e2=0."
            )
            fallbacks_used.append("central_window_mapping")
        if repaired_rs:
            warnings.append(
                "Repaired RS to the nearest supported left-side local basin before reporting TS and dG."
            )
            fallbacks_used.append("repaired_rs_basin")
        if y_range_kcal is not None and y_range_kcal > PMF_Y_RANGE_CAUTION_KCAL:
            warnings.append(
                f"RS-normalized PMF y-range is {y_range_kcal:.2f} kcal/mol, above the {PMF_Y_RANGE_CAUTION_KCAL:.1f} kcal/mol caution threshold."
            )
        meta = dict(meta)
        meta["states"] = {
            name: {
                **meta.get("states", {}).get(name, {}),
                "x": state["x_kj"],
                "raw_y": state["raw_y_kj"],
                "source": state["source"],
                "penalties": list(state.get("penalties", [])),
                "accepted_strictly": bool(state.get("accepted_strictly", False)),
            }
            for name, state in profile["states"].items()
        }
        meta["warnings"] = warnings
        meta["fallbacks_used"] = list(dict.fromkeys(fallbacks_used))
        return {
            "preset": preset["name"],
            "ts_kj": ts_kj,
            "dg_kj": dg_kj,
            "coordinate_center_kj": coordinate_center_kj,
            "coordinate_centered": (
                coordinate_center_kj is not None
                and abs(float(coordinate_center_kj)) <= COORDINATE_CENTER_TOLERANCE_KJ
            ),
            "state_detection": meta,
            "assignment_mode": meta["assignment_mode"],
            "confidence": meta["confidence"],
            "fallbacks_used": meta["fallbacks_used"],
            "warnings": meta["warnings"],
            "rs_is_local_basin": _rs_is_local_basin(profile),
            "rs_repaired": repaired_rs,
            "central_window_mapping": central_window_mapping,
            "central_window_kj": (
                central_window_info.get("half_width_kj")
                if isinstance(central_window_info, dict)
                else preset.get("central_window_kj")
            ),
            "central_window_y_range_kcal": (
                central_window_info.get("y_range_kcal")
                if isinstance(central_window_info, dict)
                else None
            ),
            "pmf_y_range_kcal": y_range_kcal,
            "profile": profile,
        }, None
    return None, last_error


def reference_state_bins_payload(alpha, h12, state_detection, metadata=None):
    resolution = state_detection["resolution"]
    profile_x = np.asarray(state_detection.get("smoothed_pmf", []), dtype=float)
    if profile_x.ndim == 2 and profile_x.shape[1]:
        profile_x = profile_x[:, 0]
    else:
        profile_x = np.asarray([], dtype=float)

    def derived_bin_payload(state):
        if "bin_index" in state:
            return {
                "bin_index": int(state["bin_index"]),
                "bin_lower_kj": float(state["bin_lower_kj"]),
                "bin_upper_kj": float(state["bin_upper_kj"]),
                "points": [
                    {
                        "window_index": int(point["window_index"]),
                        "lambda": float(point["lambda"]),
                        "time_ps": float(point["time_ps"]),
                    }
                    for point in state.get("points", [])
                ],
            }

        state_x = float(state["x"])
        if profile_x.size:
            bin_index = int(np.argmin(np.abs(profile_x - state_x)))
            if profile_x.size > 1:
                if bin_index == 0:
                    width = float(abs(profile_x[1] - profile_x[0]))
                elif bin_index == profile_x.size - 1:
                    width = float(abs(profile_x[-1] - profile_x[-2]))
                else:
                    width = float(abs(profile_x[bin_index + 1] - profile_x[bin_index - 1]) / 2.0)
            else:
                width = 0.0
            center = float(profile_x[bin_index])
            bin_lower = center - width / 2.0
            bin_upper = center + width / 2.0
        else:
            bin_index = int(state.get("index", -1))
            bin_lower = state_x
            bin_upper = state_x

        return {
            "bin_index": bin_index,
            "bin_lower_kj": float(bin_lower),
            "bin_upper_kj": float(bin_upper),
            "x_kj": state_x,
            "raw_y_kj": float(state["raw_y"]),
            "source": state.get("source"),
            "penalties": list(state.get("penalties", [])),
            "accepted_strictly": bool(state.get("accepted_strictly", False)),
            "points": [],
        }

    states = {}
    for name in ("RS", "TS", "PS"):
        state = state_detection["states"][name]
        states[name] = derived_bin_payload(state)
    payload = {
        "alpha_kj": float(alpha),
        "h12_kj": float(h12),
        "step": int(resolution["step"]),
        "bins": int(resolution["bins"]),
        "minpts": int(resolution["minpts"]),
        "assignment_mode": state_detection.get("assignment_mode"),
        "confidence": state_detection.get("confidence"),
        "fallbacks_used": list(state_detection.get("fallbacks_used", [])),
        "warnings": list(state_detection.get("warnings", [])),
        "states": states,
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    return payload


def write_reference_state_bins(path, alpha, h12, state_detection, metadata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = reference_state_bins_payload(
        alpha, h12, state_detection, metadata=metadata
    )
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def pair_xvgs(pair):
    A_xvgs, B_xvgs = evb_utils.fetchA_B_xvgs(pair[0], pair[1])
    if not A_xvgs or not B_xvgs:
        raise FileNotFoundError(f"No XVG files found for pair: {pair}")
    return A_xvgs, B_xvgs


def summarize_diabatic_gap(pair, alpha, reverse_isfs, step=1):
    A_xvgs, B_xvgs = pair_xvgs(pair)
    feps, sampled_points = evb_utils.read_data(
        A_xvgs,
        B_xvgs,
        alpha=alpha,
        hij=20.0,
        dl=0.05,
        step=step,
        reverse_isfs=reverse_isfs,
    )
    gap = feps[:, 2, :]
    frame_stats = []
    weak_overlap_frames = []
    for frame, values in enumerate(gap):
        p05, p50, p95 = np.percentile(values, [5, 50, 95])
        has_crossing_support = p05 <= 0.0 <= p95
        if not has_crossing_support:
            weak_overlap_frames.append(frame)
        frame_stats.append(
            {
                "frame": int(frame),
                "mean_kj": float(np.mean(values)),
                "p05_kj": float(p05),
                "p50_kj": float(p50),
                "p95_kj": float(p95),
                "has_crossing_support": bool(has_crossing_support),
            }
        )

    flat = gap.reshape(-1)
    warning = None
    if len(weak_overlap_frames) > len(frame_stats) // 2:
        warning = (
            "Most lambda windows do not sample both diabatic states near the crossing; "
            "the mapped profile may be endpoint-dominated."
        )
    elif weak_overlap_frames:
        warning = "Some lambda windows have weak crossing overlap."

    return {
        "frames": int(gap.shape[0]),
        "sampled_points_per_frame": int(sampled_points),
        "gap_min_kj": float(np.min(flat)),
        "gap_max_kj": float(np.max(flat)),
        "gap_mean_kj": float(np.mean(flat)),
        "gap_std_kj": float(np.std(flat)),
        "frame_stats": frame_stats,
        "weak_overlap_frames": weak_overlap_frames,
        "warning": warning,
    }


def fit_reference_with_diagnostics(config: ReactionConfig, reference_pair):
    fit_evb = build_evb(config, config.resolution_presets[0])
    fit_evb.max_iter = config.fit_max_iterations
    auto_settings = fit_evb.guess_fit_settings_from_ref([reference_pair], apply=True)
    fit_evb._apply_state_detection_settings(
        enabled=fit_evb.profile_secondary_smoothing_for_fitting
    )
    normalized_pair = fit_evb._normalize_pairs([reference_pair])[0]
    A_xvgs, B_xvgs = evb_utils.fetchA_B_xvgs(
        normalized_pair[0], normalized_pair[1]
    )
    fit_result = fit_evb._fit_evb_parameters(A_xvgs, B_xvgs)

    alpha = float(fit_result["alpha"])
    h12 = float(fit_result["H12"])
    fit_evb.alpha = alpha
    fit_evb.H12 = h12
    fit_evb._parameter_source = "fit"

    target_barrier_kj = config.barrier_kcal * KCAL_TO_KJ
    target_free_energy_kj = config.free_energy_kcal * KCAL_TO_KJ
    fitted_barrier_kj = float(fit_result["barrier"])
    fitted_free_energy_kj = float(fit_result["free_energy"])
    return {
        "alpha": alpha,
        "H12": h12,
        "parameter_source": "fit_reference",
        "converged": bool(fit_result["converged"]),
        "fit_converged": bool(fit_result["converged"]),
        "fit_iterations": int(fit_result["iterations"]),
        "fit_max_iterations": int(fit_evb.max_iter),
        "fit_barrier_kj": fitted_barrier_kj,
        "fit_free_energy_kj": fitted_free_energy_kj,
        "target_barrier_kj": target_barrier_kj,
        "target_free_energy_kj": target_free_energy_kj,
        "barrier_residual_kj": fitted_barrier_kj - target_barrier_kj,
        "free_energy_residual_kj": fitted_free_energy_kj - target_free_energy_kj,
        "fit_settings": {
            "alpha_guess": float(auto_settings["alpha_guess"]),
            "H12_guess": float(auto_settings["H12_guess"]),
            "step": int(auto_settings["step"]),
            "bins": int(auto_settings["bins"]),
            "minpts": int(auto_settings["minpts"]),
        },
    }


def _coordinate_center_from_state_detection(state_detection):
    try:
        return float(state_detection["states"]["TS"]["x"])
    except (KeyError, TypeError, ValueError):
        return None


def _state_index(profile, state_name):
    state = profile.get("states", {}).get(state_name)
    if not state:
        return None
    x_values = np.asarray(profile.get("x_kj", []), dtype=float)
    if x_values.size == 0:
        return None
    return int(np.argmin(np.abs(x_values - float(state["x_kj"]))))


def _state_y_reference(profile, state_name="RS"):
    index = _state_index(profile, state_name)
    values = np.asarray(profile.get("smooth_kj", []), dtype=float)
    if index is None or index < 0 or index >= values.size:
        finite = values[np.isfinite(values)]
        return float(finite[0]) if finite.size else 0.0
    return float(values[index])


def _normalized_profile_values(profile):
    raw = np.asarray(profile.get("raw_kj", []), dtype=float)
    smooth = np.asarray(profile.get("smooth_kj", []), dtype=float)
    reference = _state_y_reference(profile, "RS")
    return raw - reference, smooth - reference, reference


def _pmf_y_range_kcal(profile):
    _, smooth_norm, _ = _normalized_profile_values(profile)
    finite = smooth_norm[np.isfinite(smooth_norm)]
    if finite.size == 0:
        return None
    return float((np.max(finite) - np.min(finite)) / KCAL_TO_KJ)


def _rs_is_local_basin(profile):
    index = _state_index(profile, "RS")
    values = np.asarray(profile.get("smooth_kj", []), dtype=float)
    if index is None or values.size < 3:
        return False
    if index <= 0 or index >= values.size - 1:
        return False
    left = values[index - 1]
    center = values[index]
    right = values[index + 1]
    return bool(np.isfinite(center) and center <= left and center <= right)


def _repair_rs_to_left_basin(profile):
    states = profile.get("states", {})
    if "TS" not in states or "PS" not in states:
        return False
    x_values = np.asarray(profile.get("x_kj", []), dtype=float)
    raw = np.asarray(profile.get("raw_kj", []), dtype=float)
    smooth = np.asarray(profile.get("smooth_kj", []), dtype=float)
    if x_values.size < 3:
        return False
    ts_index = _state_index(profile, "TS")
    ps_index = _state_index(profile, "PS")
    if ts_index is None or ps_index is None:
        return False
    search_stop = max(1, min(ts_index, ps_index) - 1)
    candidates = []
    for index in range(1, search_stop):
        if not np.isfinite(smooth[index]):
            continue
        if smooth[index] <= smooth[index - 1] and smooth[index] <= smooth[index + 1]:
            left_relief = smooth[index - 1] - smooth[index]
            right_relief = smooth[index + 1] - smooth[index]
            candidates.append(
                (
                    min(left_relief, right_relief),
                    -smooth[index],
                    index,
                )
            )
    if not candidates:
        return False
    _, _, rs_index = max(candidates)
    states["RS"] = {
        **states.get("RS", {}),
        "x_kj": float(x_values[rs_index]),
        "raw_y_kj": float(raw[rs_index]),
        "source": "repaired_left_basin",
        "penalties": ["repaired_rs_basin"],
        "accepted_strictly": False,
    }

    between_start = rs_index + 1
    between_stop = max(between_start + 1, ps_index)
    if between_stop > between_start:
        ts_index = int(between_start + np.argmax(smooth[between_start:between_stop]))
        states["TS"] = {
            **states.get("TS", {}),
            "x_kj": float(x_values[ts_index]),
            "raw_y_kj": float(raw[ts_index]),
            "source": "repaired_bracketed_maximum",
            "penalties": ["repaired_rs_basin"],
            "accepted_strictly": False,
        }
    return True


def _side_minimum_index(x_values, smooth, indices, hint_x_kj=None):
    """Return the index of the best minimum within *indices*.

    When *hint_x_kj* is provided, minima close to that x-coordinate are
    preferred — this anchors RS detection to the reference system's basin
    position for barrierless or nearly-flat profiles where the raw PMF
    offers little topological guidance.
    """
    if indices.size == 0:
        return None
    local = []
    for index in indices:
        if index <= 0 or index >= smooth.size - 1:
            continue
        if not np.isfinite(smooth[index]):
            continue
        if smooth[index] <= smooth[index - 1] and smooth[index] <= smooth[index + 1]:
            relief = min(smooth[index - 1] - smooth[index], smooth[index + 1] - smooth[index])
            hint_penalty = abs(x_values[index] - hint_x_kj) if hint_x_kj is not None else 0.0
            local.append((relief, -hint_penalty, -smooth[index], int(index)))
    if local:
        if hint_x_kj is not None:
            # For barrierless profiles the deepest tiny local dip can sit near
            # the crossing.  When an anchor is available, choose the local
            # basin nearest that coordinate instead of letting small relief
            # differences pull RS back toward TS.
            return min(
                local,
                key=lambda item: (
                    abs(x_values[item[-1]] - hint_x_kj),
                    smooth[item[-1]],
                    -item[0],
                ),
            )[-1]
        return max(local)[-1]
    if hint_x_kj is not None:
        ranked = sorted(
            (
                (
                    abs(x_values[index] - hint_x_kj),
                    smooth[index],
                    int(index),
                )
                for index in indices
                if np.isfinite(smooth[index])
            ),
            key=lambda item: (item[0], item[1]),
        )
    else:
        ranked = sorted(
            (
                (
                    smooth[index],
                    abs(x_values[index]),
                    int(index),
                )
                for index in indices
                if np.isfinite(smooth[index])
            ),
            key=lambda item: (item[0], item[1]),
        )
    return ranked[0][-1] if ranked else None


def _central_window_candidate(profile, half_width_kj, rs_hint_x_kj=None):
    states = profile.get("states", {})
    x_values = np.asarray(profile.get("x_kj", []), dtype=float)
    raw = np.asarray(profile.get("raw_kj", []), dtype=float)
    smooth = np.asarray(profile.get("smooth_kj", []), dtype=float)
    if x_values.size < 3:
        return None, "central_window_too_short"

    half_width = float(half_width_kj)
    central = np.where(np.isfinite(x_values) & (np.abs(x_values) <= half_width))[0]
    left = central[x_values[central] < 0.0]
    right = central[x_values[central] > 0.0]
    ps_index = _side_minimum_index(x_values, smooth, right)
    ps_symmetric_hint_x_kj = (
        -abs(float(x_values[ps_index])) if ps_index is not None else None
    )
    effective_rs_hint_x_kj = rs_hint_x_kj
    anchor_source = "reference_rs"
    if ps_symmetric_hint_x_kj is not None and (
        effective_rs_hint_x_kj is None
        or abs(float(effective_rs_hint_x_kj)) < 0.5 * abs(ps_symmetric_hint_x_kj)
    ):
        effective_rs_hint_x_kj = ps_symmetric_hint_x_kj
        anchor_source = "ps_symmetric"
    rs_index = _side_minimum_index(
        x_values,
        smooth,
        left,
        hint_x_kj=effective_rs_hint_x_kj,
    )
    if rs_index is None or ps_index is None:
        return None, "central_window_missing_bracketing_basins"

    between_start = min(rs_index, ps_index)
    between_stop = max(rs_index, ps_index)
    if between_stop - between_start > 1:
        between = np.arange(between_start + 1, between_stop, dtype=int)
    else:
        between = np.arange(between_start, between_stop + 1, dtype=int)
    between = between[np.isfinite(smooth[between])]
    if between.size == 0:
        return None, "central_window_missing_ts"
    ts_index = int(between[np.argmax(smooth[between])])

    y_values = smooth[central]
    y_values = y_values[np.isfinite(y_values)]
    y_range_kcal = None
    if y_values.size:
        y_range_kcal = float((np.max(y_values) - np.min(y_values)) / KCAL_TO_KJ)

    return {
        "half_width_kj": half_width,
        "rs_index": int(rs_index),
        "ts_index": int(ts_index),
        "ps_index": int(ps_index),
        "y_range_kcal": y_range_kcal,
        "ts_kj": float(smooth[ts_index] - smooth[rs_index]),
        "dg_kj": float(smooth[ps_index] - smooth[rs_index]),
        "rs_anchor_x_kj": (
            float(effective_rs_hint_x_kj)
            if effective_rs_hint_x_kj is not None
            else None
        ),
        "rs_anchor_source": anchor_source,
    }, None


def _apply_central_window_mapping(profile, half_width_kj, y_range_limit_kcal=None, rs_hint_x_kj=None):
    windows = half_width_kj if isinstance(half_width_kj, (list, tuple)) else [half_width_kj]
    candidates = []
    failures = []
    for window in windows:
        candidate, failure = _central_window_candidate(profile, window, rs_hint_x_kj=rs_hint_x_kj)
        if candidate is None:
            failures.append(f"{window}:{failure}")
            continue
        candidates.append(candidate)
        if (
            y_range_limit_kcal is None
            or candidate["y_range_kcal"] is None
            or candidate["y_range_kcal"] <= float(y_range_limit_kcal)
        ):
            break
    if not candidates:
        return False, "|".join(failures) or "central_window_no_candidates"

    if y_range_limit_kcal is not None:
        acceptable = [
            candidate
            for candidate in candidates
            if candidate["y_range_kcal"] is not None
            and candidate["y_range_kcal"] <= float(y_range_limit_kcal)
        ]
        if acceptable:
            best = acceptable[0]
        else:
            best = min(candidates, key=lambda item: item["y_range_kcal"] or float("inf"))
            if best["y_range_kcal"] is not None and best["y_range_kcal"] > float(y_range_limit_kcal):
                return False, (
                    f"central_window_y_range_{best['y_range_kcal']:.2f}_kcal_exceeds_"
                    f"{float(y_range_limit_kcal):.2f}"
                )
    else:
        best = candidates[0]

    states = profile.get("states", {})
    x_values = np.asarray(profile.get("x_kj", []), dtype=float)
    raw = np.asarray(profile.get("raw_kj", []), dtype=float)

    rs_index = best["rs_index"]
    ts_index = best["ts_index"]
    ps_index = best["ps_index"]
    states["RS"] = {
        **states.get("RS", {}),
        "x_kj": float(x_values[rs_index]),
        "raw_y_kj": float(raw[rs_index]),
        "source": "central_window",
        "penalties": ["central_window_mapping"],
        "accepted_strictly": False,
    }
    states["TS"] = {
        **states.get("TS", {}),
        "x_kj": float(x_values[ts_index]),
        "raw_y_kj": float(raw[ts_index]),
        "source": "central_window",
        "penalties": ["central_window_mapping"],
        "accepted_strictly": False,
    }
    states["PS"] = {
        **states.get("PS", {}),
        "x_kj": float(x_values[ps_index]),
        "raw_y_kj": float(raw[ps_index]),
        "source": "central_window",
        "penalties": ["central_window_mapping"],
        "accepted_strictly": False,
    }
    return True, best


def _profile_state_delta_kj(profile, numerator_state, denominator_state="RS"):
    numerator_index = _state_index(profile, numerator_state)
    denominator_index = _state_index(profile, denominator_state)
    smooth = np.asarray(profile.get("smooth_kj", []), dtype=float)
    if (
        numerator_index is None
        or denominator_index is None
        or numerator_index >= smooth.size
        or denominator_index >= smooth.size
    ):
        return None
    return float(smooth[numerator_index] - smooth[denominator_index])


def _coordinate_origin_kj(profile):
    ts = profile.get("states", {}).get("TS")
    if ts and np.isfinite(ts.get("x_kj", np.nan)):
        return float(ts["x_kj"])
    x_values = np.asarray(profile.get("x_kj", []), dtype=float)
    finite = x_values[np.isfinite(x_values)]
    if finite.size == 0:
        return 0.0
    return float(0.5 * (np.min(finite) + np.max(finite)))


def _centered_x_values(profile):
    x_values = np.asarray(profile.get("x_kj", []), dtype=float)
    origin = _coordinate_origin_kj(profile)
    return x_values - origin, origin


def _fit_quality_score(barrier_residual, free_energy_residual, coordinate_center):
    score = float(barrier_residual) ** 2 + float(free_energy_residual) ** 2
    if coordinate_center is not None and np.isfinite(coordinate_center):
        score += (COORDINATE_CENTER_WEIGHT * float(coordinate_center)) ** 2
    return float(score)


def _fit_acceptance(barrier_residual, free_energy_residual, coordinate_center):
    residuals_ok = (
        abs(float(barrier_residual)) <= FIT_RESIDUAL_TOLERANCE_KJ
        and abs(float(free_energy_residual)) <= FIT_RESIDUAL_TOLERANCE_KJ
    )
    center_ok = (
        coordinate_center is not None
        and np.isfinite(coordinate_center)
        and abs(float(coordinate_center)) <= COORDINATE_CENTER_TOLERANCE_KJ
    )
    return residuals_ok and center_ok


def aggressive_fit_reference_with_diagnostics(
    config: ReactionConfig,
    reference_pair,
    initial_result=None,
):
    fit_evb = build_evb(config, config.resolution_presets[0])
    fit_evb.max_iter = config.fit_max_iterations
    auto_settings = fit_evb.guess_fit_settings_from_ref([reference_pair], apply=True)
    fit_evb._apply_state_detection_settings(
        enabled=fit_evb.profile_secondary_smoothing_for_fitting
    )
    normalized_pair = fit_evb._normalize_pairs([reference_pair])[0]
    A_xvgs, B_xvgs = evb_utils.fetchA_B_xvgs(normalized_pair[0], normalized_pair[1])

    target_barrier_kj = config.barrier_kcal * KCAL_TO_KJ
    target_free_energy_kj = config.free_energy_kcal * KCAL_TO_KJ
    cache = {}
    candidates = []

    def evaluate(alpha, h12, source, iterations=0, converged=False):
        key = (round(float(alpha), 8), round(float(h12), 8))
        if key in cache:
            candidate = dict(cache[key])
            candidate["source"] = source
            return candidate
        ts_energy, dg_energy, state_detection = evb_utils.evbmap_nooutput(
            A_xvgs,
            B_xvgs,
            fit_evb.step,
            float(alpha),
            float(h12),
            fit_evb.dl,
            fit_evb.kb,
            fit_evb.temp,
            fit_evb.bins,
            fit_evb.minpts,
            fit_evb.reverse_isfs,
            return_metadata=True,
            allow_resolution_retry=fit_evb._allow_state_detection_retry(),
        )
        coordinate_center = _coordinate_center_from_state_detection(state_detection)
        barrier_residual = float(ts_energy) - target_barrier_kj
        free_energy_residual = float(dg_energy) - target_free_energy_kj
        candidate = {
            "alpha": float(alpha),
            "H12": float(h12),
            "barrier": float(ts_energy),
            "free_energy": float(dg_energy),
            "barrier_residual_kj": barrier_residual,
            "free_energy_residual_kj": free_energy_residual,
            "coordinate_center_kj": coordinate_center,
            "score": _fit_quality_score(
                barrier_residual, free_energy_residual, coordinate_center
            ),
            "fit_converged": bool(converged),
            "fit_iterations": int(iterations),
            "state_detection": state_detection,
            "source": source,
        }
        cache[key] = dict(candidate)
        return candidate

    if initial_result is not None and "alpha" in initial_result and "H12" in initial_result:
        try:
            candidates.append(
                evaluate(
                    initial_result["alpha"],
                    initial_result["H12"],
                    "standard_fit",
                    initial_result.get("fit_iterations", 0),
                    initial_result.get("fit_converged", False),
                )
            )
        except Exception:
            pass

    h12_guess = float(auto_settings["H12_guess"])
    h12_magnitudes = sorted(
        {
            max(20.0, abs(h12_guess) * 0.5),
            max(20.0, abs(h12_guess)),
            max(20.0, abs(h12_guess) * 1.5),
        }
    )
    starts = [
        (auto_settings["alpha_guess"], sign * magnitude)
        for magnitude in h12_magnitudes
        for sign in (1.0, -1.0)
    ]
    if candidates:
        starts.insert(0, (candidates[0]["alpha"], candidates[0]["H12"]))

    alpha_span = max(250.0, abs(auto_settings["alpha_guess"]) + 1200.0)
    h12_span = max(500.0, abs(auto_settings["H12_guess"]) * 4.0 + 500.0)
    bounds = [
        (auto_settings["alpha_guess"] - alpha_span, auto_settings["alpha_guess"] + alpha_span),
        (-h12_span, h12_span),
    ]

    def objective(params):
        alpha, h12 = params
        if not np.isfinite(alpha) or not np.isfinite(h12):
            return 1.0e12
        try:
            candidate = evaluate(alpha, h12, "aggressive_fit")
        except Exception:
            return 1.0e10
        return candidate["score"]

    alpha_offsets = sorted(
        {
            -1000.0,
            -600.0,
            -300.0,
            -100.0,
            -target_free_energy_kj,
            0.0,
            target_free_energy_kj,
            100.0,
            300.0,
            600.0,
            1000.0,
        }
    )
    for alpha0, h120 in starts:
        for da in alpha_offsets:
            for scale in (0.75, 1.0, 1.25):
                try:
                    candidates.append(
                        evaluate(alpha0 + da, h120 * scale, "aggressive_grid")
                    )
                except Exception:
                    continue

    if candidates:
        coarse_best = min(candidates, key=lambda item: item["score"])
        for da in (-25.0, 0.0, 25.0):
            for dh in (-50.0, 0.0, 50.0):
                try:
                    candidates.append(
                        evaluate(
                            coarse_best["alpha"] + da,
                            coarse_best["H12"] + dh,
                            "aggressive_grid_refined",
                        )
                    )
                except Exception:
                    continue

    if not candidates:
        raise EvbMappingError("Aggressive reference fit produced no finite candidates.")

    best = min(candidates, key=lambda item: item["score"])
    return {
        "alpha": best["alpha"],
        "H12": best["H12"],
        "parameter_source": best["source"],
        "converged": bool(
            best["fit_converged"]
            or _fit_acceptance(
                best["barrier_residual_kj"],
                best["free_energy_residual_kj"],
                best["coordinate_center_kj"],
            )
        ),
        "fit_converged": bool(best["fit_converged"]),
        "fit_iterations": int(best["fit_iterations"]),
        "fit_max_iterations": AGGRESSIVE_MAXITER,
        "fit_barrier_kj": best["barrier"],
        "fit_free_energy_kj": best["free_energy"],
        "target_barrier_kj": target_barrier_kj,
        "target_free_energy_kj": target_free_energy_kj,
        "barrier_residual_kj": best["barrier_residual_kj"],
        "free_energy_residual_kj": best["free_energy_residual_kj"],
        "coordinate_center_kj": best["coordinate_center_kj"],
        "fit_score": best["score"],
        "within_residual_tolerance": (
            abs(best["barrier_residual_kj"]) <= FIT_RESIDUAL_TOLERANCE_KJ
            and abs(best["free_energy_residual_kj"]) <= FIT_RESIDUAL_TOLERANCE_KJ
        ),
        "coordinate_centered": (
            best["coordinate_center_kj"] is not None
            and abs(best["coordinate_center_kj"]) <= COORDINATE_CENTER_TOLERANCE_KJ
        ),
        "candidate_count": len(candidates),
        "fit_settings": {
            "alpha_guess": float(auto_settings["alpha_guess"]),
            "H12_guess": float(auto_settings["H12_guess"]),
            "step": int(auto_settings["step"]),
            "bins": int(auto_settings["bins"]),
            "minpts": int(auto_settings["minpts"]),
            "residual_tolerance_kj": FIT_RESIDUAL_TOLERANCE_KJ,
            "coordinate_center_tolerance_kj": COORDINATE_CENTER_TOLERANCE_KJ,
        },
    }


def _reference_mapping_validation(config: ReactionConfig, reference_pair, parameter_summary):
    preset_evbs = {}
    for preset in config.resolution_presets:
        evb = build_evb(config, preset)
        evb.alpha = float(parameter_summary["alpha"])
        evb.H12 = float(parameter_summary["H12"])
        evb._parameter_source = parameter_summary.get("parameter_source", "fit")
        preset_evbs[preset["name"]] = evb

    success, error = map_pair_with_presets(config, preset_evbs, reference_pair)
    target_barrier_kj = config.barrier_kcal * KCAL_TO_KJ
    target_free_energy_kj = config.free_energy_kcal * KCAL_TO_KJ
    if success is None:
        return {
            "accepted": False,
            "error": error.__class__.__name__ if error else "UnknownError",
            "message": str(error) if error else "Unknown reference mapping failure",
        }

    barrier_residual = float(success["ts_kj"]) - target_barrier_kj
    free_energy_residual = float(success["dg_kj"]) - target_free_energy_kj
    accepted = (
        float(success["ts_kj"]) > 0.0
        and abs(barrier_residual) <= FIT_RESIDUAL_TOLERANCE_KJ
        and abs(free_energy_residual) <= FIT_RESIDUAL_TOLERANCE_KJ
    )
    return {
        "accepted": bool(accepted),
        "preset": success["preset"],
        "ts_kj": float(success["ts_kj"]),
        "dg_kj": float(success["dg_kj"]),
        "barrier_residual_kj": barrier_residual,
        "free_energy_residual_kj": free_energy_residual,
        "assignment_mode": success["assignment_mode"],
        "confidence": success["confidence"],
        "fallbacks_used": list(success.get("fallbacks_used", [])),
        "warnings": list(success.get("warnings", [])),
    }


def fit_reference_with_aggressive_backup(config: ReactionConfig, reference_pair):
    aggressive = aggressive_fit_reference_with_diagnostics(
        config, reference_pair, initial_result=None
    )
    aggressive_validation = _reference_mapping_validation(
        config, reference_pair, aggressive
    )
    if aggressive_validation["accepted"]:
        return {
            **aggressive,
            "reference_fit_validation": aggressive_validation,
            "fit_selection": "aggressive",
        }

    standard = fit_reference_with_diagnostics(config, reference_pair)
    standard_validation = _reference_mapping_validation(
        config, reference_pair, standard
    )
    if standard_validation["accepted"]:
        return {
            **standard,
            "parameter_source": "fit_reference_standard_backup",
            "reference_fit_validation": standard_validation,
            "rejected_aggressive_fit": {
                "alpha": aggressive["alpha"],
                "H12": aggressive["H12"],
                "parameter_source": aggressive.get("parameter_source"),
                "reference_fit_validation": aggressive_validation,
            },
            "fit_selection": "standard_backup",
        }

    # Neither fit remaps cleanly through the reporting presets. Keep the
    # aggressive result if its internal optimizer converged, but preserve both
    # validation records so the failure mode is visible in downstream JSON.
    if aggressive.get("converged"):
        return {
            **aggressive,
            "reference_fit_validation": aggressive_validation,
            "standard_backup_fit": {
                "alpha": standard["alpha"],
                "H12": standard["H12"],
                "parameter_source": standard.get("parameter_source"),
                "reference_fit_validation": standard_validation,
            },
            "fit_selection": "aggressive_unvalidated",
        }

    return {
        **standard,
        "parameter_source": "fit_reference_standard_backup_unvalidated",
        "reference_fit_validation": standard_validation,
        "rejected_aggressive_fit": {
            "alpha": aggressive["alpha"],
            "H12": aggressive["H12"],
            "parameter_source": aggressive.get("parameter_source"),
            "reference_fit_validation": aggressive_validation,
        },
        "fit_selection": "standard_backup_unvalidated",
    }


def fit_reference_standard_only(config: ReactionConfig, reference_pair):
    standard = fit_reference_with_diagnostics(config, reference_pair)
    standard_validation = _reference_mapping_validation(
        config, reference_pair, standard
    )
    return {
        **standard,
        "parameter_source": (
            "fit_reference_standard"
            if standard_validation["accepted"]
            else "fit_reference_standard_unvalidated"
        ),
        "reference_fit_validation": standard_validation,
        "fit_selection": (
            "standard"
            if standard_validation["accepted"]
            else "standard_unvalidated"
        ),
    }


def fit_barrierless_reference_with_diagnostics(
    config: ReactionConfig, reference_pair
):
    fit_evb = build_evb(config, config.resolution_presets[0])
    auto_settings = fit_evb.guess_fit_settings_from_ref(
        [reference_pair], apply=False
    )
    normalized_pair = fit_evb._normalize_pairs([reference_pair])[0]
    A_xvgs, B_xvgs = evb_utils.fetchA_B_xvgs(
        normalized_pair[0], normalized_pair[1]
    )

    target_free_energy_kj = config.free_energy_kcal * KCAL_TO_KJ
    barrier_limit_kj = (
        config.barrierless_barrier_tolerance_kcal * KCAL_TO_KJ
    )
    free_energy_limit_kj = (
        config.barrierless_free_energy_tolerance_kcal * KCAL_TO_KJ
    )
    alpha = float(auto_settings["alpha_guess"])
    h12 = max(5000.0, float(auto_settings["H12_guess"]) * 1.9)
    best = None
    evaluations = 0

    for _ in range(80):
        step_best = None
        for _ in range(120):
            ts_energy, dg_energy, state_detection = evb_utils.evbmap_nooutput(
                A_xvgs,
                B_xvgs,
                fit_evb.step,
                alpha,
                h12,
                fit_evb.dl,
                fit_evb.kb,
                fit_evb.temp,
                fit_evb.bins,
                fit_evb.minpts,
                fit_evb.reverse_isfs,
                return_metadata=True,
                allow_resolution_retry=False,
            )
            evaluations += 1
            free_energy_residual = dg_energy - target_free_energy_kj
            score = max(ts_energy - barrier_limit_kj, 0.0) + abs(
                free_energy_residual
            )
            candidate = {
                "alpha": float(alpha),
                "H12": float(h12),
                "barrier": float(ts_energy),
                "free_energy": float(dg_energy),
                "state_detection": state_detection,
                "score": float(score),
            }
            if best is None or score < best["score"]:
                best = candidate
            if step_best is None or score < step_best["score"]:
                step_best = candidate
            if (
                ts_energy <= barrier_limit_kj
                and abs(free_energy_residual) <= free_energy_limit_kj
            ):
                best = candidate
                break
            alpha -= free_energy_residual

        if (
            best["barrier"] <= barrier_limit_kj
            and abs(best["free_energy"] - target_free_energy_kj)
            <= free_energy_limit_kj
        ):
            break

        h12 += max(
            1.0,
            min(
                10.0,
                max(step_best["barrier"] - barrier_limit_kj, 0.0),
            ),
        )
        alpha = step_best["alpha"]
    else:
        raise EvbMappingError(
            "Barrierless reference fit did not satisfy the requested "
            f"{config.barrierless_barrier_tolerance_kcal:.2f} kcal/mol "
            "barrier and reaction-energy tolerances."
        )

    fitted_barrier_kj = best["barrier"]
    fitted_free_energy_kj = best["free_energy"]
    return {
        "alpha": best["alpha"],
        "H12": best["H12"],
        "parameter_source": "fit_barrierless_reference",
        "converged": True,
        "fit_converged": True,
        "fit_iterations": evaluations,
        "fit_max_iterations": 80 * 120,
        "fit_barrier_kj": fitted_barrier_kj,
        "fit_free_energy_kj": fitted_free_energy_kj,
        "target_barrier_kj": 0.0,
        "target_free_energy_kj": target_free_energy_kj,
        "barrier_residual_kj": fitted_barrier_kj,
        "free_energy_residual_kj": (
            fitted_free_energy_kj - target_free_energy_kj
        ),
        "fit_settings": {
            "alpha_guess": float(auto_settings["alpha_guess"]),
            "H12_guess": float(auto_settings["H12_guess"]),
            "step": int(fit_evb.step),
            "bins": int(fit_evb.bins),
            "minpts": int(fit_evb.minpts),
            "barrier_limit_kj": float(barrier_limit_kj),
            "free_energy_limit_kj": float(free_energy_limit_kj),
        },
    }


def prepare_parameters(root: Path, config: ReactionConfig):
    configure_boltzmann_helpers(config.use_stable_boltzmann)
    reference_pair = [str(root / config.reference_pair[0]), str(root / config.reference_pair[1])]
    overlap = summarize_diabatic_gap(reference_pair, 0.0, config.reverse_isfs)

    if config.parameter_mode == "fit_reference":
        return {
            **fit_reference_with_aggressive_backup(config, reference_pair),
            "reference_overlap": overlap,
        }

    if config.parameter_mode == "fit_reference_standard":
        return {
            **fit_reference_standard_only(config, reference_pair),
            "reference_overlap": overlap,
        }

    if config.parameter_mode == "fit_barrierless_reference":
        return {
            **fit_barrierless_reference_with_diagnostics(
                config, reference_pair
            ),
            "reference_overlap": overlap,
        }

    if config.parameter_mode == "fit_reference_or_adjusted":
        try:
            return {
                **fit_reference_with_aggressive_backup(config, reference_pair),
                "reference_overlap": overlap,
            }
        except Exception as exc:
            if config.manual_alpha_kj is None or config.manual_h12_kj is None:
                raise
            adjusted_overlap = summarize_diabatic_gap(
                reference_pair, config.manual_alpha_kj, config.reverse_isfs
            )
            return {
                "alpha": float(config.manual_alpha_kj),
                "H12": float(config.manual_h12_kj),
                "parameter_source": "adjusted_reference_after_fit_failure",
                "converged": True,
                "fit_converged": False,
                "fit_error": exc.__class__.__name__,
                "fit_failure_code": getattr(exc, "failure_code", None),
                "fit_message": str(exc),
                "reference_overlap": adjusted_overlap,
                "weak_overlap_frames": adjusted_overlap["weak_overlap_frames"],
                "overlap_warning": adjusted_overlap["warning"],
            }

    if config.parameter_mode == "manual_crossing_preserving":
        if config.manual_alpha_kj is None or config.manual_h12_kj is None:
            raise ValueError(f"{config.name} manual parameter mode requires alpha and H12")
        overlap = summarize_diabatic_gap(reference_pair, config.manual_alpha_kj, config.reverse_isfs)
        manual_summary = aggressive_fit_reference_with_diagnostics(
            config,
            reference_pair,
            initial_result={
                "alpha": config.manual_alpha_kj,
                "H12": config.manual_h12_kj,
                "fit_converged": True,
                "fit_iterations": 0,
            },
        )
        return {
            **manual_summary,
            "reference_overlap": overlap,
            "weak_overlap_frames": overlap["weak_overlap_frames"],
            "overlap_warning": overlap["warning"],
        }

    raise ValueError(f"Unknown parameter_mode: {config.parameter_mode}")


def save_profile_artifacts(
    output_stem: Path,
    profile,
    state_detection,
    alpha,
    h12,
    metadata,
):
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    x_kj = np.asarray(profile["x_kj"], dtype=float)
    x_centered_kj, x_origin_kj = _centered_x_values(profile)
    raw_kj = np.asarray(profile["raw_kj"], dtype=float)
    smooth_kj = np.asarray(profile["smooth_kj"], dtype=float)
    raw_norm_kj, smooth_norm_kj, y_reference_kj = _normalized_profile_values(profile)
    csv_path = output_stem.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "e1_minus_e2_kj_mol",
                "e1_minus_e2_centered_kj_mol",
                "raw_pmf_kj_mol",
                "smoothed_pmf_kj_mol",
                "raw_pmf_rs_normalized_kj_mol",
                "smoothed_pmf_rs_normalized_kj_mol",
                "raw_pmf_kcal_mol",
                "smoothed_pmf_kcal_mol",
                "raw_pmf_rs_normalized_kcal_mol",
                "smoothed_pmf_rs_normalized_kcal_mol",
                "rs_reference_y_kj_mol",
                "coordinate_origin_kj_mol",
            ]
        )
        writer.writerows(
            zip(
                x_kj,
                x_centered_kj,
                raw_kj,
                smooth_kj,
                raw_norm_kj,
                smooth_norm_kj,
                raw_kj / KCAL_TO_KJ,
                smooth_kj / KCAL_TO_KJ,
                raw_norm_kj / KCAL_TO_KJ,
                smooth_norm_kj / KCAL_TO_KJ,
                np.full_like(x_kj, y_reference_kj, dtype=float),
                np.full_like(x_kj, x_origin_kj, dtype=float),
            )
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        x_centered_kj,
        raw_norm_kj / KCAL_TO_KJ,
        color="#8ecae6",
        lw=1.2,
        alpha=0.75,
        label="raw PMF (RS-normalized)",
    )
    ax.plot(
        x_centered_kj,
        smooth_norm_kj / KCAL_TO_KJ,
        color="#023047",
        lw=2.2,
        label="smoothed PMF (RS-normalized)",
    )
    state_colors = {"RS": "#2a9d8f", "TS": "#e63946", "PS": "#f4a261"}
    for state_name in ("RS", "TS", "PS"):
        state = profile["states"][state_name]
        state_x = float(state["x_kj"]) - x_origin_kj
        state_y = float(np.interp(float(state["x_kj"]), x_kj, smooth_norm_kj)) / KCAL_TO_KJ
        ax.scatter(
            [state_x],
            [state_y],
            s=48,
            color=state_colors[state_name],
            zorder=5,
            label=state_name,
        )
    ax.set_xlabel("e1-e2 relative to TS (kJ/mol)")
    ax.set_ylabel("Delta G relative to RS (kcal/mol)")
    ax.set_title(metadata["label"])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    png_path = output_stem.with_suffix(".png")
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    bins_path = output_stem.with_name(
        output_stem.name + "_state_bins"
    ).with_suffix(".json")
    bins_payload = write_reference_state_bins(
        bins_path,
        alpha,
        h12,
        state_detection,
        metadata=metadata,
    )
    return {
        "pmf_csv": str(csv_path),
        "pmf_png": str(png_path),
        "state_bins_json": str(bins_path),
        "state_bins": bins_payload,
        "rs_reference_y_kj": y_reference_kj,
        "coordinate_origin_kj": x_origin_kj,
        "pmf_y_range_kcal": _pmf_y_range_kcal(profile),
    }


def run_reaction(root: Path, config: ReactionConfig, out_dir: Path):
    configure_boltzmann_helpers(config.use_stable_boltzmann)
    active_systems = discover_active_systems(root, config.collection_dir)
    sites = sorted(next(iter(active_systems.values())))
    parameter_summary = prepare_parameters(root, config)
    reaction_out = out_dir / config.name
    reaction_out.mkdir(parents=True, exist_ok=True)

    preset_evbs = {}
    for preset in config.resolution_presets:
        evb = build_evb(config, preset)
        evb.alpha = parameter_summary["alpha"]
        evb.H12 = parameter_summary["H12"]
        evb._parameter_source = parameter_summary["parameter_source"]
        preset_evbs[preset["name"]] = evb

    reference_pair = [str(root / config.reference_pair[0]), str(root / config.reference_pair[1])]
    reference_success, reference_error = map_pair_with_presets(config, preset_evbs, reference_pair)
    reference_profile = None
    reference_profile_error = None
    reference_mapping = None
    reference_state_bins = None
    reference_state_bins_path = None
    if reference_success is None:
        reference_profile_error = {
            "error": reference_error.__class__.__name__ if reference_error else "UnknownError",
            "failure_code": getattr(reference_error, "failure_code", None),
            "message": str(reference_error) if reference_error else "Unknown reference mapping failure",
        }
    else:
        reference_profile = {
            "label": "reference_free_linker",
            "preset": reference_success["preset"],
            "assignment_mode": reference_success["assignment_mode"],
            "confidence": reference_success["confidence"],
            "fallbacks_used": reference_success["fallbacks_used"],
            "warnings": reference_success["warnings"],
            **reference_success["profile"],
        }
        target_barrier_kj = config.barrier_kcal * KCAL_TO_KJ
        target_free_energy_kj = config.free_energy_kcal * KCAL_TO_KJ
        reference_mapping = {
            "preset": reference_success["preset"],
            "ts_kj": reference_success["ts_kj"],
            "dg_kj": reference_success["dg_kj"],
            "ts_kcal": reference_success["ts_kj"] / KCAL_TO_KJ,
            "dg_kcal": reference_success["dg_kj"] / KCAL_TO_KJ,
            "target_barrier_kj": target_barrier_kj,
            "target_free_energy_kj": target_free_energy_kj,
            "barrier_residual_kj": reference_success["ts_kj"] - target_barrier_kj,
            "free_energy_residual_kj": (
                reference_success["dg_kj"] - target_free_energy_kj
            ),
            "coordinate_center_kj": reference_success["coordinate_center_kj"],
            "coordinate_centered": reference_success["coordinate_centered"],
            "rs_is_local_basin": reference_success["rs_is_local_basin"],
            "rs_repaired": reference_success["rs_repaired"],
            "central_window_mapping": reference_success["central_window_mapping"],
            "central_window_kj": reference_success["central_window_kj"],
            "central_window_y_range_kcal": reference_success["central_window_y_range_kcal"],
            "pmf_y_range_kcal": reference_success["pmf_y_range_kcal"],
            "assignment_mode": reference_success["assignment_mode"],
            "confidence": reference_success["confidence"],
            "fallbacks_used": reference_success["fallbacks_used"],
            "warnings": reference_success["warnings"],
        }
        if parameter_summary.get("fit_converged") is True:
            reference_artifacts = save_profile_artifacts(
                reaction_out / "pmf" / "reference",
                reference_success["profile"],
                reference_success["state_detection"],
                parameter_summary["alpha"],
                parameter_summary["H12"],
                {
                    "reaction": config.name,
                    "case_type": "reference",
                    "preset": reference_success["preset"],
                    "assignment_mode": reference_success["assignment_mode"],
                    "confidence": reference_success["confidence"],
                    "label": f"{config.label}: reference PMF",
                },
            )
            reference_state_bins = reference_artifacts["state_bins"]
            reference_state_bins_path = Path(
                reference_artifacts["state_bins_json"]
            )
            reference_mapping.update(
                {
                    key: _relative_artifact_path(value, out_dir)
                    for key, value in reference_artifacts.items()
                    if key != "state_bins" and key.endswith(("_csv", "_png", "_json"))
                }
            )

    # --- Reference RS anchor for site-system state detection -----------------
    reference_rs_x_kj = None
    if reference_success is not None:
        ref_states = reference_success.get("profile", {}).get("states", {})
        if "RS" in ref_states:
            reference_rs_x_kj = float(ref_states["RS"]["x_kj"])

    results = []
    profiles = {}
    for system_name, site_names in active_systems.items():
        profiles[system_name] = {}
        for site in site_names:
            pair = [
                str(root / config.collection_dir / system_name / site / "potentialA"),
                str(root / config.collection_dir / system_name / site / "potentialB"),
            ]
            success, last_error = map_pair_with_presets(
                config, preset_evbs, pair,
                rs_hint_x_kj=reference_rs_x_kj,
            )
            row_base = {
                "reaction": config.name,
                "reaction_label": config.label,
                "system": system_name,
                "site": site,
            }
            if success is None:
                results.append(
                    {
                        **row_base,
                        "status": "failed",
                        "error": last_error.__class__.__name__ if last_error else "UnknownError",
                        "failure_code": getattr(last_error, "failure_code", None),
                        "message": str(last_error) if last_error else "Unknown mapping failure",
                    }
                )
                continue

            results.append(
                {
                    **row_base,
                    "status": "success",
                    "preset": success["preset"],
                    "ts_kj": success["ts_kj"],
                    "dg_kj": success["dg_kj"],
                    "ts_kcal": success["ts_kj"] / KCAL_TO_KJ,
                    "dg_kcal": success["dg_kj"] / KCAL_TO_KJ,
                    "coordinate_center_kj": success["coordinate_center_kj"],
                    "coordinate_centered": success["coordinate_centered"],
                    "rs_is_local_basin": success["rs_is_local_basin"],
                    "rs_repaired": success["rs_repaired"],
                    "central_window_mapping": success["central_window_mapping"],
                    "central_window_kj": success["central_window_kj"],
                    "central_window_y_range_kcal": success["central_window_y_range_kcal"],
                    "pmf_y_range_kcal": success["pmf_y_range_kcal"],
                    "pmf_y_range_caution": (
                        success["pmf_y_range_kcal"] is not None
                        and success["pmf_y_range_kcal"] > PMF_Y_RANGE_CAUTION_KCAL
                    ),
                    "assignment_mode": success["assignment_mode"],
                    "confidence": success["confidence"],
                    "fallbacks_used": success["fallbacks_used"],
                    "warnings": success["warnings"],
                }
            )
            artifacts = save_profile_artifacts(
                reaction_out / "pmf" / system_name / site,
                success["profile"],
                success["state_detection"],
                parameter_summary["alpha"],
                parameter_summary["H12"],
                {
                    "reaction": config.name,
                    "case_type": "system_site",
                    "system": system_name,
                    "site": site,
                    "preset": success["preset"],
                    "assignment_mode": success["assignment_mode"],
                    "confidence": success["confidence"],
                    "label": f"{config.label}: {system_name}/{site}",
                },
            )
            results[-1].update(
                {
                    key: _relative_artifact_path(value, out_dir)
                    for key, value in artifacts.items()
                    if key != "state_bins" and key.endswith(("_csv", "_png", "_json"))
                }
            )
            profiles[system_name][site] = success["profile"]

    summary = {
        "config": asdict(config),
        "fit": {k: v for k, v in parameter_summary.items() if k != "reference_overlap"},
        "reference_overlap": parameter_summary["reference_overlap"],
        "active_systems": active_systems,
        "sites": sites,
        "reference_mapping": reference_mapping,
        "reference_profile": reference_profile,
        "reference_profile_error": reference_profile_error,
        "reference_state_bins": reference_state_bins,
        "reference_state_bins_path": (
            str(reference_state_bins_path.relative_to(out_dir))
            if reference_state_bins_path is not None else None
        ),
        "results": results,
        "profiles": profiles,
    }
    summary["figure_paths"] = save_reaction_figures(summary, reaction_out)
    return summary


def _relative_artifact_path(path, output_dir: Path) -> str:
    """Return an output artifact path relative to the run directory."""
    return str(Path(path).resolve().relative_to(Path(output_dir).resolve()))


def _successful_row_lookup(rows):
    return {(row["system"], row["site"]): row for row in rows if row["status"] == "success"}


def _state_plot_limits(profiles, padding_fraction):
    state_x = []
    bin_widths = []
    for profile in profiles:
        if profile is None:
            continue
        states = profile.get("states", {})
        origin = _coordinate_origin_kj(profile)
        for state_name in ("RS", "PS"):
            state = states.get(state_name)
            if state is not None and np.isfinite(state.get("x_kj", np.nan)):
                state_x.append(float(state["x_kj"]) - origin)
        x, _ = _centered_x_values(profile)
        if x.size > 1:
            finite_widths = np.abs(np.diff(x))
            finite_widths = finite_widths[np.isfinite(finite_widths) & (finite_widths > 0)]
            if finite_widths.size:
                bin_widths.append(float(np.median(finite_widths)))

    if len(state_x) < 2:
        return None
    lower = min(state_x)
    upper = max(state_x)
    span = upper - lower
    typical_bin_width = float(np.median(bin_widths)) if bin_widths else 1.0
    minimum_padding = max(typical_bin_width, 1.0)
    # Large r4 state envelopes can make fractional padding show excessive basin
    # head/tail in comparison summaries; keep padding local to the state bins.
    maximum_padding = max(typical_bin_width * 1.2, minimum_padding)
    padding = max(min(span * float(padding_fraction), maximum_padding), minimum_padding)
    return lower - padding, upper + padding


def _apply_state_plot_limits(ax, profiles, padding_fraction):
    if padding_fraction is None:
        return
    limits = _state_plot_limits(profiles, padding_fraction)
    if limits is not None:
        ax.set_xlim(*limits)
        visible_y = []
        for line in ax.lines:
            x = np.asarray(line.get_xdata(), dtype=float)
            y = np.asarray(line.get_ydata(), dtype=float)
            visible = (
                np.isfinite(x)
                & np.isfinite(y)
                & (x >= limits[0])
                & (x <= limits[1])
            )
            visible_y.extend(y[visible].tolist())
        if visible_y:
            lower = min(visible_y)
            upper = max(visible_y)
            span = upper - lower
            padding = max(span * 0.08, 0.5)
            ax.set_ylim(lower - padding, upper + padding)


def save_reference_overlap_figure(summary, out_dir):
    stats = summary["reference_overlap"]["frame_stats"]
    lambdas = np.linspace(0, 1, len(stats))
    mean_gap = np.array([row["mean_kj"] for row in stats])
    p05 = np.array([row["p05_kj"] for row in stats])
    p95 = np.array([row["p95_kj"] for row in stats])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(lambdas, p05, p95, color="#8ecae6", alpha=0.35, label="5-95% sampled gap range")
    ax.plot(lambdas, mean_gap, color="#023047", lw=2.2, label="mean diabatic gap")
    ax.axhline(0.0, color="black", lw=1.2, ls="--", label="diabatic crossing")
    ax.set_xlabel("lambda")
    ax.set_ylabel("E_A - E_B - alpha (kJ/mol)")
    ax.set_title(f"{summary['config']['label']} reference PES overlap")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "reference_pes_overlap.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path.name


def save_reaction_figures(summary, out_dir):
    rows = summary["results"]
    profiles = summary["profiles"]
    sites = summary["sites"]
    system_order = list(summary["active_systems"])
    reference_profile = summary.get("reference_profile")
    padding_fraction = summary["config"].get("plot_state_padding_fraction")
    row_lookup = _successful_row_lookup(rows)
    system_colors = dict(zip(system_order, ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]))
    site_colors = plt.cm.tab10(np.linspace(0, 1, len(sites)))
    figure_paths = {"reference_overlap": save_reference_overlap_figure(summary, out_dir)}

    ncols = 2
    nrows = int(np.ceil(len(sites) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, max(5, 4.2 * nrows)), squeeze=False)
    for ax, site in zip(axes.ravel(), sites):
        plotted_profiles = []
        for system_name in system_order:
            row = row_lookup.get((system_name, site))
            prof = profiles.get(system_name, {}).get(site)
            if row is None or prof is None:
                continue
            plotted_profiles.append(prof)
            _, smooth_norm, _ = _normalized_profile_values(prof)
            x_centered, _ = _centered_x_values(prof)
            ax.plot(
                x_centered,
                smooth_norm / KCAL_TO_KJ,
                label=f"{system_name} ({row['preset']}, {row['assignment_mode']})",
                lw=2.0,
                color=system_colors.get(system_name),
            )
        if reference_profile is not None:
            plotted_profiles.append(reference_profile)
            _, reference_smooth_norm, _ = _normalized_profile_values(reference_profile)
            reference_x_centered, _ = _centered_x_values(reference_profile)
            ax.plot(
                reference_x_centered,
                reference_smooth_norm / KCAL_TO_KJ,
                label=f"reference ({reference_profile['assignment_mode']})",
                lw=2.2,
                ls="--",
                color="black",
                alpha=0.85,
            )
        ax.set_title(site)
        ax.set_xlabel("e1-e2 relative to TS (kJ/mol)")
        ax.set_ylabel("Delta G relative to RS (kcal/mol)")
        _apply_state_plot_limits(ax, plotted_profiles, padding_fraction)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    for ax in axes.ravel()[len(sites):]:
        ax.axis("off")
    fig.suptitle(f"{summary['config']['label']}: site-wise EVB comparison", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    path = out_dir / "site_comparison.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    figure_paths["site_comparison"] = path.name

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), squeeze=False)
    for ax, system_name in zip(axes.ravel(), system_order):
        plotted_profiles = []
        for color, site in zip(site_colors, sites):
            row = row_lookup.get((system_name, site))
            prof = profiles.get(system_name, {}).get(site)
            if row is None or prof is None:
                continue
            plotted_profiles.append(prof)
            _, smooth_norm, _ = _normalized_profile_values(prof)
            x_centered, _ = _centered_x_values(prof)
            ax.plot(
                x_centered,
                smooth_norm / KCAL_TO_KJ,
                lw=1.7,
                color=color,
                label=f"{site} ({row['assignment_mode']})",
            )
        if reference_profile is not None:
            plotted_profiles.append(reference_profile)
            _, reference_smooth_norm, _ = _normalized_profile_values(reference_profile)
            reference_x_centered, _ = _centered_x_values(reference_profile)
            ax.plot(
                reference_x_centered,
                reference_smooth_norm / KCAL_TO_KJ,
                lw=2.2,
                ls="--",
                color="black",
                alpha=0.85,
                label=f"reference ({reference_profile['assignment_mode']})",
            )
        ax.set_title(system_name)
        ax.set_xlabel("e1-e2 relative to TS (kJ/mol)")
        ax.set_ylabel("Delta G relative to RS (kcal/mol)")
        _apply_state_plot_limits(ax, plotted_profiles, padding_fraction)
        ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(f"{summary['config']['label']}: system-wise EVB comparison", fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    path = out_dir / "system_comparison.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    figure_paths["system_comparison"] = path.name

    ts = np.array(
        [[row_lookup[(system, site)]["ts_kcal"] if (system, site) in row_lookup else np.nan for site in sites] for system in system_order]
    )
    dg = np.array(
        [[row_lookup[(system, site)]["dg_kcal"] if (system, site) in row_lookup else np.nan for site in sites] for system in system_order]
    )
    path = out_dir / "metric_heatmaps_kcal.png"
    plot_metric_heatmap(ts, dg, sites, system_order, path)
    figure_paths["metric_heatmaps_kcal"] = path.name
    return figure_paths


def plot_metric_heatmap(ts, dg, sites, system_order, path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, arr, title, cmap, text_color in [
        (axes[0], ts, "TS barrier (kcal/mol)", "magma", "white"),
        (axes[1], dg, "Reaction free energy dG (kcal/mol)", "viridis", "black"),
    ]:
        im = ax.imshow(arr, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(sites)), sites, rotation=45, ha="right")
        ax.set_yticks(range(len(system_order)), system_order)
        ax.set_title(title)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                label = "n/a" if np.isnan(arr[i, j]) else f"{arr[i, j]:.1f}"
                ax.text(j, i, label, ha="center", va="center", color=text_color, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_all_reactions(root: Path, configs: list[ReactionConfig], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for config in configs:
        try:
            summaries.append(run_reaction(root, config, out_dir))
        except Exception as exc:
            summaries.append(
                {
                    "config": asdict(config),
                    "fit": {
                        "parameter_source": "failed",
                        "fit_converged": False,
                        "fit_error": exc.__class__.__name__,
                        "fit_message": str(exc),
                    },
                    "reference_overlap": None,
                    "active_systems": {},
                    "sites": [],
                    "reference_mapping": None,
                    "reference_profile": None,
                    "reference_profile_error": {
                        "error": exc.__class__.__name__,
                        "message": str(exc),
                    },
                    "results": [
                        {
                            "reaction": config.name,
                            "reaction_label": config.label,
                            "system": None,
                            "site": None,
                            "status": "failed",
                            "error": exc.__class__.__name__,
                            "message": str(exc),
                        }
                    ],
                    "profiles": {},
                    "figure_paths": {},
                }
            )
    combined = {
        "reactions": [summary["config"] for summary in summaries],
        "summaries": summaries,
        "results": [row for summary in summaries for row in summary["results"]],
    }
    (out_dir / "four_reaction_mof_environment_effect_summary.json").write_text(
        json.dumps(combined, indent=2)
    )
    write_results_csv(combined["results"], out_dir / "four_reaction_mof_environment_effect_summary.csv")
    return combined


def write_results_csv(rows, path):
    fieldnames = [
        "reaction",
        "reaction_label",
        "system",
        "site",
        "status",
        "preset",
        "ts_kj",
        "dg_kj",
        "ts_kcal",
        "dg_kcal",
        "coordinate_center_kj",
        "coordinate_centered",
        "rs_is_local_basin",
        "rs_repaired",
        "central_window_mapping",
        "central_window_kj",
        "central_window_y_range_kcal",
        "pmf_y_range_kcal",
        "pmf_y_range_caution",
        "assignment_mode",
        "confidence",
        "fallbacks_used",
        "warnings",
        "error",
        "failure_code",
        "message",
        "pmf_csv",
        "pmf_png",
        "state_bins_json",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            serializable["fallbacks_used"] = "|".join(row.get("fallbacks_used", []))
            serializable["warnings"] = "|".join(row.get("warnings", []))
            writer.writerow(serializable)


def default_reaction_configs():
    baseline_state = dict(
        state_min_prominence=1.0,
        state_min_basin_width=2.0,
        state_edge_exclusion_bins=2,
    )
    relaxed_state = dict(
        state_min_prominence=0.0,
        state_min_basin_width=0.0,
        state_edge_exclusion_bins=2,
    )
    return [
        ReactionConfig(
            name="reaction1_co_adsorption",
            label="Reaction 1: CO adsorption",
            collection_dir="1.adsorption_recalculation_collection",
            reference_pair=["1.adsorption_ref_sys/potentialA", "1.adsorption_ref_sys/potentialB"],
            barrier_kcal=17.3,
            free_energy_kcal=1.8,
            reverse_isfs=True,
            resolution_presets=[
                dict(name="baseline", step=1, bins=50, minpts=1, **baseline_state),
                dict(name="relaxed_topology", step=1, bins=50, minpts=1, **relaxed_state),
            ],
            parameter_mode="fit_reference",
        ),
        ReactionConfig(
            name="reaction2_hydrogenation",
            label="Reaction 2: hydrogenation",
            collection_dir="2.hydrogenation_recalculation_collection",
            reference_pair=["2.hydrogenation_ref_sys/potentialA", "2.hydrogenation_ref_sys/potentialB"],
            barrier_kcal=9.4,
            free_energy_kcal=1.7,
            reverse_isfs=False,
            resolution_presets=[
                dict(name="baseline", step=2, bins=40, minpts=100, **baseline_state),
                dict(name="relaxed_topology", step=2, bins=40, minpts=100, **relaxed_state),
            ],
            parameter_mode="fit_reference_or_adjusted",
            manual_alpha_kj=-57.45773,
            manual_h12_kj=432.50274,
        ),
        ReactionConfig(
            name="reaction3_c_o_cleavage",
            label="Reaction 3: C-O cleavage",
            collection_dir="3.COcleavage_recalculation_collection",
            reference_pair=["3.COcleavage_ref_sys/potentialA", "3.COcleavage_ref_sys/potentialB"],
            barrier_kcal=6.4,
            free_energy_kcal=-15.0,
            reverse_isfs=False,
            resolution_presets=[
                dict(name="baseline", step=1, bins=60, minpts=15, **baseline_state),
                dict(name="relaxed_topology", step=1, bins=60, minpts=15, **relaxed_state),
            ],
            parameter_mode="manual_crossing_preserving",
            manual_alpha_kj=0.0,
            manual_h12_kj=20.0,
            use_stable_boltzmann=True,
        ),
        ReactionConfig(
            name="reaction4_co_desorption",
            label="Reaction 4: CO desorption",
            collection_dir="4.COdesorptionrecalculation_collection",
            reference_pair=["4.COdesorption_ref_sys/potentialA", "4.COdesorption_ref_sys/potentialB"],
            barrier_kcal=9.4,
            free_energy_kcal=1.7,
            reverse_isfs=False,
            resolution_presets=[
                dict(name="baseline", step=1, bins=40, minpts=30, **baseline_state),
                dict(name="relaxed_topology", step=1, bins=40, minpts=30, **relaxed_state),
            ],
            parameter_mode="fit_reference",
            use_stable_boltzmann=True,
        ),
    ]
