import glob
import math as m
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from .evb_plot import plot_mapped_pmf_diagnostic, secondary_smooth_data


class EvbMappingError(RuntimeError):
    """Raised when EVB profile construction cannot identify the required states."""

    def __init__(self, message, *, failure_code=None, diagnostics=None):
        super().__init__(message)
        self.failure_code = failure_code
        self.diagnostics = {} if diagnostics is None else diagnostics


class EvbStateAssignmentError(EvbMappingError):
    """Raised when RS/TS/PS assignment fails physical topology validation."""


class EvbInputError(EvbMappingError):
    """Raised when EVB input files are missing or malformed."""


class EvbNumericalError(EvbMappingError):
    """Raised when EVB numerical evaluation becomes unstable."""


class EvbNoTrustworthyReactantStateError(EvbStateAssignmentError):
    """Raised when no trustworthy reactant basin can be identified."""


class EvbNoTrustworthyProductStateError(EvbStateAssignmentError):
    """Raised when no trustworthy product basin can be identified."""


class EvbNoValidTransitionStateError(EvbStateAssignmentError):
    """Raised when no barrier can be validated between trustworthy basins."""


class EvbAmbiguousTransitionStateError(EvbStateAssignmentError):
    """Raised when multiple barriers compete with similar support."""


class EvbInsufficientSamplingError(EvbStateAssignmentError):
    """Raised when PMF support is too weak for robust state assignment."""


STATE_DETECTION_CONFIG = {
    "enabled": True,
    "gaussian_sigma": 1.0,
    "savgol_window": 9,
    "savgol_polyorder": 3,
    "min_prominence": 1.0,
    "min_basin_width": 2.0,
    "edge_exclusion_bins": 2,
    "auto_retry_resolution": True,
    "assignment_policy": "robust",
}
PROFILE_SMOOTHING_CONFIG = STATE_DETECTION_CONFIG


def set_state_detection_config(enabled=True,
                               gaussian_sigma=1.0,
                               savgol_window=9,
                               savgol_polyorder=3,
                               min_prominence=1.0,
                               min_basin_width=2.0,
                               edge_exclusion_bins=2,
                               auto_retry_resolution=True,
                               assignment_policy="robust"):
    if assignment_policy not in {"robust", "strict"}:
        raise ValueError(
            "assignment_policy must be either 'robust' or 'strict'.")
    STATE_DETECTION_CONFIG.update({
        "enabled": bool(enabled),
        "gaussian_sigma": float(gaussian_sigma),
        "savgol_window": int(savgol_window),
        "savgol_polyorder": int(savgol_polyorder),
        "min_prominence": float(min_prominence),
        "min_basin_width": float(min_basin_width),
        "edge_exclusion_bins": int(edge_exclusion_bins),
        "auto_retry_resolution": bool(auto_retry_resolution),
        "assignment_policy": str(assignment_policy),
    })


def get_state_detection_config():
    return dict(STATE_DETECTION_CONFIG)


def set_profile_smoothing_config(enabled=True,
                                 gaussian_sigma=1.0,
                                 savgol_window=9,
                                 savgol_polyorder=3,
                                 min_prominence=1.0,
                                 min_basin_width=2.0,
                                 edge_exclusion_bins=2,
                                 auto_retry_resolution=True,
                                 assignment_policy="robust"):
    set_state_detection_config(enabled=enabled,
                               gaussian_sigma=gaussian_sigma,
                               savgol_window=savgol_window,
                               savgol_polyorder=savgol_polyorder,
                               min_prominence=min_prominence,
                               min_basin_width=min_basin_width,
                               edge_exclusion_bins=edge_exclusion_bins,
                               auto_retry_resolution=auto_retry_resolution,
                               assignment_policy=assignment_policy)


def get_profile_smoothing_config():
    return get_state_detection_config()


def summarize_reference_potentials(A_xvgs, B_xvgs, reverse_isfs=False):
    _validate_xvg_inputs(A_xvgs, B_xvgs, "reference potential summary")
    raw_points_per_frame = _load_xvg_array(A_xvgs[0], 1).shape[0]
    feps, sampled_points = read_data(A_xvgs,
                                     B_xvgs,
                                     alpha=0.0,
                                     hij=20.0,
                                     dl=0.05,
                                     step=1,
                                     reverse_isfs=reverse_isfs)
    gaps = feps[:, 2, :].reshape(-1)
    q10, q50, q90 = np.quantile(gaps, [0.1, 0.5, 0.9])
    return {
        "frames": len(A_xvgs),
        "raw_points_per_frame": int(raw_points_per_frame),
        "sampled_points_per_frame": int(sampled_points),
        "gap_mean": float(np.mean(gaps)),
        "gap_std": float(np.std(gaps)),
        "gap_q10": float(q10),
        "gap_q50": float(q50),
        "gap_q90": float(q90),
        "gap_span_q10_q90": float(q90 - q10),
        "gap_center_q10_q90": float(0.5 * (q10 + q90)),
    }


def fetch_unrelated_energy(arr_potentialA, step):
    if step <= 0:
        raise EvbInputError("step must be a positive integer.")
    return arr_potentialA[::step, :]


def _parse_lambda_value(path):
    match = re.match(r"(\d+)", os.path.basename(path))
    if not match:
        raise EvbInputError(
            f"Could not parse lambda value from XVG filename: {path}")
    return int(match.group(1))


def _load_xvg_array(path, step):
    try:
        data = np.loadtxt(path,
                          comments="@",
                          delimiter=None,
                          usecols=(0, 1),
                          skiprows=14)
    except OSError as exc:
        raise EvbInputError(f"Could not read XVG file: {path}") from exc
    except ValueError as exc:
        raise EvbInputError(f"Malformed XVG file: {path}") from exc

    data = np.atleast_2d(data)
    if data.size == 0 or data.shape[1] != 2:
        raise EvbInputError(f"XVG file does not contain two numeric columns: {path}")
    return fetch_unrelated_energy(data, step)


def _validate_xvg_inputs(A_xvgs, B_xvgs, context):
    if not A_xvgs or not B_xvgs:
        raise EvbInputError(
            f"Missing XVG inputs for {context}: found {len(A_xvgs)} A files and "
            f"{len(B_xvgs)} B files. Check that both potential folders exist and "
            "contain matching `.xvg` files.")
    if len(A_xvgs) != len(B_xvgs):
        raise EvbInputError(
            f"Mismatched XVG counts for {context}: found {len(A_xvgs)} A files and "
            f"{len(B_xvgs)} B files.")


def _safe_positive(value, context):
    if not np.isfinite(value) or value <= 0:
        raise EvbNumericalError(f"{context} produced a non-positive value: {value}")
    return value


def _exp_average(values, average, kT, context):
    try:
        with np.errstate(over="raise", invalid="raise"):
            exp_values = np.exp(-(values - average) / kT)
    except FloatingPointError as exc:
        raise EvbNumericalError(
            f"{context} overflowed while evaluating the Boltzmann factor.") from exc
    return _safe_positive(float(np.mean(exp_values)), context)


def read_data(A_xvgs, B_xvgs, alpha, hij, dl, step=2, reverse_isfs=False):
    _validate_xvg_inputs(A_xvgs, B_xvgs, "EVB mapping")

    frames = len(A_xvgs)
    first_frame = _load_xvg_array(A_xvgs[0], step)
    N = first_frame.shape[0]
    feps = np.zeros(shape=(frames, 8, N))

    for A_path, B_path in zip(A_xvgs, B_xvgs):
        lambda_a = _parse_lambda_value(A_path)
        lambda_b = _parse_lambda_value(B_path)
        if lambda_a != lambda_b:
            raise EvbInputError(
                f"Mismatched lambda files: {os.path.basename(A_path)} vs "
                f"{os.path.basename(B_path)}")

        df_potentialA = _load_xvg_array(A_path, step)
        df_potentialB = _load_xvg_array(B_path, step)
        if df_potentialA.shape[0] != N or df_potentialB.shape[0] != N:
            raise EvbInputError(
                "All XVG files must contain the same number of sampled points after "
                f"applying step={step}.")

        lambda_index = frames - 1 - lambda_a if reverse_isfs else lambda_a
        if lambda_index < 0 or lambda_index >= frames:
            raise EvbInputError(
                f"Lambda index {lambda_index} is outside the expected range 0-{frames - 1}."
            )

        feps[lambda_index][0][:] = df_potentialA[:, 1]
        feps[lambda_index][1][:] = df_potentialB[:, 1]

    for frame in range(frames):
        for point in range(N):
            feps[frame][2][point] = feps[frame][0][point] - feps[frame][1][
                point] - alpha
            feps[frame][3][point] = ((1 - frame * dl) * feps[frame][0][point] +
                                     (frame * dl) *
                                     (feps[frame][1][point] + alpha))
            feps[frame][4][point] = (
                0.5 * (feps[frame][0][point] + feps[frame][1][point] + alpha)
                - 0.5 * m.sqrt((feps[frame][2][point])**2 + 4 * hij**2))
            feps[frame][5][point] = feps[frame][0][point] - feps[frame][3][point]
            feps[frame][6][point] = (
                feps[frame][1][point] + alpha - feps[frame][3][point])
            feps[frame][7][point] = feps[frame][4][point] - feps[frame][3][point]

    return feps, N


def gofl(feps, frames, N, kT, dl):
    dgf = []
    dgb = []
    dg_lambda = [0.0]

    for frame in range(frames - 1):
        try:
            with np.errstate(over="raise", invalid="raise"):
                forward_sum = np.exp(feps[frame][2] * dl / kT).sum()
                backward_sum = np.exp(-feps[frame + 1][2] * dl / kT).sum()
        except FloatingPointError as exc:
            raise EvbNumericalError(
                f"Numerical overflow while computing dG(lambda) at frames {frame} "
                f"and {frame + 1}.") from exc

        dgf.append(-kT * m.log(_safe_positive(forward_sum, "Forward Zwanzig sum")) +
                   kT * m.log(N))
        dgb.append(kT * m.log(_safe_positive(backward_sum, "Backward Zwanzig sum"))
                   - kT * m.log(N))
        dg_lambda.append(dg_lambda[-1] + 0.5 * (dgf[frame] + dgb[frame]))

    lc = [dl * frame for frame in range(frames)]
    lc.reverse()
    return dg_lambda, lc


def gaps(feps, bins):
    min_gap = min(feps[frame][2].min() for frame in range(feps.shape[0]))
    max_gap = max(feps[frame][2].max() for frame in range(feps.shape[0]))
    dgap = (max_gap - min_gap) / bins
    x = [min_gap]
    for _ in range(bins):
        x.append(x[-1] + dgap)
    return x


def uofx(feps, bins, x, frames):
    free_en = [[[[], [], []] for _ in range(frames)] for _ in range(bins)]

    for frame in range(frames):
        for point_index, gap in enumerate(feps[frame][2]):
            for nbin in range(bins - 1):
                if nbin == 0:
                    in_bin = x[nbin] <= gap <= x[nbin + 1]
                else:
                    in_bin = x[nbin] < gap <= x[nbin + 1] + 1
                if not in_bin:
                    continue
                free_en[nbin][frame][0].append(feps[frame][5][point_index])
                free_en[nbin][frame][1].append(feps[frame][6][point_index])
                free_en[nbin][frame][2].append(feps[frame][7][point_index])
                break

    return free_en


def gofx(dg_lambda, free_en, bins, frames, minpts, kT, x, N):
    del N  # kept for signature compatibility with the original helper flow

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

            dga.append(dg_lambda[frame] -
                       kT * m.log(_exp_average(e1_values, avv1, kT,
                                               f"dGa bin {nbin} frame {frame}"))
                       + avv1)
            dgb.append(dg_lambda[frame] -
                       kT * m.log(_exp_average(e2_values, avv2, kT,
                                               f"dGb bin {nbin} frame {frame}"))
                       + avv2)
            dgg.append(dg_lambda[frame] -
                       kT * m.log(_exp_average(gg_values, avvgg, kT,
                                               f"dGg bin {nbin} frame {frame}"))
                       + avvgg)
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


def _output_stem(out):
    parent = os.path.dirname(out)
    if parent and os.path.isdir(parent):
        return out
    artifact_dir = os.environ.get("EVB_PROFILE_ARTIFACT_DIR")
    if artifact_dir:
        out = out.replace(os.getcwd() + os.sep, "")
        safe_name = out.replace("\\", "_").replace("/", "_").replace(" ", "")
        os.makedirs(artifact_dir, exist_ok=True)
        return os.path.join(artifact_dir, safe_name)
    safe_name = out.replace("\\", "_").replace("/", "_").replace(" ", "")
    return os.path.join(os.getcwd(), safe_name)


def _smoothed_profile_for_detection(gg, config=None):
    gg = np.asarray(gg, dtype=float)
    config = get_state_detection_config() if config is None else dict(config)
    if (not config["enabled"]) or gg.size < 5:
        return gg.copy()
    return secondary_smooth_data(
        gg,
        gaussian_sigma=config["gaussian_sigma"],
        savgol_window=config["savgol_window"],
        savgol_polyorder=config["savgol_polyorder"],
    )


def _profile_sampling_arrays(free_en, minpts):
    max_population = []
    total_population = []
    frame_support = []

    if free_en is None:
        return {
            "max_population": np.asarray([], dtype=float),
            "total_population": np.asarray([], dtype=float),
            "frame_support": np.asarray([], dtype=float),
        }

    for bin_frames in free_en:
        counts = [len(frame[0]) for frame in bin_frames]
        if max(counts, default=0) < minpts:
            continue
        max_population.append(max(counts, default=0))
        total_population.append(sum(counts))
        frame_support.append(sum(1 for count in counts if count >= minpts))

    return {
        "max_population": np.asarray(max_population, dtype=float),
        "total_population": np.asarray(total_population, dtype=float),
        "frame_support": np.asarray(frame_support, dtype=float),
    }


def _window_bounds(index, size, half_window):
    start = max(0, index - half_window)
    stop = min(size, index + half_window + 1)
    return start, stop


def _reason(code, detail):
    return {"code": code, "detail": detail}


def _candidate_payload(kind,
                       index,
                       x_values,
                       raw_y,
                       smoothed_y,
                       prominence,
                       width,
                       sampling,
                       config,
                       source="interior"):
    half_window = max(1, int(np.ceil(float(config["min_basin_width"]) / 2.0)))
    lower, upper = _window_bounds(index, len(x_values), half_window)
    edge_bins = max(0, int(config["edge_exclusion_bins"]))
    edge_side = None
    if index <= edge_bins:
        edge_side = "left"
    elif index >= len(x_values) - 1 - edge_bins:
        edge_side = "right"

    return {
        "id": f"{kind}:{index}:{source}",
        "kind": kind,
        "source": source,
        "index": int(index),
        "x": float(x_values[index]),
        "y": float(smoothed_y[index]),
        "raw_y": float(raw_y[index]),
        "prominence": float(prominence),
        "width": float(width),
        "population": int(sampling["max_population"][index]) if sampling[
            "max_population"].size else 0,
        "local_population": int(np.sum(sampling["max_population"][lower:upper]))
        if sampling["max_population"].size else 0,
        "frame_support": int(sampling["frame_support"][index]) if sampling[
            "frame_support"].size else 0,
        "local_frame_support": int(np.sum(sampling["frame_support"][lower:upper]))
        if sampling["frame_support"].size else 0,
        "edge_side": edge_side,
        "edge_relief": 0.0,
        "reasons": [],
    }


def _extract_candidates(kind, x_values, raw_y, smoothed_y, sampling, config):
    prominence_floor = max(0.05, float(config["min_prominence"]) * 0.35)
    width_floor = max(1.0, float(config["min_basin_width"]) * 0.5)
    signal = -smoothed_y if kind == "minimum" else smoothed_y
    indices, props = find_peaks(signal,
                                prominence=prominence_floor,
                                width=width_floor)
    candidates = [
        _candidate_payload(kind,
                           int(index),
                           x_values,
                           raw_y,
                           smoothed_y,
                           props["prominences"][offset],
                           props["widths"][offset],
                           sampling,
                           config)
        for offset, index in enumerate(indices)
    ]

    if kind == "minimum" and len(x_values) >= 3:
        edge_window = max(3, int(config["edge_exclusion_bins"]) + 3)
        left_interior = smoothed_y[1:min(len(smoothed_y), edge_window)]
        if left_interior.size and smoothed_y[0] <= smoothed_y[1]:
            candidate = _candidate_payload(kind,
                                           0,
                                           x_values,
                                           raw_y,
                                           smoothed_y,
                                           np.max(left_interior) - smoothed_y[0],
                                           left_interior.size,
                                           sampling,
                                           config,
                                           source="endpoint")
            candidate["edge_relief"] = float(np.max(left_interior) - smoothed_y[0])
            candidates.append(candidate)

        right_interior = smoothed_y[max(0, len(smoothed_y) - edge_window):-1]
        if right_interior.size and smoothed_y[-1] <= smoothed_y[-2]:
            candidate = _candidate_payload(kind,
                                           len(x_values) - 1,
                                           x_values,
                                           raw_y,
                                           smoothed_y,
                                           np.max(right_interior) - smoothed_y[-1],
                                           right_interior.size,
                                           sampling,
                                           config,
                                           source="endpoint")
            candidate["edge_relief"] = float(np.max(right_interior) - smoothed_y[-1])
            candidates.append(candidate)

    return candidates


def _edge_relief(candidate, smoothed_y, config):
    edge_bins = max(1, int(config["edge_exclusion_bins"]))
    if candidate["edge_side"] == "left":
        interior = smoothed_y[candidate["index"] + 1:min(len(smoothed_y),
                                                         candidate["index"] +
                                                         edge_bins + 3)]
    elif candidate["edge_side"] == "right":
        interior = smoothed_y[max(0, candidate["index"] - edge_bins - 2):
                              candidate["index"]]
    else:
        return 0.0

    if interior.size == 0:
        return 0.0
    if candidate["kind"] == "minimum":
        return float(np.max(interior) - candidate["y"])
    return float(candidate["y"] - np.min(interior))


def _annotate_candidate_rejections(candidates, smoothed_y, minpts, frames, config):
    required_frames = 1 if frames <= 1 else 2
    local_population_floor = max(minpts + 1, int(np.ceil(1.5 * minpts)))
    edge_frame_floor = min(frames, required_frames + 1)

    for candidate in candidates:
        if candidate["prominence"] < float(config["min_prominence"]):
            candidate["reasons"].append(
                _reason("prominence",
                        "Prominence is below the minimum basin/barrier threshold."))
        if (candidate["source"] != "endpoint" and
                candidate["width"] < float(config["min_basin_width"])):
            candidate["reasons"].append(
                _reason("width", "Candidate basin/barrier is narrower than the minimum width."))
        if candidate["population"] < int(minpts):
            candidate["reasons"].append(
                _reason("sampling", "Candidate sits in a weakly populated bin."))
        if candidate["local_population"] < local_population_floor:
            candidate["reasons"].append(
                _reason("sampling", "Candidate lacks enough nearby populated bins."))
        if frames > 1 and candidate["frame_support"] < required_frames:
            candidate["reasons"].append(
                _reason("sampling", "Candidate is not supported by multiple lambda windows."))
        if frames > 1 and candidate["local_frame_support"] < required_frames:
            candidate["reasons"].append(
                _reason("sampling",
                        "Nearby bins do not show enough multi-window support."))

        if candidate["edge_side"] is not None:
            candidate["edge_relief"] = _edge_relief(candidate, smoothed_y, config)
            if candidate["kind"] == "maximum":
                candidate["reasons"].append(
                    _reason("topology",
                            "Barrier candidate lies inside the edge-exclusion zone."))
            elif candidate["edge_relief"] < float(config["min_prominence"]):
                candidate["reasons"].append(
                    _reason("edge_support",
                            "Edge basin does not rise enough into the interior."))
            if frames > 1 and candidate["local_frame_support"] < edge_frame_floor:
                candidate["reasons"].append(
                    _reason("edge_support",
                            "Edge basin lacks enough nearby multi-window support."))


def _rejection_summary(candidates):
    summary = {}
    for candidate in candidates:
        for reason in candidate["reasons"]:
            summary[reason["code"]] = summary.get(reason["code"], 0) + 1
    return summary


def _sampling_dominates(candidates):
    reason_codes = [
        reason["code"] for candidate in candidates for reason in candidate["reasons"]
    ]
    if not reason_codes:
        return False
    sampling_like = sum(code in {"sampling", "edge_support"} for code in reason_codes)
    return sampling_like >= max(1, int(np.ceil(0.6 * len(reason_codes))))


def _candidate_triplets(minima, maxima, config):
    spacing_floor = max(1, int(np.ceil(float(config["min_basin_width"]))))
    triplets = []
    accepted_minima = [candidate for candidate in minima if not candidate["reasons"]]
    accepted_maxima = [candidate for candidate in maxima if not candidate["reasons"]]

    for barrier in accepted_maxima:
        left_basins = [candidate for candidate in accepted_minima
                       if candidate["index"] < barrier["index"]]
        right_basins = [candidate for candidate in accepted_minima
                        if candidate["index"] > barrier["index"]]
        for rs in left_basins:
            for ps in right_basins:
                left_height = barrier["y"] - rs["y"]
                right_height = barrier["y"] - ps["y"]
                if left_height < float(config["min_prominence"]):
                    continue
                if right_height < float(config["min_prominence"]):
                    continue
                if (barrier["index"] - rs["index"]) < spacing_floor:
                    continue
                if (ps["index"] - barrier["index"]) < spacing_floor:
                    continue

                score = (
                    min(left_height, right_height) +
                    0.25 * (rs["prominence"] + ps["prominence"]) +
                    0.15 * barrier["prominence"] -
                    0.05 * abs(left_height - right_height) +
                    0.02 * (rs["local_population"] + ps["local_population"] +
                            barrier["local_population"]))
                triplets.append({
                    "rs": rs,
                    "ts": barrier,
                    "ps": ps,
                    "left_barrier": float(left_height),
                    "right_barrier": float(right_height),
                    "score": float(score),
                })

    triplets.sort(key=lambda triplet: triplet["score"], reverse=True)
    return triplets


def _reason_codes(candidate):
    return {reason["code"] for reason in candidate["reasons"]}


def _unique_list(items):
    return list(dict.fromkeys(items))


def _state_with_assignment(candidate, tier, penalties=None, warnings=None):
    assigned = dict(candidate)
    assigned["tier"] = tier
    assigned["penalties"] = _unique_list([] if penalties is None else penalties)
    assigned["warnings"] = _unique_list([] if warnings is None else warnings)
    assigned["accepted_strictly"] = tier == "strict"
    return assigned


def _assignment_mode_priority(mode):
    return {"strict": 2, "relaxed": 1, "fallback": 0}[mode]


def _penalty_weight(code):
    return {
        "endpoint_basin": 0.8,
        "relaxed_basin_prominence": 0.8,
        "relaxed_width": 0.6,
        "weak_edge_support": 0.8,
        "relaxed_prominence": 0.8,
        "side_fallback_basin": 1.6,
        "bracketed_ts": 1.4,
        "reduced_barrier": 0.8,
        "downhill_barrier": 1.2,
        "barrier_asymmetry": 0.4,
    }.get(code, 0.5)


def _robust_minimum_candidate(candidate):
    codes = _reason_codes(candidate)
    penalties = []
    warnings = []

    if candidate["source"] == "endpoint":
        penalties.append("endpoint_basin")
        warnings.append(
            "Selected an endpoint basin because interior support was weak on this side."
        )

    if not codes and candidate["source"] != "endpoint":
        return _state_with_assignment(candidate, "strict")

    allowed_codes = {"width", "prominence", "edge_support"}
    if codes - allowed_codes:
        return None

    if "width" in codes:
        penalties.append("relaxed_width")
        warnings.append(
            "Accepted a basin narrower than the preferred minimum basin width.")
    if "prominence" in codes:
        penalties.append("relaxed_basin_prominence")
        warnings.append(
            "Accepted a basin with reduced prominence relative to the default threshold."
        )
    if "edge_support" in codes:
        penalties.append("weak_edge_support")
        warnings.append(
            "Accepted an edge basin with limited interior support.")

    tier = "relaxed" if penalties else "strict"
    return _state_with_assignment(candidate, tier, penalties, warnings)


def _robust_maximum_candidate(candidate):
    codes = _reason_codes(candidate)
    penalties = []
    warnings = []

    if not codes:
        return _state_with_assignment(candidate, "strict")

    allowed_codes = {"width", "prominence"}
    if codes - allowed_codes:
        return None

    if "width" in codes:
        penalties.append("relaxed_width")
        warnings.append(
            "Accepted a barrier narrower than the preferred minimum basin width."
        )
    if "prominence" in codes:
        penalties.append("relaxed_prominence")
        warnings.append(
            "Accepted a barrier with reduced prominence relative to the default threshold."
        )

    return _state_with_assignment(candidate, "relaxed", penalties, warnings)


def _candidate_side(index, size):
    return "left" if index < (size / 2.0) else "right"


def _fallback_side_indices(size, side):
    midpoint = size // 2
    if side == "left":
        stop = max(1, midpoint)
        return np.arange(0, stop, dtype=int)
    start = midpoint if size % 2 == 0 else midpoint + 1
    return np.arange(start, size, dtype=int)


def _fallback_minimum_candidate(profile, smoothed_gg, side, config):
    indices = _fallback_side_indices(profile["dx"].size, side)
    if indices.size == 0:
        return None

    sampling = profile["sampling"]

    def rank_key(index):
        return (profile["gg"][index], -sampling["max_population"][index], index)

    best_index = int(sorted(indices, key=rank_key)[0])
    fallback = _candidate_payload("minimum",
                                  best_index,
                                  profile["dx"],
                                  profile["gg"],
                                  smoothed_gg,
                                  0.0,
                                  1.0,
                                  sampling,
                                  config,
                                  source="fallback")
    return _state_with_assignment(
        fallback,
        "fallback",
        penalties=["side_fallback_basin"],
        warnings=[
            f"Using the lowest-energy sampled {side} bin as a fallback basin."
        ],
    )


def _bracketed_ts_candidate(profile, smoothed_gg, rs, ps, config):
    indices = np.arange(rs["index"] + 1, ps["index"], dtype=int)
    if indices.size == 0:
        return None

    sampling = profile["sampling"]

    def rank_key(index):
        return (-profile["gg"][index], -sampling["max_population"][index], index)

    best_index = int(sorted(indices, key=rank_key)[0])
    bracketed = _candidate_payload("maximum",
                                   best_index,
                                   profile["dx"],
                                   profile["gg"],
                                   smoothed_gg,
                                   0.0,
                                   1.0,
                                   sampling,
                                   config,
                                   source="bracketed")
    return _state_with_assignment(
        bracketed,
        "fallback",
        penalties=["bracketed_ts"],
        warnings=[
            "Using the highest sampled bin between RS and PS as a fallback transition state."
        ],
    )


def _triplet_mode(rs, ts, ps):
    tiers = {rs["tier"], ts["tier"], ps["tier"]}
    if "fallback" in tiers:
        return "fallback"
    if "relaxed" in tiers:
        return "relaxed"
    return "strict"


def _triplet_confidence(mode):
    return {"strict": "high", "relaxed": "medium", "fallback": "low"}[mode]


def _ranked_triplet(rs, ts, ps, config):
    spacing_floor = 1
    if rs["index"] >= ts["index"] or ts["index"] >= ps["index"]:
        return None
    if (ts["index"] - rs["index"]) < spacing_floor:
        return None
    if (ps["index"] - ts["index"]) < spacing_floor:
        return None

    left_height = ts["y"] - rs["y"]
    right_height = ts["y"] - ps["y"]
    if left_height <= 0.0 and right_height <= 0.0:
        return None

    penalties = []
    warnings = []
    for state in (rs, ts, ps):
        penalties.extend(state.get("penalties", []))
        warnings.extend(state.get("warnings", []))

    if min(left_height, right_height) <= 0.0:
        penalties.append("downhill_barrier")
        warnings.append(
            "Selected TS lies at or below one basin, so that side of the profile has a zero or negative barrier."
        )

    if min(left_height, right_height) < float(config["min_prominence"]):
        penalties.append("reduced_barrier")
        warnings.append(
            "Selected TS gives a shallow barrier on at least one side of the profile."
        )

    if abs(left_height - right_height) > max(0.5, 0.5 * float(config["min_prominence"])):
        penalties.append("barrier_asymmetry")
        warnings.append(
            "Selected TS is noticeably asymmetric relative to the reactant and product basins."
        )

    penalties = _unique_list(penalties)
    warnings = _unique_list(warnings)
    penalty_total = sum(_penalty_weight(code) for code in penalties)

    score = (
        min(left_height, right_height) +
        0.20 * (rs["prominence"] + ps["prominence"]) +
        0.12 * ts["prominence"] +
        0.04 * (rs["width"] + ps["width"] + ts["width"]) +
        0.02 * (rs["local_population"] + ps["local_population"] +
                ts["local_population"]) - 0.05 * abs(left_height - right_height) -
        penalty_total)

    mode = _triplet_mode(rs, ts, ps)
    return {
        "rs": rs,
        "ts": ts,
        "ps": ps,
        "left_barrier": float(left_height),
        "right_barrier": float(right_height),
        "score": float(score),
        "assignment_mode": mode,
        "confidence": _triplet_confidence(mode),
        "fallbacks_used": penalties,
        "warnings": warnings,
    }


def _robust_candidate_triplets(profile, smoothed_gg, minima, maxima, config):
    size = profile["dx"].size
    side_minima = {"left": [], "right": []}
    for candidate in minima:
        assigned = _robust_minimum_candidate(candidate)
        if assigned is None:
            continue
        side_minima[_candidate_side(assigned["index"], size)].append(assigned)

    for side in ("left", "right"):
        if not side_minima[side]:
            fallback = _fallback_minimum_candidate(profile, smoothed_gg, side, config)
            if fallback is not None:
                side_minima[side].append(fallback)

    robust_maxima = []
    for candidate in maxima:
        assigned = _robust_maximum_candidate(candidate)
        if assigned is not None:
            robust_maxima.append(assigned)

    triplets = []
    for rs in side_minima["left"]:
        for ps in side_minima["right"]:
            if rs["index"] >= ps["index"] - 1:
                continue
            between = [
                ts for ts in robust_maxima
                if rs["index"] < ts["index"] < ps["index"]
            ]
            if between:
                for ts in between:
                    triplet = _ranked_triplet(rs, ts, ps, config)
                    if triplet is not None:
                        triplets.append(triplet)
            else:
                fallback_ts = _bracketed_ts_candidate(profile, smoothed_gg, rs, ps,
                                                      config)
                if fallback_ts is None:
                    continue
                triplet = _ranked_triplet(rs, fallback_ts, ps, config)
                if triplet is not None:
                    triplets.append(triplet)

    triplets.sort(key=lambda triplet:
                  (_assignment_mode_priority(triplet["assignment_mode"]),
                   triplet["score"]),
                  reverse=True)
    return triplets


def _combined_rejection_summary(diagnostics):
    combined = {}
    for section in ("minima", "maxima"):
        for code, count in diagnostics["rejection_summary"][section].items():
            combined[code] = combined.get(code, 0) + count
    return combined


def _build_detection_diagnostics(profile,
                                 smoothed_gg,
                                 minima,
                                 maxima,
                                 triplets,
                                 selected_triplet=None,
                                 competing_triplets=None):
    return {
        "raw_profile": np.c_[profile["dx"], profile["gg"]],
        "smoothed_profile": np.c_[profile["dx"], smoothed_gg],
        "sampling": {
            key: profile["sampling"][key].copy()
            for key in ("max_population", "total_population", "frame_support")
        },
        "resolution": dict(profile["resolution"]),
        "candidate_counts": {
            "minima_total": len(minima),
            "minima_accepted": sum(not candidate["reasons"] for candidate in minima),
            "maxima_total": len(maxima),
            "maxima_accepted": sum(not candidate["reasons"] for candidate in maxima),
            "valid_triplets": len(triplets),
        },
        "rejection_summary": {
            "minima": _rejection_summary(minima),
            "maxima": _rejection_summary(maxima),
        },
        "candidates": {
            "minima": minima,
            "maxima": maxima,
        },
        "valid_triplets": triplets,
        "selected_triplet": selected_triplet,
        "competing_triplets": [] if competing_triplets is None else competing_triplets,
    }


def _format_detection_failure(title, context, diagnostics, guidance):
    counts = diagnostics["candidate_counts"]
    rejection_bits = [
        f"{code}={count}" for code, count in sorted(
            _combined_rejection_summary(diagnostics).items())
    ]
    message = (
        f"{title} during EVB {context}. candidates: minima "
        f"{counts['minima_accepted']}/{counts['minima_total']} accepted, maxima "
        f"{counts['maxima_accepted']}/{counts['maxima_total']} accepted, "
        f"valid triplets={counts['valid_triplets']}."
    )
    if rejection_bits:
        message += " rejections: " + ", ".join(rejection_bits) + "."
    if guidance:
        message += " " + guidance
    return message


def _raise_detection_failure(error_cls, title, context, diagnostics, guidance):
    raise error_cls(_format_detection_failure(title, context, diagnostics, guidance),
                    failure_code=error_cls.__name__,
                    diagnostics=diagnostics)


def _detect_profile_states(profile, context="mapping"):
    config = get_state_detection_config()
    assignment_policy = str(config.get("assignment_policy", "robust"))
    if profile["dx"].size == 0:
        diagnostics = {
            "raw_profile": np.empty((0, 2)),
            "smoothed_profile": np.empty((0, 2)),
            "sampling": {
                key: profile["sampling"][key].copy()
                for key in ("max_population", "total_population", "frame_support")
            },
            "resolution": dict(profile["resolution"]),
            "candidate_counts": {
                "minima_total": 0,
                "minima_accepted": 0,
                "maxima_total": 0,
                "maxima_accepted": 0,
                "valid_triplets": 0,
            },
            "rejection_summary": {
                "minima": {},
                "maxima": {},
            },
            "candidates": {
                "minima": [],
                "maxima": [],
            },
            "valid_triplets": [],
            "selected_triplet": None,
            "competing_triplets": [],
        }
        raise EvbInsufficientSamplingError(
            f"No EVB bins satisfied minpts={profile['resolution']['minpts']} across "
            f"{profile['resolution']['bins']} bins during EVB {context}. Consider "
            "lowering minpts, reducing bins, or sampling more densely.",
            failure_code="empty_profile",
            diagnostics=diagnostics)
    if profile["dx"].size < 3:
        diagnostics = {
            "raw_profile": np.c_[profile["dx"], profile["gg"]],
            "smoothed_profile": np.c_[profile["dx"], profile["gg"]],
            "sampling": {
                key: profile["sampling"][key].copy()
                for key in ("max_population", "total_population", "frame_support")
            },
            "resolution": dict(profile["resolution"]),
            "candidate_counts": {
                "minima_total": 0,
                "minima_accepted": 0,
                "maxima_total": 0,
                "maxima_accepted": 0,
                "valid_triplets": 0,
            },
            "rejection_summary": {
                "minima": {},
                "maxima": {},
            },
            "candidates": {
                "minima": [],
                "maxima": [],
            },
            "valid_triplets": [],
            "selected_triplet": None,
            "competing_triplets": [],
        }
        raise EvbInsufficientSamplingError(
            f"Resolved EVB profile contains only {profile['dx'].size} populated bin(s) "
            f"during EVB {context}; at least 3 bins are required to bracket RS, TS, and PS.",
            failure_code="profile_too_short",
            diagnostics=diagnostics)

    smoothed_gg = _smoothed_profile_for_detection(profile["gg"], config)
    minima = _extract_candidates("minimum",
                                 profile["dx"],
                                 profile["gg"],
                                 smoothed_gg,
                                 profile["sampling"],
                                 config)
    maxima = _extract_candidates("maximum",
                                 profile["dx"],
                                 profile["gg"],
                                 smoothed_gg,
                                 profile["sampling"],
                                 config)

    _annotate_candidate_rejections(minima,
                                   smoothed_gg,
                                   profile["resolution"]["minpts"],
                                   profile["resolution"]["frames"],
                                   config)
    _annotate_candidate_rejections(maxima,
                                   smoothed_gg,
                                   profile["resolution"]["minpts"],
                                   profile["resolution"]["frames"],
                                   config)

    strict_triplets = _candidate_triplets(minima, maxima, config)
    diagnostics = _build_detection_diagnostics(profile, smoothed_gg, minima, maxima,
                                               strict_triplets)
    accepted_minima = [candidate for candidate in minima if not candidate["reasons"]]
    accepted_maxima = [candidate for candidate in maxima if not candidate["reasons"]]
    profile_midpoint = 0.5 * (profile["dx"][0] + profile["dx"][-1])
    left_minima = [candidate for candidate in accepted_minima
                   if candidate["x"] <= profile_midpoint]
    right_minima = [candidate for candidate in accepted_minima
                    if candidate["x"] >= profile_midpoint]

    if assignment_policy == "strict":
        if not accepted_minima and _sampling_dominates(minima):
            _raise_detection_failure(
                EvbInsufficientSamplingError,
                "Insufficient PMF sampling for robust basin detection",
                context,
                diagnostics,
                "Increase sampling density or relax auto-managed minpts/bins.")
        if not left_minima:
            _raise_detection_failure(
                EvbNoTrustworthyReactantStateError,
                "No trustworthy reactant basin",
                context,
                diagnostics,
                "Endpoint-like basins are rejected unless they have clear interior support.")
        if not right_minima:
            _raise_detection_failure(
                EvbNoTrustworthyProductStateError,
                "No trustworthy product basin",
                context,
                diagnostics,
                "Endpoint-like basins are rejected unless they have clear interior support.")
        if not accepted_maxima:
            error_cls = (EvbInsufficientSamplingError
                         if _sampling_dominates(maxima) else
                         EvbNoValidTransitionStateError)
            _raise_detection_failure(
                error_cls,
                "No trustworthy barrier candidate",
                context,
                diagnostics,
                "Inspect the smoothed PMF and consider adjusting minpts, bins, or sampling.")
        if not strict_triplets:
            error_cls = (EvbInsufficientSamplingError
                         if (_sampling_dominates(minima) or _sampling_dominates(maxima))
                         else EvbNoValidTransitionStateError)
            _raise_detection_failure(
                error_cls,
                "No valid TS between trustworthy basins",
                context,
                diagnostics,
                "A TS must sit between two supported basins with enough prominence and spacing.")

        best_triplet = strict_triplets[0]
        tolerance = max(0.5, 0.1 * best_triplet["score"])
        competing_triplets = [
            triplet for triplet in strict_triplets[1:]
            if (triplet["ts"]["index"] != best_triplet["ts"]["index"] and
                abs(triplet["score"] - best_triplet["score"]) <= tolerance)
        ]
        if competing_triplets:
            diagnostics = _build_detection_diagnostics(profile,
                                                       smoothed_gg,
                                                       minima,
                                                       maxima,
                                                       strict_triplets,
                                                       selected_triplet=best_triplet,
                                                       competing_triplets=competing_triplets)
            _raise_detection_failure(
                EvbAmbiguousTransitionStateError,
                "Multiple competing TS candidates have similar support",
                context,
                diagnostics,
                "Inspect the diagnostic plot or tighten the smoothing/resolution choices.")

        best_triplet["assignment_mode"] = "strict"
        best_triplet["confidence"] = "high"
        best_triplet["fallbacks_used"] = []
        best_triplet["warnings"] = []
        best_triplet["rs"] = _state_with_assignment(best_triplet["rs"], "strict")
        best_triplet["ts"] = _state_with_assignment(best_triplet["ts"], "strict")
        best_triplet["ps"] = _state_with_assignment(best_triplet["ps"], "strict")
        diagnostics["selected_triplet"] = best_triplet
        diagnostics["assignment_mode"] = "strict"
        diagnostics["confidence"] = "high"
        diagnostics["warnings"] = []
        diagnostics["fallbacks_used"] = []
        diagnostics["assignment_policy"] = assignment_policy
        diagnostics["triplet_score"] = best_triplet["score"]
        return {
            "smoothed_gg": smoothed_gg,
            "selected_triplet": best_triplet,
            "diagnostics": diagnostics,
            "assignment_policy": assignment_policy,
            "assignment_mode": "strict",
            "confidence": "high",
            "warnings": [],
            "fallbacks_used": [],
            "triplet_score": best_triplet["score"],
        }

    triplets = _robust_candidate_triplets(profile, smoothed_gg, minima, maxima,
                                          config)
    if not triplets:
        error_cls = (EvbInsufficientSamplingError
                     if (_sampling_dominates(minima) or _sampling_dominates(maxima))
                     else EvbNoValidTransitionStateError)
        diagnostics = _build_detection_diagnostics(profile,
                                                   smoothed_gg,
                                                   minima,
                                                   maxima,
                                                   [])
        _raise_detection_failure(
            error_cls,
            "No physical RS/TS/PS assignment could be constructed",
            context,
            diagnostics,
            "Check whether the mapped profile retains at least three supported bins across the reaction coordinate.")

    best_triplet = triplets[0]
    diagnostics = _build_detection_diagnostics(profile,
                                               smoothed_gg,
                                               minima,
                                               maxima,
                                               triplets,
                                               selected_triplet=best_triplet,
                                               competing_triplets=triplets[1:4])
    diagnostics["assignment_mode"] = best_triplet["assignment_mode"]
    diagnostics["confidence"] = best_triplet["confidence"]
    diagnostics["warnings"] = best_triplet["warnings"]
    diagnostics["fallbacks_used"] = best_triplet["fallbacks_used"]
    diagnostics["assignment_policy"] = assignment_policy
    diagnostics["triplet_score"] = best_triplet["score"]
    return {
        "smoothed_gg": smoothed_gg,
        "selected_triplet": best_triplet,
        "diagnostics": diagnostics,
        "assignment_policy": assignment_policy,
        "assignment_mode": best_triplet["assignment_mode"],
        "confidence": best_triplet["confidence"],
        "warnings": best_triplet["warnings"],
        "fallbacks_used": best_triplet["fallbacks_used"],
        "triplet_score": best_triplet["score"],
    }


def _save_xy_table(dx, g1, g2, gg, output_stem):
    data = np.c_[dx, g1, g2, gg]
    np.savetxt(output_stem + ".dat",
               data,
               header="e1-e2 dGa dGb dGg",
               comments="",
               fmt="%12.6f")
    return data


def _plot_profile(output_path,
                  x,
                  series,
                  labels,
                  title,
                  ylabel,
                  markers=None,
                  line_styles=None):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    if line_styles is None:
        line_styles = [{} for _ in series]
    for values, label, style in zip(series, labels, line_styles):
        kwargs = {"linewidth": 1.6, "label": label}
        kwargs.update(style)
        ax.plot(x, values, **kwargs)
    if markers:
        for marker in markers:
            ax.scatter(marker["x"],
                       marker["y"],
                       color=marker["color"],
                       label=marker["label"])
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel(r"$\epsilon_1 - \epsilon_2$ (kJ/mol)")
    ax.set_ylabel(ylabel)
    if labels or markers:
        ax.legend(loc="best")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _failure_diagnostic_markers(diagnostics):
    accepted_markers = []
    rejected_markers = []

    selected_triplet = diagnostics.get("selected_triplet")
    if selected_triplet is not None:
        accepted_markers = [{
            "x": selected_triplet["rs"]["x"],
            "y": selected_triplet["rs"]["y"],
            "color": "tab:green",
            "marker": "o",
            "label": "RS",
        }, {
            "x": selected_triplet["ts"]["x"],
            "y": selected_triplet["ts"]["y"],
            "color": "tab:red",
            "marker": "o",
            "label": "TS",
        }, {
            "x": selected_triplet["ps"]["x"],
            "y": selected_triplet["ps"]["y"],
            "color": "tab:orange",
            "marker": "o",
            "label": "PS",
        }]

    for kind in ("minima", "maxima"):
        label = "Rejected basin" if kind == "minima" else "Rejected barrier"
        color = "tab:blue" if kind == "minima" else "tab:red"
        marker = "x" if kind == "minima" else "1"
        for candidate in diagnostics.get("candidates", {}).get(kind, []):
            if not candidate["reasons"]:
                continue
            rejected_markers.append({
                "x": candidate["x"],
                "y": candidate["y"],
                "color": color,
                "marker": marker,
                "label": label,
            })
    return accepted_markers, rejected_markers


def _save_failure_diagnostic(out, reason, diagnostics):
    if out is None:
        return None
    plot_path = _output_stem(out) + "_failure.png"
    accepted_markers, rejected_markers = _failure_diagnostic_markers(diagnostics)
    plot_mapped_pmf_diagnostic(
        plot_path,
        diagnostics.get("raw_profile", np.empty((0, 2))),
        detection_profile=diagnostics.get("smoothed_profile", np.empty((0, 2))),
        accepted_markers=accepted_markers,
        rejected_markers=rejected_markers,
        populations=diagnostics.get("sampling", {}).get("max_population"),
        support_counts=diagnostics.get("sampling", {}).get("frame_support"),
        minpts=diagnostics.get("resolution", {}).get("minpts"),
        title="Mapped EVB Profile",
        failure_reason=reason)
    return plot_path


def _raise_mapping_error(exc, out=None):
    diagnostics = {} if getattr(exc, "diagnostics", None) is None else exc.diagnostics
    message = str(exc)
    plot_path = _save_failure_diagnostic(out, message, diagnostics)
    if plot_path is not None:
        message = f"{message} Diagnostic plot saved to {plot_path}."
    raise exc.__class__(message,
                        failure_code=getattr(exc, "failure_code", None),
                        diagnostics=diagnostics)


def _build_profile(A_xvgs, B_xvgs, step, alpha, hij, dl, kb, temp, bins,
                   minpts, reverse_isfs, context):
    _validate_xvg_inputs(A_xvgs, B_xvgs, context)
    kT = kb * temp
    feps, N = read_data(A_xvgs,
                        B_xvgs,
                        alpha,
                        hij,
                        dl,
                        step,
                        reverse_isfs=reverse_isfs)
    dg_lambda, _ = gofl(feps, len(A_xvgs), N, kT, dl)
    x = gaps(feps, bins - 1)
    free_en = uofx(feps, bins, x, len(A_xvgs))
    dx, g1, g2, gg = gofx(dg_lambda, free_en, bins, len(A_xvgs), minpts, kT, x,
                          N)
    return {
        "dx": dx,
        "g1": g1,
        "g2": g2,
        "gg": gg,
        "sampling": _profile_sampling_arrays(free_en, minpts),
        "resolution": {
            "step": int(step),
            "bins": int(bins),
            "minpts": int(minpts),
            "frames": int(len(A_xvgs)),
        },
    }


def _finalize_profile(profile, detection_result, retry_attempts=None):
    retry_attempts = [] if retry_attempts is None else retry_attempts
    triplet = detection_result["selected_triplet"]
    rs_energy = triplet["rs"]["raw_y"]
    ts_energy = triplet["ts"]["raw_y"] - rs_energy
    dg_energy = triplet["ps"]["raw_y"] - rs_energy
    shifted_smoothed = detection_result["smoothed_gg"] - rs_energy
    resolution_retry_used = any(
        attempt.get("resolution") != profile["resolution"] for attempt in retry_attempts)

    state_detection = {
        "status": "success",
        "resolution": dict(profile["resolution"]),
        "resolution_retry_used": resolution_retry_used,
        "retry_attempts": retry_attempts,
        "assignment_policy": detection_result.get("assignment_policy", "robust"),
        "assignment_mode": detection_result.get("assignment_mode", "strict"),
        "confidence": detection_result.get("confidence", "high"),
        "warnings": detection_result.get("warnings", []),
        "fallbacks_used": detection_result.get("fallbacks_used", []),
        "triplet_score": detection_result.get("triplet_score"),
        "raw_pmf": np.c_[profile["dx"], profile["gg"] - rs_energy],
        "smoothed_pmf": np.c_[profile["dx"], shifted_smoothed],
        "states": {
            "RS": {
                "x": triplet["rs"]["x"],
                "raw_y": triplet["rs"]["raw_y"],
                "source": triplet["rs"].get("source", "interior"),
                "penalties": triplet["rs"].get("penalties", []),
                "accepted_strictly": triplet["rs"].get("accepted_strictly", True),
            },
            "TS": {
                "x": triplet["ts"]["x"],
                "raw_y": triplet["ts"]["raw_y"],
                "source": triplet["ts"].get("source", "interior"),
                "penalties": triplet["ts"].get("penalties", []),
                "accepted_strictly": triplet["ts"].get("accepted_strictly", True),
            },
            "PS": {
                "x": triplet["ps"]["x"],
                "raw_y": triplet["ps"]["raw_y"],
                "source": triplet["ps"].get("source", "interior"),
                "penalties": triplet["ps"].get("penalties", []),
                "accepted_strictly": triplet["ps"].get("accepted_strictly", True),
            },
        },
        "candidate_counts": detection_result["diagnostics"]["candidate_counts"],
        "candidate_summary": detection_result["diagnostics"]["rejection_summary"],
        "candidates": detection_result["diagnostics"]["candidates"],
        "valid_triplets": detection_result["diagnostics"]["valid_triplets"],
        "selected_triplet": triplet,
        "sampling": detection_result["diagnostics"]["sampling"],
    }

    return {
        "dx": profile["dx"],
        "g1": profile["g1"] - rs_energy,
        "g2": profile["g2"] - rs_energy,
        "gg": profile["gg"] - rs_energy,
        "gg_smoothed": shifted_smoothed,
        "gg_smoothed_x": profile["dx"],
        "ts_energy": ts_energy,
        "dg_energy": dg_energy,
        "rst": [triplet["rs"]["index"], triplet["rs"]["x"], triplet["rs"]["raw_y"], 0],
        "pst": [triplet["ps"]["index"], triplet["ps"]["x"], triplet["ps"]["raw_y"], 0],
        "tst": [[triplet["ts"]["index"], triplet["ts"]["x"], triplet["ts"]["raw_y"]]],
        "state_detection": state_detection,
    }


def _resolution_retry_ladder(step, bins, minpts):
    retries = []
    seen = {(int(step), int(bins), int(minpts))}

    def add(step_value, bins_value, minpts_value):
        candidate = (int(max(1, step_value)), int(max(20, bins_value)),
                     int(max(3, minpts_value)))
        if candidate in seen:
            return
        seen.add(candidate)
        retries.append({
            "step": candidate[0],
            "bins": candidate[1],
            "minpts": candidate[2],
        })

    add(step, bins, minpts - 2)
    add(step, bins, int(round(minpts * 0.8)))
    add(step, int(round(bins * 0.9)), max(3, minpts - 1))
    add(step, int(round(bins * 0.8)), int(round(minpts * 0.85)))
    add(step + 1, int(round(bins * 0.9)), int(round(minpts * 0.85)))
    return retries


def _resolve_profile(A_xvgs,
                     B_xvgs,
                     step,
                     alpha,
                     hij,
                     dl,
                     kb,
                     temp,
                     bins,
                     minpts,
                     reverse_isfs,
                     context,
                     allow_resolution_retry=None):
    config = get_state_detection_config()
    assignment_policy = str(config.get("assignment_policy", "robust"))
    if allow_resolution_retry is None:
        allow_resolution_retry = bool(config["auto_retry_resolution"])
    attempts = [{
        "step": int(step),
        "bins": int(bins),
        "minpts": int(minpts),
    }]
    if allow_resolution_retry:
        attempts.extend(_resolution_retry_ladder(step, bins, minpts))

    retry_attempts = []
    successes = []
    first_exc = None

    for attempt in attempts:
        current_profile = _build_profile(A_xvgs, B_xvgs, attempt["step"], alpha, hij,
                                         dl, kb, temp, attempt["bins"],
                                         attempt["minpts"], reverse_isfs, context)
        try:
            detection_result = _detect_profile_states(current_profile,
                                                      context=context)
            retry_attempts.append({
                "resolution": dict(current_profile["resolution"]),
                "status": "success",
                "assignment_mode": detection_result.get("assignment_mode",
                                                          "strict"),
                "confidence": detection_result.get("confidence", "high"),
                "triplet_score": detection_result.get("triplet_score"),
                "selected": False,
            })
            successes.append((current_profile, detection_result,
                              len(retry_attempts) - 1))
            if assignment_policy == "strict":
                retry_attempts[-1]["selected"] = True
                return _finalize_profile(current_profile,
                                         detection_result,
                                         retry_attempts=retry_attempts)
        except EvbMappingError as exc:
            if first_exc is None:
                first_exc = exc
            retry_attempts.append({
                "resolution": dict(current_profile["resolution"]),
                "status": "failed",
                "failure_class": exc.__class__.__name__,
                "failure_code": getattr(exc, "failure_code", None),
                "selected": False,
            })
            if not allow_resolution_retry:
                break

    if successes:
        best_success = max(
            successes,
            key=lambda item: (
                _assignment_mode_priority(item[1].get("assignment_mode",
                                                      "strict")),
                item[1].get("triplet_score", float("-inf")),
            ),
        )
        retry_attempts[best_success[2]]["selected"] = True
        return _finalize_profile(best_success[0],
                                 best_success[1],
                                 retry_attempts=retry_attempts)

    diagnostics = dict(first_exc.diagnostics)
    diagnostics["retry_attempts"] = retry_attempts
    raise first_exc.__class__(str(first_exc),
                              failure_code=getattr(first_exc, "failure_code",
                                                   None),
                              diagnostics=diagnostics)


def _write_profile_outputs(output_stem, profile):
    data = _save_xy_table(output_stem=output_stem,
                          dx=profile["dx"],
                          g1=profile["g1"],
                          g2=profile["g2"],
                          gg=profile["gg"])
    plot_mapped_pmf_diagnostic(
        output_stem + "_evb.png",
        np.c_[profile["dx"], profile["gg"]],
        detection_profile=np.c_[profile["gg_smoothed_x"], profile["gg_smoothed"]],
        accepted_markers=[{
            "x": profile["rst"][1],
            "y": 0.0,
            "color": "tab:green",
            "marker": "o",
            "label": "RS",
        }, {
            "x": profile["pst"][1],
            "y": profile["dg_energy"],
            "color": "tab:orange",
            "marker": "o",
            "label": "PS",
        }, {
            "x": profile["tst"][0][1],
            "y": profile["ts_energy"],
            "color": "tab:red",
            "marker": "o",
            "label": "TS",
        }],
        populations=profile["state_detection"]["sampling"]["max_population"],
        support_counts=profile["state_detection"]["sampling"]["frame_support"],
        minpts=profile["state_detection"]["resolution"]["minpts"],
        title="Mapped EVB Profile")
    _plot_profile(output_stem + "_diabatics.png",
                  profile["dx"], [profile["g1"], profile["g2"]],
                  ["Diabatic A", "Diabatic B"], "Diabatic Profiles",
                  r"$\Delta$G (kJ/mol)")
    return data


def evbmap(A_xvgs,
           B_xvgs,
           step,
           alpha,
           hij,
           dl,
           kb,
           temp,
           bins,
           minpts,
           out,
           reverse_isfs=False,
           return_metadata=False,
           allow_resolution_retry=None):
    try:
        profile = _resolve_profile(A_xvgs,
                                   B_xvgs,
                                   step,
                                   alpha,
                                   hij,
                                   dl,
                                   kb,
                                   temp,
                                   bins,
                                   minpts,
                                   reverse_isfs,
                                   "mapping",
                                   allow_resolution_retry=allow_resolution_retry)
    except EvbInputError as exc:
        _raise_mapping_error(exc, out=out)
    except EvbMappingError as exc:
        _raise_mapping_error(exc, out=out)

    output_stem = _output_stem(out)
    dg_arr = _write_profile_outputs(output_stem, profile)
    if return_metadata:
        return profile["ts_energy"], profile["dg_energy"], dg_arr, profile[
            "state_detection"]
    return profile["ts_energy"], profile["dg_energy"], dg_arr


def fetchA_B_xvgs(A_path, B_path):
    A_xvgs = sorted(glob.glob(os.path.join(A_path, "*.xvg"), recursive=False))
    B_xvgs = sorted(glob.glob(os.path.join(B_path, "*.xvg"), recursive=False))
    return A_xvgs, B_xvgs


def evbmap_nooutput(A_xvgs, B_xvgs, step, alpha, hij, dl, kb, temp, bins,
                    minpts, reverse_isfs, return_metadata=False,
                    allow_resolution_retry=None):
    profile = _resolve_profile(A_xvgs,
                               B_xvgs,
                               step,
                               alpha,
                               hij,
                               dl,
                               kb,
                               temp,
                               bins,
                               minpts,
                               reverse_isfs,
                               "fitting",
                               allow_resolution_retry=allow_resolution_retry)
    if return_metadata:
        return profile["ts_energy"], profile["dg_energy"], profile[
            "state_detection"]
    return profile["ts_energy"], profile["dg_energy"]
