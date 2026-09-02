from copy import deepcopy

import numpy as np

from .evb_plot import plot_data, remove_underline, smooth_data
from .evb_utils import (EvbMappingError, evbmap, evbmap_nooutput, fetchA_B_xvgs,
                        set_state_detection_config,
                        summarize_reference_potentials,
                        set_profile_smoothing_config)


class Evb:
    """High-level EVB fitting and mapping helper."""

    _AUTO_TRACKED_FIELDS = {"alpha_guess", "H12_guess", "step", "bins", "minpts"}

    def __init__(self):
        object.__setattr__(self, "_track_user_overrides", False)
        object.__setattr__(self, "_manual_overrides",
                           {field: False for field in self._AUTO_TRACKED_FIELDS})
        self.step = 1
        self.barrier = 0
        self.free_energy = 0
        self.alpha_guess = 0
        self.H12_guess = 20
        self.alpha = None
        self.H12 = None
        # Normal batch runs report only completed mapping results.  Set this to
        # True explicitly when fit progress or state-assignment diagnostics are
        # needed during troubleshooting.
        self.verbose = False
        self.max_iter = 2000
        self.tol = 1e-3
        self.fitting_slowness = 0.1
        self.bins = 100
        self.minpts = 10
        self.reverse_isfs = False
        self.dl = 0.05
        self.ref_ABxvgs = None
        self.kb = 0.008314
        self.temp = 298.15
        self.system_data = None
        self._parameter_source = "unset"
        self.profile_secondary_smoothing = True
        self.profile_secondary_smoothing_for_fitting = False
        self.profile_smoothing_gaussian_sigma = 1.0
        self.profile_smoothing_savgol_window = 9
        self.profile_smoothing_savgol_polyorder = 3
        self.state_detection_enabled = True
        self.state_smoothing_gaussian_sigma = 1.0
        self.state_smoothing_savgol_window = 9
        self.state_smoothing_savgol_polyorder = 3
        self.state_min_prominence = 1.0
        self.state_min_basin_width = 2.0
        self.state_edge_exclusion_bins = 2
        self.state_auto_retry_resolution = True
        self.state_assignment_policy = "robust"
        self._last_auto_guess = None
        object.__setattr__(self, "_track_user_overrides", True)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if getattr(self, "_track_user_overrides", False) and name in getattr(
                self, "_manual_overrides", {}):
            self._manual_overrides[name] = True

    def _set_internal(self, name, value):
        object.__setattr__(self, name, value)

    def _set_if_auto(self, name, value):
        if not self._manual_overrides[name]:
            self._set_internal(name, value)
            return True
        return False

    def _emit(self, message, *, force=False):
        if force or self.verbose:
            print(message)

    def _normalize_pair(self, pair):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                "Each EVB pair must contain exactly two paths: [A_path, B_path]."
            )
        return [str(pair[0]).strip(), str(pair[1]).strip()]

    def _normalize_pairs(self, pairs):
        normalized = [self._normalize_pair(pair) for pair in pairs]
        if self.reverse_isfs:
            return [[pair[1], pair[0]] for pair in normalized]
        return normalized

    def _normalized_systems(self, systems):
        normalized = {}
        for system_id, system in deepcopy(systems).items():
            if "AB_pairs" not in system:
                raise ValueError(
                    f"System {system_id} is missing the required 'AB_pairs' key."
                )
            normalized[system_id] = dict(system)
            normalized[system_id]["AB_pairs"] = self._normalize_pairs(
                system["AB_pairs"])
        return normalized

    def _initial_guesses(self):
        return float(self.alpha_guess), float(self.H12_guess)

    def _adaptive_slowness(self, barrier_diff, free_energy_diff):
        largest_diff = max(abs(barrier_diff), abs(free_energy_diff))
        if largest_diff < 0.2:
            return 0.1
        if largest_diff < 1:
            return 0.5
        return 1.0

    def _format_iteration(self, iteration, alpha, h12, ts_energy, dg_energy):
        return (
            f"iter {iteration:>4} | alpha={alpha:>10.5f} | "
            f"H12={h12:>10.5f} | barrier={ts_energy:>10.5f} | "
            f"dG={dg_energy:>10.5f}"
        )

    def _format_parameter_summary(self, alpha, h12, ref_count):
        return (
            f"Fitted EVB parameters from {ref_count} reference pair(s): "
            f"alpha={alpha:.5f}, H12={h12:.5f} "
            f"(initial guesses: alpha={self.alpha_guess:.5f}, "
            f"H12={self.H12_guess:.5f})"
        )

    def _format_mapping_summary(self):
        if self.alpha is None or self.H12 is None:
            return None
        if self._parameter_source == "fit":
            source = "using fitted parameters"
        else:
            source = "using current parameters"
        smoothing = "state smoothing on"
        if not self.state_detection_enabled:
            smoothing = "state smoothing off"
        retry = "auto-retry on" if self._allow_state_detection_retry(
        ) else "auto-retry off"
        return (
            f"Running EVB mapping {source}: alpha={self.alpha:.5f}, "
            f"H12={self.H12:.5f} ({smoothing}, {retry}, "
            f"assignment={self.state_assignment_policy})"
        )

    def _format_auto_guess_summary(self, guessed, applied_fields):
        parts = [
            f"alpha_guess={guessed['alpha_guess']:.5f}",
            f"H12_guess={guessed['H12_guess']:.5f}",
            f"step={guessed['step']}",
            f"bins={guessed['bins']}",
            f"minpts={guessed['minpts']}",
        ]
        prefix = "Auto-guessed fit setup from reference potentials"
        if applied_fields and len(applied_fields) != len(self._AUTO_TRACKED_FIELDS):
            prefix += f" (applied: {', '.join(applied_fields)})"
        return prefix + ": " + ", ".join(parts)

    def _apply_profile_smoothing_settings(self, enabled=None):
        set_profile_smoothing_config(
            enabled=self.profile_secondary_smoothing
            if enabled is None else enabled,
            gaussian_sigma=self.profile_smoothing_gaussian_sigma,
            savgol_window=self.profile_smoothing_savgol_window,
            savgol_polyorder=self.profile_smoothing_savgol_polyorder,
        )

    def _apply_state_detection_settings(self, enabled=None):
        set_state_detection_config(
            enabled=self.state_detection_enabled if enabled is None else enabled,
            gaussian_sigma=self.state_smoothing_gaussian_sigma,
            savgol_window=self.state_smoothing_savgol_window,
            savgol_polyorder=self.state_smoothing_savgol_polyorder,
            min_prominence=self.state_min_prominence,
            min_basin_width=self.state_min_basin_width,
            edge_exclusion_bins=self.state_edge_exclusion_bins,
            auto_retry_resolution=self.state_auto_retry_resolution,
            assignment_policy=self.state_assignment_policy,
        )

    def _emit_assignment_warning(self, context, state_detection):
        if state_detection.get("assignment_mode", "strict") == "strict":
            return
        fallbacks = ", ".join(state_detection.get("fallbacks_used", []))
        if not fallbacks:
            fallbacks = "downgraded assignment"
        self._emit(
            f"{context}: state assignment used "
            f"{state_detection.get('assignment_mode')} / "
            f"{state_detection.get('confidence')} confidence "
            f"({fallbacks}).",
        )

    def _allow_state_detection_retry(self):
        if not self.state_auto_retry_resolution:
            return False
        return not any(self._manual_overrides[field]
                       for field in ("step", "bins", "minpts"))

    def _guess_alpha_from_stats(self, stats):
        return stats["gap_center_q10_q90"]

    def _guess_h12_from_stats(self, stats):
        return max(20.0, 0.15 * stats["gap_span_q10_q90"])

    def _guess_step_from_stats(self, stats):
        raw_points = stats["raw_points_per_frame"]
        return max(1, int(raw_points // 400))

    def _guess_bins_from_stats(self, stats, step_guess):
        sampled_points = max(1, int(np.ceil(stats["raw_points_per_frame"] / step_guess)))
        bins = int(np.sqrt(stats["frames"] * sampled_points) / 2)
        return int(np.clip(bins, 40, 80))

    def _guess_minpts_from_stats(self, stats, step_guess, bins_guess):
        sampled_points = max(1, int(np.ceil(stats["raw_points_per_frame"] / step_guess)))
        minpts = int((sampled_points / max(1, bins_guess)) * 1.5)
        return int(np.clip(minpts, 10, 30))

    def _combine_reference_stats(self, stats_list):
        combined = {}
        for key in stats_list[0]:
            values = [stats[key] for stats in stats_list]
            if isinstance(values[0], (int, float)):
                combined[key] = float(np.mean(values))
            else:
                combined[key] = values[0]
        combined["frames"] = int(round(combined["frames"]))
        combined["raw_points_per_frame"] = int(round(
            combined["raw_points_per_frame"]))
        combined["sampled_points_per_frame"] = int(round(
            combined["sampled_points_per_frame"]))
        return combined

    def guess_fit_settings_from_ref(self, ref_pairs, apply=True):
        normalized_pairs = self._normalize_pairs(ref_pairs)
        stats_list = []
        for pair in normalized_pairs:
            A_xvgs, B_xvgs = fetchA_B_xvgs(pair[0], pair[1])
            stats_list.append(
                summarize_reference_potentials(A_xvgs,
                                               B_xvgs,
                                               reverse_isfs=self.reverse_isfs))

        combined_stats = self._combine_reference_stats(stats_list)
        step_guess = self._guess_step_from_stats(combined_stats)
        bins_guess = self._guess_bins_from_stats(combined_stats, step_guess)
        guessed = {
            "alpha_guess": self._guess_alpha_from_stats(combined_stats),
            "H12_guess": self._guess_h12_from_stats(combined_stats),
            "step": step_guess,
            "bins": bins_guess,
            "minpts": self._guess_minpts_from_stats(combined_stats, step_guess,
                                                    bins_guess),
            "reference_stats": combined_stats,
        }

        applied_fields = []
        if apply:
            for field in self._AUTO_TRACKED_FIELDS:
                if self._set_if_auto(field, guessed[field]):
                    applied_fields.append(field)
            self._last_auto_guess = guessed
            self._emit(self._format_auto_guess_summary(guessed, applied_fields))
        return guessed

    def _fit_evb_parameters(self, A_xvgs, B_xvgs):
        alpha, h12 = self._initial_guesses()
        converged = False
        last_ts = None
        last_dg = None

        for iteration in range(1, self.max_iter + 1):
            ts_energy, dg_energy, state_detection = evbmap_nooutput(
                A_xvgs,
                B_xvgs,
                self.step,
                alpha,
                h12,
                self.dl,
                self.kb,
                self.temp,
                self.bins,
                self.minpts,
                self.reverse_isfs,
                return_metadata=True,
                allow_resolution_retry=self._allow_state_detection_retry(),
            )
            last_ts = ts_energy
            last_dg = dg_energy
            self._emit_assignment_warning(f"  fit iteration {iteration}",
                                          state_detection)
            barrier_diff = self.barrier - ts_energy
            free_energy_diff = self.free_energy - dg_energy

            self._emit(
                self._format_iteration(iteration, alpha, h12, ts_energy,
                                       dg_energy))

            if abs(barrier_diff) < self.tol and abs(free_energy_diff) < self.tol:
                converged = True
                break

            slowness = self._adaptive_slowness(barrier_diff, free_energy_diff)
            if abs(barrier_diff) > self.tol:
                h12 += barrier_diff * slowness
            if abs(free_energy_diff) > self.tol:
                alpha += free_energy_diff * slowness

        return {
            "alpha": alpha,
            "H12": h12,
            "barrier": last_ts,
            "free_energy": last_dg,
            "iterations": iteration,
            "converged": converged,
        }

    def fit_alpha_h12_from_ref(self, ref_pairs):
        self.guess_fit_settings_from_ref(ref_pairs, apply=True)
        self._apply_state_detection_settings(
            enabled=self.profile_secondary_smoothing_for_fitting)
        normalized_pairs = self._normalize_pairs(ref_pairs)
        fit_results = []

        for pair in normalized_pairs:
            self._emit(f"Fitting reference pair: {pair[0]} vs {pair[1]}")
            A_xvgs, B_xvgs = fetchA_B_xvgs(pair[0], pair[1])
            fit_result = self._fit_evb_parameters(A_xvgs, B_xvgs)
            fit_results.append(fit_result)
            if self.verbose:
                status = "converged" if fit_result["converged"] else "not converged"
                self._emit(
                    f"  {status}: alpha={fit_result['alpha']:.5f}, "
                    f"H12={fit_result['H12']:.5f}, "
                    f"barrier={fit_result['barrier']:.5f}, "
                    f"dG={fit_result['free_energy']:.5f}"
                )

        self.alpha = float(np.mean([result["alpha"] for result in fit_results]))
        self.H12 = float(np.mean([result["H12"] for result in fit_results]))
        self._parameter_source = "fit"

        self._emit(
            self._format_parameter_summary(self.alpha, self.H12,
                                           len(normalized_pairs)),
        )

        if not all(result["converged"] for result in fit_results):
            self._emit(
                "Warning: at least one reference fit reached max_iter before "
                "meeting tolerance; final averaged parameters were still applied.",
            )

        return self.alpha, self.H12

    def _smoothed_profile(self, dg_arr):
        if dg_arr.shape[0] < 4:
            return np.c_[dg_arr[:, 0], dg_arr[:, -1]]
        return smooth_data(
            dg_arr[:, 0],
            dg_arr[:, -1],
            gaussian_sigma=self.profile_smoothing_gaussian_sigma,
            savgol_window=self.profile_smoothing_savgol_window,
            savgol_polyorder=self.profile_smoothing_savgol_polyorder,
        )

    def _system_summary(self, system_id, system):
        ts_mean = float(np.mean(system["ts_energy"]))
        dg_mean = float(np.mean(system["dg_energy"]))
        ts_std = float(np.std(system["ts_energy"]))
        dg_std = float(np.std(system["dg_energy"]))
        return (
            f"System {system_id}: ts={ts_mean:.2f} kJ/mol "
            f"(std {ts_std:.2f}), dG={dg_mean:.2f} kJ/mol "
            f"(std {dg_std:.2f})"
        )

    def _plot_systems(self, systems, kcal):
        plot_data_sets = []
        labels = []
        for system_id, system in systems.items():
            for pair_index, smooth_profile in enumerate(system["smooth_dg"]):
                plot_data_sets.append(smooth_profile)
                labels.append(
                    remove_underline(f"{system_id}_{pair_index}_{system['name']}"))

        if plot_data_sets:
            plot_data(
                plot_data_sets,
                labels,
                unit="kcal/mol" if kcal else "kJ/mol",
            )

    def evb_systems(self, systems, plot=True, kcal=True):
        if self.alpha is None or self.H12 is None:
            raise ValueError(
                "EVB parameters are not set. Run fit_alpha_h12_from_ref(...) "
                "or assign alpha/H12 before calling evb_systems(...)."
            )

        self._apply_state_detection_settings(
            enabled=self.state_detection_enabled)
        mapping_summary = self._format_mapping_summary()
        if mapping_summary is not None:
            self._emit(mapping_summary)

        normalized_systems = self._normalized_systems(systems)

        for system_id, system in normalized_systems.items():
            system["ts_energy"] = []
            system["dg_energy"] = []
            system["dg_arr"] = []
            system["smooth_dg"] = []
            system["state_detection"] = []

            for pair in system["AB_pairs"]:
                self._emit(
                    f"Mapping system {system_id} pair: {pair[0]} vs {pair[1]}")
                A_xvgs, B_xvgs = fetchA_B_xvgs(pair[0], pair[1])
                try:
                    ts_energy, dg_energy, dg_arr, state_detection = evbmap(
                        A_xvgs,
                        B_xvgs,
                        self.step,
                        self.alpha,
                        self.H12,
                        self.dl,
                        self.kb,
                        self.temp,
                        self.bins,
                        self.minpts,
                        f"{system_id}_{pair[0]}",
                        reverse_isfs=self.reverse_isfs,
                        return_metadata=True,
                        allow_resolution_retry=self._allow_state_detection_retry(),
                    )
                except EvbMappingError as exc:
                    self._emit(
                        f"System {system_id} state detection failed: "
                        f"{exc.__class__.__name__}",
                    )
                    raise
                system["ts_energy"].append(ts_energy)
                system["dg_energy"].append(dg_energy)
                system["dg_arr"].append(dg_arr)
                system["smooth_dg"].append(self._smoothed_profile(dg_arr))
                system["state_detection"].append(state_detection)
                self._emit_assignment_warning(
                    f"System {system_id} pair {pair[0]} vs {pair[1]}",
                    state_detection,
                )
                system["name"] = pair[0]

            self._emit(self._system_summary(system_id, system), force=True)

        self.system_data = normalized_systems
        if plot:
            self._plot_systems(normalized_systems, kcal)
        return self.system_data
