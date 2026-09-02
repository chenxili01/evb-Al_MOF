import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


def load_data(*paths):
    data = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        data.append(np.loadtxt(path, skiprows=1))
    return data


def _normalized_savgol_window(window, size, polyorder):
    if size < 5:
        return None
    window = int(window)
    polyorder = int(polyorder)
    minimum_window = polyorder + 2
    if minimum_window % 2 == 0:
        minimum_window += 1
    if window < minimum_window:
        window = minimum_window
    if window % 2 == 0:
        window += 1
    if window > size:
        window = size if size % 2 == 1 else size - 1
    if window <= polyorder or window < 3:
        return None
    return window


def secondary_smooth_data(y,
                          gaussian_sigma=1.0,
                          savgol_window=9,
                          savgol_polyorder=3):
    y = np.asarray(y, dtype=float)
    if y.size < 5:
        return y.copy()

    smoothed = y.copy()
    if gaussian_sigma and gaussian_sigma > 0:
        smoothed = gaussian_filter1d(smoothed,
                                     sigma=float(gaussian_sigma),
                                     mode="nearest")

    window = _normalized_savgol_window(savgol_window, y.size, savgol_polyorder)
    if window is not None:
        polyorder = min(int(savgol_polyorder), window - 1)
        smoothed = savgol_filter(smoothed,
                                 window_length=window,
                                 polyorder=polyorder,
                                 mode="interp")
    return smoothed


def smooth_data(x,
                y,
                gaussian_sigma=1.0,
                savgol_window=9,
                savgol_polyorder=3):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        raise ValueError("x and y must contain the same number of points.")
    if x.size < 4:
        return np.vstack((x, y)).T

    x_new = np.linspace(x.min(), x.max(), 300)
    y_smooth = secondary_smooth_data(y,
                                     gaussian_sigma=gaussian_sigma,
                                     savgol_window=savgol_window,
                                     savgol_polyorder=savgol_polyorder)
    spline = make_interp_spline(x, y_smooth, k=3)
    return np.vstack((x_new, spline(x_new))).T


def _convert_unit(values, unit):
    if unit == "kJ/mol":
        return values, r"$\Delta$G (kJ/mol)"
    if unit == "kcal/mol":
        return values / 4.184, r"$\Delta$G (kcal/mol)"
    raise ValueError("unit must be either 'kJ/mol' or 'kcal/mol'.")


def plot_data(data, label, background=None, unit="kJ/mol"):
    fig, ax = plt.subplots(figsize=(10, 6))

    if background is not None:
        if not os.path.exists(background):
            raise FileNotFoundError(f"Background image not found: {background}")
        image = mpimg.imread(background)
        ax.imshow(image, extent=[-730, 663, -3, 25], aspect="auto", zorder=-1)

    for index, series in enumerate(data):
        x_data = series[:, 0]
        y_data, ylabel = _convert_unit(series[:, -1], unit)
        line_kwargs = {"label": label[index], "linewidth": 2}
        if index == 0:
            line_kwargs.update({"linestyle": "--", "color": "black"})
        ax.plot(x_data, y_data, **line_kwargs)

    ax.set_xlabel("IS-TS")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.tick_params(axis="x", which="both", length=0)
    ax.set_xticklabels([])
    ax.legend()
    fig.savefig("evbmap.png",
                dpi=300,
                bbox_inches="tight",
                transparent=False)
    plt.close(fig)


def plot_mapped_pmf_diagnostic(output_path,
                               raw_profile,
                               detection_profile=None,
                               accepted_markers=None,
                               rejected_markers=None,
                               populations=None,
                               support_counts=None,
                               minpts=None,
                               title="Mapped EVB Profile",
                               failure_reason=None):
    raw_profile = np.asarray(raw_profile, dtype=float)
    detection_profile = (None if detection_profile is None else
                         np.asarray(detection_profile, dtype=float))
    populations = np.asarray([] if populations is None else populations, dtype=float)
    support_counts = np.asarray([] if support_counts is None else support_counts,
                                dtype=float)
    accepted_markers = [] if accepted_markers is None else accepted_markers
    rejected_markers = [] if rejected_markers is None else rejected_markers

    fig, axes = plt.subplots(2,
                             1,
                             figsize=(8, 6),
                             sharex=True,
                             gridspec_kw={"height_ratios": [3.2, 1.2]})
    ax_profile, ax_support = axes

    if raw_profile.size:
        ax_profile.plot(raw_profile[:, 0],
                        raw_profile[:, 1],
                        label="Raw PMF",
                        linewidth=1.5,
                        color="black",
                        alpha=0.75)
    if detection_profile is not None and detection_profile.size:
        ax_profile.plot(detection_profile[:, 0],
                        detection_profile[:, 1],
                        label="State-detection PMF",
                        linewidth=2.1,
                        color="tab:purple")

    for marker in accepted_markers:
        ax_profile.scatter(marker["x"],
                           marker["y"],
                           color=marker.get("color", "tab:green"),
                           marker=marker.get("marker", "o"),
                           s=50,
                           label=marker.get("label"))
    for marker in rejected_markers:
        ax_profile.scatter(marker["x"],
                           marker["y"],
                           color=marker.get("color", "tab:gray"),
                           marker=marker.get("marker", "x"),
                           s=45,
                           alpha=0.9,
                           label=marker.get("label"))

    ax_profile.set_ylabel(r"$\Delta$G (kJ/mol)")
    ax_profile.set_title(title)
    ax_profile.grid(True, alpha=0.3)
    handles, labels = ax_profile.get_legend_handles_labels()
    if labels:
        deduped = {}
        for handle, label in zip(handles, labels):
            if label and label not in deduped:
                deduped[label] = handle
        ax_profile.legend(deduped.values(), deduped.keys(), loc="best")

    if raw_profile.size and populations.size == raw_profile.shape[0]:
        ax_support.plot(raw_profile[:, 0],
                        populations,
                        linewidth=1.5,
                        color="tab:blue",
                        label="Per-bin population")
    elif populations.size:
        ax_support.plot(populations,
                        linewidth=1.5,
                        color="tab:blue",
                        label="Per-bin population")

    if raw_profile.size and support_counts.size == raw_profile.shape[0]:
        ax_support.plot(raw_profile[:, 0],
                        support_counts,
                        linewidth=1.3,
                        color="tab:orange",
                        label="Window support")
    elif support_counts.size:
        ax_support.plot(support_counts,
                        linewidth=1.3,
                        color="tab:orange",
                        label="Window support")

    if minpts is not None:
        ax_support.axhline(float(minpts),
                           color="tab:red",
                           linestyle="--",
                           linewidth=1.1,
                           label=f"minpts={minpts}")
    ax_support.set_xlabel(r"$\epsilon_1 - \epsilon_2$ (kJ/mol)")
    ax_support.set_ylabel("Support")
    ax_support.grid(True, alpha=0.3)
    handles, labels = ax_support.get_legend_handles_labels()
    if labels:
        deduped = {}
        for handle, label in zip(handles, labels):
            if label and label not in deduped:
                deduped[label] = handle
        ax_support.legend(deduped.values(), deduped.keys(), loc="best")

    if failure_reason:
        fig.suptitle(failure_reason)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=False)
    plt.close(fig)


def remove_underline(text):
    text = text.replace("_", " ").replace(".dat", "")
    text = text.replace("potentialA", "").replace("potentialB", "")
    text = text.replace(".", "")
    return text.replace("_", " ").replace(".dat", "")
