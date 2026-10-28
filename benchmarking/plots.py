"""Draw the README runtime-scaling charts from ``results/readme_scaling.json``.

    python benchmarking/benchmarking.py   # measure (writes the JSON)
    python benchmarking/plots.py          # draw light and dark PNGs

Needs only matplotlib. Each chart is three small multiples (ReliefF, SURF,
MultiSURF) on log-log axes. Colors follow the backend in every panel; the
categorical pair passed the dataviz palette validator (CVD and normal-vision
separation, contrast) in both themes, and scikit-rebate is a deliberately
muted reference series with its own marker and a direct label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarking" / "results" / "readme_scaling.json"
ALGORITHMS = ("ReliefF", "SURF", "MultiSURF")

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": {"cpu": "#2a78d6", "metal": "#eb6834", "scikit-rebate": "#898781"},
    },
    "dark": {
        "surface": "#1a1a19",
        "primary": "#f0efec",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": {"cpu": "#3987e5", "metal": "#d95926", "scikit-rebate": "#898781"},
    },
}
SERIES_STYLE = {
    "cpu": {"label": "FastSelect, CPU", "short": "CPU", "marker": "o"},
    "metal": {"label": "FastSelect, Metal (Apple GPU)", "short": "Metal", "marker": "s"},
    "scikit-rebate": {"label": "scikit-rebate", "short": "scikit-rebate", "marker": "^"},
}
CHARTS = {
    "samples": {
        "x": "n_samples",
        "title": "Runtime as the number of samples grows",
        "fixed": "100 features",
        "output": "benchmark_n_dominant_runtime",
    },
    "features": {
        "x": "n_features",
        "title": "Runtime as the number of features grows",
        "fixed": "100 samples",
        "output": "benchmark_p_dominant_runtime",
    },
}


def _size_label(value, _position=None):
    return f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"


def _seconds_label(value, _position=None):
    if value >= 1:
        return f"{value:g} s"
    if value >= 1e-3:
        return f"{value * 1e3:g} ms"
    return f"{value * 1e6:g} µs"


def _series(data, scenario, algorithm):
    """``{key: (sizes, seconds)}`` for one panel, keyed by backend or library."""
    x_name = CHARTS[scenario]["x"]
    panel = {}
    for entry in data["series"]:
        if entry["algorithm"] != algorithm:
            continue
        key = entry["backend"] if entry["library"] == "fast-select" else "scikit-rebate"
        points = sorted(
            (point[x_name], point["median_seconds"]) for point in entry["points"] if point["scenario"] == scenario
        )
        panel[key] = ([size for size, _ in points], [seconds for _, seconds in points])
    return panel


def _spread_labels(positions, minimum_gap):
    """Nudge sorted label positions (axes fraction) apart by at least ``minimum_gap``."""
    order = sorted(range(len(positions)), key=positions.__getitem__)
    placed = list(positions)
    for previous, current in zip(order, order[1:]):
        placed[current] = max(placed[current], placed[previous] + minimum_gap)
    return placed


def _draw(data, scenario, theme_name, output_dir):
    theme = THEMES[theme_name]
    chart = CHARTS[scenario]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.edgecolor": theme["axis"],
            "axes.labelcolor": theme["secondary"],
            "xtick.color": theme["axis"],
            "ytick.color": theme["axis"],
            "xtick.labelcolor": theme["secondary"],
            "ytick.labelcolor": theme["secondary"],
            "text.color": theme["primary"],
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(8.4, 3.35), sharey=True, dpi=300)
    figure.patch.set_facecolor(theme["surface"])
    all_seconds = []
    for axis, algorithm in zip(axes, ALGORITHMS):
        panel = _series(data, scenario, algorithm)
        axis.set_facecolor(theme["surface"])
        for key in ("scikit-rebate", "metal", "cpu"):  # FastSelect CPU drawn on top
            sizes, seconds = panel[key]
            all_seconds += seconds
            axis.plot(
                sizes,
                seconds,
                color=theme["series"][key],
                linewidth=1.5,
                marker=SERIES_STYLE[key]["marker"],
                markersize=5.2,
                markeredgecolor=theme["surface"],
                markeredgewidth=1.0,
                solid_capstyle="round",
                zorder=3 if key == "cpu" else 2,
                clip_on=False,
            )
        # Bracket the gap to scikit-rebate at the largest size: the headline number.
        largest = panel["cpu"][0][-1]
        fast, reference = panel["cpu"][1][-1], panel["scikit-rebate"][1][-1]
        axis.plot([largest * 1.18] * 2, [fast, reference], color=theme["muted"], linewidth=0.8, zorder=1)
        for value in (fast, reference):
            axis.plot([largest * 1.12, largest * 1.18], [value] * 2, color=theme["muted"], linewidth=0.8)
        middle = (fast * reference) ** 0.5
        axis.annotate(
            f"{reference / fast:,.0f}×",
            xy=(largest * 1.27, middle),
            xytext=(0, 1.5),
            textcoords="offset points",
            va="bottom",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
            color=theme["primary"],
        )
        axis.annotate(
            "faster",
            xy=(largest * 1.27, middle),
            xytext=(0, -1.5),
            textcoords="offset points",
            va="top",
            ha="left",
            fontsize=7.5,
            color=theme["secondary"],
        )
        axis.set_title(algorithm, loc="left", fontsize=9.5, fontweight="bold", color=theme["primary"], pad=6)
        axis.set_xscale("log")
        axis.set_yscale("log")
        sizes = panel["cpu"][0]
        axis.set_xlim(sizes[0] / 1.25, sizes[-1] * 3.3)
        axis.xaxis.set_major_locator(FixedLocator(sizes))
        axis.xaxis.set_major_formatter(FuncFormatter(_size_label))
        axis.xaxis.set_minor_locator(NullLocator())
        axis.yaxis.set_major_locator(LogLocator(base=10))
        axis.yaxis.set_major_formatter(FuncFormatter(_seconds_label))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.yaxis.set_minor_locator(NullLocator())
        axis.grid(axis="y", which="major", color=theme["grid"], linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_linewidth(0.6)
        axis.tick_params(length=2.5, width=0.6, labelsize=7.5)
        axis.set_xlabel(chart["x"].replace("n_", "").capitalize(), fontsize=8, labelpad=3)

    axes[0].set_ylabel("Median fit time (log scale)", fontsize=8, labelpad=4)
    axes[0].set_ylim(min(all_seconds) / 1.8, max(all_seconds) * 2.2)

    handles = [
        plt.Line2D(
            [],
            [],
            color=theme["series"][key],
            marker=SERIES_STYLE[key]["marker"],
            markersize=5.2,
            markeredgecolor=theme["surface"],
            linewidth=1.5,
            label=SERIES_STYLE[key]["label"],
        )
        for key in ("cpu", "metal", "scikit-rebate")
    ]
    provenance = data["provenance"]
    figure.text(0.012, 0.975, chart["title"], fontsize=12, fontweight="bold", color=theme["primary"], va="top")
    figure.text(
        0.012,
        0.905,
        f"{chart['fixed']}, {provenance['n_features_to_select']} selected; lower is better. Brackets compare "
        f"FastSelect CPU with scikit-rebate at the largest size.\nApple M1 Pro, 8 threads. FastSelect "
        f"{provenance['git_commit'][:7]}; scikit-rebate {provenance['skrebate']}, faster of n_jobs = 1 or all cores. "
        f"Median of {data['rounds']} round{'s' if data['rounds'] != 1 else ''}.",
        fontsize=7.5,
        color=theme["secondary"],
        va="top",
        linespacing=1.35,
    )
    legend = figure.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.005, 0.815),
        ncol=3,
        frameon=False,
        fontsize=8,
        handlelength=2.2,
        columnspacing=1.6,
    )
    for text in legend.get_texts():
        text.set_color(theme["secondary"])
    figure.subplots_adjust(left=0.085, right=0.985, top=0.655, bottom=0.14, wspace=0.12)
    suffix = "" if theme_name == "light" else "_dark"
    output = output_dir / f"{chart['output']}{suffix}.png"
    figure.savefig(output, dpi=300, facecolor=theme["surface"])
    plt.close(figure)
    print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RESULTS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmarking")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    for scenario in CHARTS:
        for theme_name in THEMES:
            _draw(data, scenario, theme_name, args.output_dir)


if __name__ == "__main__":
    main()
