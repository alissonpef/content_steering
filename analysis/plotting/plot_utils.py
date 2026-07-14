import logging
import math
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

logger = logging.getLogger("plot_utils")
CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_GREEN = "#009E73"
CB_RED = "#D55E00"
CB_PURPLE = "#882255"
CB_CYAN = "#44AA99"
CB_MAGENTA = "#CC79A7"
CB_GREY = "#999999"
CB_BLACK = "#000000"
SERVER_PALETTE = {
    "delivery-node-1": CB_GREEN,
    "delivery-node-2": CB_ORANGE,
    "delivery-node-3": CB_BLUE,
}
SERVER_PALETTE_LIGHT = {
    "delivery-node-1": "#009E7340",
    "delivery-node-2": "#E69F0040",
    "delivery-node-3": "#0072B240",
}
SERVER_LABELS = {
    "delivery-node-1": "Node 1 — Brazil",
    "delivery-node-2": "Node 2 — Chile",
    "delivery-node-3": "Node 3 — Colombia",
    "DynamicBest": "Oracle Optimal",
    "N/A": "N/A",
}
STRATEGY_STYLE = {
    "linucb": {
        "label": "LinUCB (contextual)",
        "color": CB_RED,
        "linewidth": 1.6,
        "linestyle": "-",
        "zorder": 10,
        "marker": None,
        "alpha": 0.95,
    },
    "ucb1": {
        "label": "UCB1",
        "color": CB_GREEN,
        "linewidth": 1.6,
        "linestyle": "-",
        "zorder": 5,
        "marker": None,
        "alpha": 0.85,
    },
    "epsilon_greedy": {
        "label": r"$\varepsilon$-Greedy",
        "color": CB_CYAN,
        "linewidth": 1.6,
        "linestyle": "-",
        "zorder": 5,
        "marker": None,
        "alpha": 0.85,
    },
    "random": {
        "label": "Random",
        "color": CB_GREY,
        "linewidth": 2.2,
        "linestyle": "-",
        "zorder": 3,
        "marker": None,
        "alpha": 0.70,
    },
    "round_robin": {
        "label": "Round Robin",
        "color": "#8b4513",
        "linewidth": 1.6,
        "linestyle": "-",
        "zorder": 4,
        "marker": None,
        "alpha": 0.75,
    },
    "ppo_hybrid": {
        "label": "PPO Hybrid",
        "color": CB_PURPLE,
        "linewidth": 1.6,
        "linestyle": "-",
        "zorder": 7,
        "marker": None,
        "alpha": 0.90,
    },
    "best": {
        "label": "Perfect (Oracle)",
        "color": CB_BLACK,
        "linewidth": 2.2,
        "linestyle": "-",
        "zorder": 35,
        "marker": None,
        "alpha": 0.85,
    },
    "thompson_sampling": {
        "label": "Thompson Sampling",
        "color": CB_BLUE,
        "linewidth": 1.6,
        "linestyle": "-",
        "zorder": 6,
        "marker": None,
        "alpha": 0.90,
    },
    "no_steering": {
        "label": "No Steering",
        "color": CB_GREY,
        "linewidth": 1.4,
        "linestyle": "-.",
        "zorder": 2,
        "marker": None,
        "alpha": 0.70,
    },
    "default": {
        "label": "Unknown",
        "color": CB_GREY,
        "linewidth": 1.2,
        "linestyle": "-",
        "zorder": 1,
        "marker": None,
        "alpha": 0.60,
    },
}
STRATEGY_LEGEND_ORDER = [
    "best",
    "ppo_hybrid",
    "linucb",
    "ucb1",
    "epsilon_greedy",
    "thompson_sampling",
    "round_robin",
    "random",
    "no_steering",
]
KNOWN_SERVER_KEYS_UNDERSCORE = [
    "delivery_node_1",
    "delivery_node_2",
    "delivery_node_3",
]

KNOWN_STRATEGY_KEYS = [
    "best",
    "ppo_hybrid",
    "epsilon_greedy",
    "no_steering",
    "linucb",
    "ucb1",
    "thompson_sampling",
    "round_robin",
    "random",
]


def apply_global_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update(
        {
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "figure.dpi": 150,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.title_fontsize": 10,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.45,
            "grid.linestyle": "--",
            "legend.frameon": True,
            "legend.framealpha": 0.85,
            "legend.edgecolor": "0.7",
            "legend.fancybox": True,
            "legend.borderpad": 0.4,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "mathtext.fontset": "dejavusans",
        }
    )


def get_strategy_style(strategy_key: str) -> dict:
    return STRATEGY_STYLE.get(strategy_key, STRATEGY_STYLE["default"])


def get_strategy_display_name(strategy_key: str) -> str:
    return get_strategy_style(strategy_key)["label"]


def get_server_color(server_name_hyphen: str) -> str:
    return SERVER_PALETTE.get(server_name_hyphen, CB_GREY)


def get_server_label(server_name_hyphen: str) -> str:
    return SERVER_LABELS.get(server_name_hyphen, server_name_hyphen.replace("-", " ").title())


def extract_strategy_from_filename(filename_no_ext: str) -> str:
    name = filename_no_ext
    if name.startswith("log_"):
        name = name[4:]
    if name.endswith("_average"):
        name = name[: -len("_average")]
    for key in sorted(KNOWN_STRATEGY_KEYS, key=len, reverse=True):
        if name == key or name.startswith(key + "_"):
            return key
    m = re.match(r"([a-zA-Z0-9_]+)", name)
    if not m:
        return "Unknown"
    candidate = m.group(1)
    return candidate if candidate in KNOWN_STRATEGY_KEYS else "Unknown"


def format_axes(
    ax,
    title: str,
    xlabel: str,
    ylabel: str,
    *,
    legend_loc: str = "best",
    xlim_max: float | None = None,
    y_log_scale: bool = False,
    custom_handles=None,
    custom_labels=None,
):
    ax.set_title(title, pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim_max is not None and ax.has_data():
        ax.set_xlim(left=0, right=xlim_max)
        ax.set_xticks(np.arange(0, xlim_max + 1, 15))
    elif ax.has_data():
        lo, hi = ax.get_xlim()
        start = max(0, math.floor(lo / 15) * 15)
        ax.set_xticks(np.arange(start, hi + 1, 15))
    if y_log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    elif ax.has_data():
        ax.set_ylim(bottom=0)
    if custom_handles and custom_labels:
        ax.legend(custom_handles, custom_labels, loc=legend_loc)
    else:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc=legend_loc)
    ax.grid(True, which="major")
    if y_log_scale and ax.has_data():
        ax.grid(True, which="minor", alpha=0.2)


def sort_legend_by_strategy(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    label_to_key = {}
    for key, sty in STRATEGY_STYLE.items():
        label_to_key[sty["label"]] = key

    def _sort_key(lbl):
        k = label_to_key.get(lbl, "zzz")
        if k in STRATEGY_LEGEND_ORDER:
            return STRATEGY_LEGEND_ORDER.index(k)
        return len(STRATEGY_LEGEND_ORDER)

    paired = sorted(zip(labels, handles, strict=False), key=lambda t: _sort_key(t[0]))
    if paired:
        labels_s, handles_s = zip(*paired, strict=False)
        legend = ax.get_legend()
        if not kwargs:
            legend_loc = getattr(legend, "_loc", "best") if legend else "best"
            kwargs = {"loc": legend_loc}
        ax.legend(handles_s, labels_s, **kwargs)


SAVE_FORMATS = (".png", ".pdf", ".svg")


def save_figure(fig, path_without_ext: str, formats=SAVE_FORMATS, close: bool = True):
    directory = os.path.dirname(path_without_ext)
    if directory:
        os.makedirs(directory, exist_ok=True)
    for ext in formats:
        out_path = path_without_ext + ext
        try:
            fig.savefig(out_path)
            logger.debug(f"Saved: {out_path}")
        except Exception as exc:
            logger.error(f"Failed to save {out_path}: {exc}")
    if close:
        plt.close(fig)


def configure_logger(logger_instance, verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    if not logger_instance.handlers:
        handler = logging.StreamHandler()
        logger_instance.addHandler(handler)
    logger_instance.setLevel(level)


def parse_json_column(series, prefix: str = ""):
    import json

    import pandas as pd

    rows, indices = [], []
    for idx, raw in series.dropna().items():
        try:
            d = json.loads(raw) if isinstance(raw, str) else {}
            if isinstance(d, dict):
                rows.append({f"{prefix}{k.replace('-', '_')}": v for k, v in d.items()})
                indices.append(idx)
        except json.JSONDecodeError, TypeError:
            pass
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, index=indices)


def extract_scenario_from_filename(filename_no_ext: str) -> str:

    name = filename_no_ext
    if name.startswith("log_"):
        name = name[4:]
    if name.endswith("_average"):
        name = name[: -len("_average")]
    for key in sorted(KNOWN_STRATEGY_KEYS, key=len, reverse=True):
        if name == key:
            return "default"
        if name.startswith(key + "_"):
            return name[len(key) + 1 :]
    m = re.match(r"([a-zA-Z0-9_]+)", name)
    if not m:
        return "Unknown"
    candidate = m.group(1)
    if candidate in KNOWN_STRATEGY_KEYS:
        return "default"
    return "Unknown"
