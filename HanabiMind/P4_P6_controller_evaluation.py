from __future__ import annotations

from typing import Dict, List, Optional
import os

import numpy as np

from P0_P1_project_config import HanabiConfig
from P0_P1_environment import PettingZooHanabiRunner
from P1_P5_foundation_checks import _evaluate_controller
from P1_P5_P6_P8_P10_controllers import (
    HeuristicController,
    MaximaxController,
    MaximaxStats,
    RandomController,
)


PNG_DIR = "png"


def _score_stats(scores: List[int]) -> Dict[str, object]:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return {"scores": [], "mean": 0.0, "std": 0.0, "variance": 0.0,
                "min": 0, "max": 0, "n": 0}
    std = float(arr.std(ddof=0))
    return {
        "scores": [int(s) for s in scores],
        "mean":   float(arr.mean()),
        "std":    std,
        "variance": float(std * std),
        "min":    int(arr.min()),
        "max":    int(arr.max()),
        "n":      int(arr.size),
    }


def run_phase4_eval(
    num_games: int = 30,
    seed: int = 0,
    plot_path: Optional[str] = os.path.join(PNG_DIR, "phase4_controller_scores.png"),
    cfg: Optional[HanabiConfig] = None,
    verbose: bool = True,
    debug_belief: bool = False,
    maximax_samples: int = 80,
) -> Dict[str, Dict[str, object]]:
    cfg = cfg or HanabiConfig()
    runner = PettingZooHanabiRunner(cfg)

    def _make_maximax(i):
        return MaximaxController(
            f"max{i}", cfg,
            num_samples=maximax_samples,
            debug=False,
        )

    configs = [
        ("RandomController",  lambda i: RandomController(f"rand{i}", cfg)),
        ("MaximaxController", _make_maximax),
    ]

    results: Dict[str, Dict[str, object]] = {}
    if verbose:
        print(f"[Phase 4] evaluating {len(configs)} controllers "
              f"over {num_games} games each ({cfg.players} players)")
        if debug_belief:
            print(f"         belief-debug ON  num_samples={maximax_samples}")

    for label, factory in configs:
        if verbose:
            print(f"  running {label} ...")

        agg = MaximaxStats() if (debug_belief and label == "MaximaxController") else None

        def _hook(controllers, _score, _agg=agg):
            if _agg is None:
                return
            for ctrl in controllers.values():
                if isinstance(ctrl, MaximaxController):
                    _agg.merge(ctrl.stats)

        raw = _evaluate_controller(
            cfg, runner,
            make_controller=factory,
            num_episodes=num_games,
            seed=seed,
            post_episode=_hook if agg is not None else None,
        )
        stats = _score_stats(raw["scores"])
        results[label] = stats
        if verbose:
            print(
                f"    {label:<20s} mean={stats['mean']:.2f}  "
                f"std={stats['std']:.2f}  min={stats['min']}  "
                f"max={stats['max']}  n={stats['n']}"
            )
        if agg is not None:
            print("    " + agg.summary().replace("\n", "\n    "))
            results[label]["belief_diagnostics"] = {
                "decisions":               agg.decisions,
                "marginal_fallback_rate":  agg.decisions_marginal_fallback / max(1, agg.decisions),
                "avg_samples_per_decision": agg.samples_drawn / max(1, agg.decisions_with_samples),
                "avg_unique_per_decision": agg.unique_samples_sum / max(1, agg.decisions_with_samples),
                "rejection_rate":          agg.rejection_samples / max(1, agg.samples_drawn),
                "greedy_rate":             agg.greedy_samples / max(1, agg.samples_drawn),
                "belief":                  agg.belief,
            }

    if plot_path:
        plot_phase4_results(results, plot_path, cfg)
    return results


def plot_phase4_results(
    results: Dict[str, Dict[str, object]],
    plot_path: str,
    cfg: HanabiConfig,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Phase 4] matplotlib not installed; skipping plot")
        return

    labels = list(results.keys())
    means = [float(results[l]["mean"]) for l in labels]
    stds  = [float(results[l]["std"])  for l in labels]
    max_score = cfg.colors * cfg.ranks

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    x = np.arange(len(labels))
    ax.bar(
        x, means, yerr=stds, capsize=8,
        color=["#8ea8d8", "#d89a6a"][: len(labels)],
        edgecolor="black", linewidth=1.0,
    )
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + 0.15, f"{m:.2f} +/- {s:.2f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Final fireworks score")
    ax.set_ylim(0, max_score + 2)
    ax.axhline(max_score, color="gray", linestyle="--", linewidth=0.8,
               alpha=0.5, label=f"max ({max_score})")
    ax.grid(True, axis="y", alpha=0.3)

    any_stats = next(iter(results.values()), {})
    n = any_stats.get("n", "?")
    ax.set_title(f"Phase 4 - controller comparison (N={n} games)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    parent = os.path.dirname(os.path.abspath(plot_path)) or "."
    os.makedirs(parent, exist_ok=True)
    fig.savefig(plot_path, dpi=120)
    print(f"[Phase 4] bar chart saved -> {plot_path}")
    plt.close(fig)


def run_phase6_compare(
    num_games: int = 20,
    seed: int = 0,
    plot_path: Optional[str] = os.path.join(PNG_DIR, "phase6_controller_scores.png"),
    cfg: Optional[HanabiConfig] = None,
    verbose: bool = True,
    maximax_samples: int = 80,
    lookahead_depth: int = 2,
) -> Dict[str, Dict[str, object]]:
    cfg = cfg or HanabiConfig()
    runner = PettingZooHanabiRunner(cfg)

    deep_label = f"MaximaxController (depth={lookahead_depth})"

    def _make_maximax_shallow(i):
        return MaximaxController(
            f"max1_{i}", cfg,
            num_samples=maximax_samples,
            lookahead_depth=1,
            debug=False,
        )

    def _make_maximax_deep(i):
        return MaximaxController(
            f"maxD_{i}", cfg,
            num_samples=maximax_samples,
            lookahead_depth=lookahead_depth,
            debug=False,
        )

    configs = [
        ("RandomController",            lambda i: RandomController(f"rand{i}", cfg)),
        ("HeuristicController",         lambda i: HeuristicController(f"heur{i}", cfg)),
        ("MaximaxController (depth=1)", _make_maximax_shallow),
        (deep_label,                    _make_maximax_deep),
    ]

    results: Dict[str, Dict[str, object]] = {}
    if verbose:
        print(
            f"[Phase 6] evaluating {len(configs)} controllers over "
            f"{num_games} games each ({cfg.players} players)  "
            f"samples={maximax_samples}  lookahead_depth={lookahead_depth}"
        )

    for label, factory in configs:
        if verbose:
            print(f"  running {label} ...")
        raw = _evaluate_controller(
            cfg, runner,
            make_controller=factory,
            num_episodes=num_games,
            seed=seed,
        )
        stats = _score_stats(raw["scores"])
        results[label] = stats
        if verbose:
            print(
                f"    {label:<32s} mean={stats['mean']:.2f}  "
                f"std={stats['std']:.2f}  min={stats['min']}  "
                f"max={stats['max']}  n={stats['n']}"
            )

    if plot_path:
        plot_phase6_results(results, plot_path, cfg)
    return results


def plot_phase6_results(
    results: Dict[str, Dict[str, object]],
    plot_path: str,
    cfg: HanabiConfig,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Phase 6] matplotlib not installed; skipping plot")
        return

    labels = list(results.keys())
    means = [float(results[l]["mean"]) for l in labels]
    stds  = [float(results[l]["std"])  for l in labels]
    max_score = cfg.colors * cfg.ranks

    palette = ["#8ea8d8", "#a3c8a3", "#d89a6a", "#c8689e", "#8c8ad8"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(labels))
    ax.bar(
        x, means, yerr=stds, capsize=8,
        color=palette[: len(labels)],
        edgecolor="black", linewidth=1.0,
    )
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + 0.15, f"{m:.2f} +/- {s:.2f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=9)
    ax.set_ylabel("Final fireworks score")
    ax.set_ylim(0, max_score + 2)
    ax.axhline(max_score, color="gray", linestyle="--", linewidth=0.8,
               alpha=0.5, label=f"max ({max_score})")
    ax.grid(True, axis="y", alpha=0.3)

    any_stats = next(iter(results.values()), {})
    n = any_stats.get("n", "?")
    ax.set_title(f"Phase 6 - lookahead + heuristic comparison (N={n} games)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    parent = os.path.dirname(os.path.abspath(plot_path)) or "."
    os.makedirs(parent, exist_ok=True)
    fig.savefig(plot_path, dpi=120)
    print(f"[Phase 6] bar chart saved -> {plot_path}")
    plt.close(fig)
