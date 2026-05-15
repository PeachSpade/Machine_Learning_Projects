from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import os
import pickle
import time

import numpy as np

from P0_P1_project_config import HanabiConfig
from P0_P1_environment import PettingZooHanabiRunner
from P1_P5_P6_P8_P10_controllers import (
    MLGuidanceStats,
    MaximaxController,
    ROLLOUT_WEIGHTS,
)
from P2_P7_trajectory_schema import (
    CONTROLLER_LABELS,
    LABEL_TO_INDEX,
    LABEL_TO_PLAYSTYLE,
    PLAYSTYLE_ORDER,
    TrajectoryStep,
)
from P2_P7_ml_training import (
    GRUClassifier,
    LSTMClassifier,
    PHASE2_GRU_MODEL_PATH,
    PHASE7_MODEL_PATHS,
    TransformerClassifier,
    feature_vector_size,
    step_to_features,
    torch,
)
from P4_P6_controller_evaluation import _score_stats


PNG_DIR = "png"

DEFAULT_ML_MODEL_PATHS: Dict[str, str] = {
    "gru": PHASE7_MODEL_PATHS["GRU"],
    "lstm": PHASE7_MODEL_PATHS["LSTM"],
    "transformer": PHASE7_MODEL_PATHS["Transformer"],
}

LEGACY_ML_MODEL_PATHS: Dict[str, List[str]] = {
    "gru": [PHASE2_GRU_MODEL_PATH, "phase2_gru.pt", "phase7_gru.pt"],
    "lstm": ["phase7_lstm.pt"],
    "transformer": ["phase7_transformer.pt"],
}


def _phase8_resolve_path(model_type: str, path: Optional[str]) -> Optional[str]:
    key = _phase8_model_key(model_type)
    candidates: List[str] = []
    if path:
        candidates.append(path)
    if DEFAULT_ML_MODEL_PATHS.get(key):
        candidates.append(DEFAULT_ML_MODEL_PATHS[key])
    candidates.extend(LEGACY_ML_MODEL_PATHS.get(key, []))
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def _phase8_model_key(name: str) -> str:
    key = str(name).strip().lower()
    return {"transformers": "transformer"}.get(key, key)


if torch is not None:

    class MLInferenceWrapper:
        def __init__(
            self,
            model,
            cfg: HanabiConfig,
            label_map: Optional[Dict[str, int]] = None,
            device: Optional[object] = None,
        ) -> None:
            self.cfg = cfg
            self.label_map = dict(label_map or LABEL_TO_INDEX)
            self.num_classes = len(self.label_map)
            self.device = device if device is not None else torch.device("cpu")
            self.model = model
            self.model.to(self.device)
            self.model.eval()
            self._feature_dim = feature_vector_size(cfg)

        def _sequence_to_features(self, steps: List[TrajectoryStep]) -> np.ndarray:
            if not steps:
                return np.zeros((0, self._feature_dim), dtype=np.float32)
            return np.stack(
                [step_to_features(s, self.cfg) for s in steps], axis=0,
            ).astype(np.float32)

        def predict_proba_batch(
            self, sequences: List[List[TrajectoryStep]],
        ) -> np.ndarray:
            if not sequences:
                return np.zeros((0, self.num_classes), dtype=np.float32)

            feats_list = [self._sequence_to_features(s) for s in sequences]
            raw_lengths = [max(1, f.shape[0]) for f in feats_list]
            max_T = max(raw_lengths)
            B = len(feats_list)
            F = self._feature_dim

            padded = torch.zeros((B, max_T, F), dtype=torch.float32)
            for i, f in enumerate(feats_list):
                if f.shape[0] > 0:
                    padded[i, : f.shape[0]] = torch.from_numpy(f)
            padded = padded.to(self.device)
            lengths = torch.tensor(raw_lengths, dtype=torch.long)

            with torch.no_grad():
                logits = self.model(padded, lengths=lengths)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            return probs.astype(np.float32)

        def predict_proba(self, steps: List[TrajectoryStep]) -> np.ndarray:
            return self.predict_proba_batch([list(steps)])[0]

        def prob_of_class(
            self, steps: List[TrajectoryStep], class_name: str = "maximax",
        ) -> float:
            probs = self.predict_proba(steps)
            idx = self.label_map.get(class_name, 0)
            idx = max(0, min(int(idx), probs.shape[0] - 1))
            return float(probs[idx])


    def _build_phase7_module(
        arch: str,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
    ):
        key = _phase8_model_key(arch)
        if key == "gru":
            return GRUClassifier(input_dim, hidden_dim, num_classes)
        if key == "lstm":
            if LSTMClassifier is None:
                raise RuntimeError("LSTMClassifier unavailable (PyTorch build issue)")
            return LSTMClassifier(input_dim, hidden_dim, num_classes)
        if key == "transformer":
            if TransformerClassifier is None:
                raise RuntimeError("TransformerClassifier unavailable (PyTorch build issue)")
            return TransformerClassifier(input_dim, hidden_dim, num_classes)
        raise ValueError(f"unknown ML model arch: {arch!r}")


    def load_phase7_model(
        model_type: str,
        path: Optional[str] = None,
        cfg: Optional[HanabiConfig] = None,
        hidden_dim: int = 64,
        num_classes: int = len(CONTROLLER_LABELS),
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required for Phase 8")
        cfg = cfg or HanabiConfig()
        input_dim = feature_vector_size(cfg)
        key = _phase8_model_key(model_type)
        resolved = _phase8_resolve_path(model_type, path)
        if resolved is None:
            primary = path or DEFAULT_ML_MODEL_PATHS.get(key)
            raise FileNotFoundError(
                f"Phase 7 weights for '{model_type}' not found at {primary!r}. "
                f"Run `python \"<script>\" --phase7` first, or pass --phase8-train "
                f"to auto-train before Phase 8."
            )
        path = resolved

        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")

        arch = key
        state = ckpt
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            arch = _phase8_model_key(ckpt.get("arch", key))
            hidden_dim = int(ckpt.get("hidden_dim", hidden_dim))
            num_classes = int(ckpt.get("num_classes", num_classes))
            input_dim = int(ckpt.get("input_dim", input_dim))
            state = ckpt["state_dict"]

        model = _build_phase7_module(arch, input_dim, hidden_dim, num_classes)
        model.load_state_dict(state)
        model.eval()
        return model


    def load_phase8_wrapper(
        model_type: str,
        path: Optional[str] = None,
        cfg: Optional[HanabiConfig] = None,
        device: Optional[object] = None,
    ) -> "MLInferenceWrapper":
        cfg = cfg or HanabiConfig()
        model = load_phase7_model(model_type, path=path, cfg=cfg)
        return MLInferenceWrapper(model, cfg, device=device)

else:
    MLInferenceWrapper = None
    load_phase7_model = None
    load_phase8_wrapper = None


def _phase8_maximax_factory(
    wrapper,
    ml_weight: float,
    samples: int,
    lookahead_depth: int,
    ml_debug: bool,
):
    def _factory(cfg: HanabiConfig):
        def _make(idx: int, tag: str = "") -> MaximaxController:
            return MaximaxController(
                f"maxml{tag}_{idx}", cfg,
                num_samples=samples,
                lookahead_depth=lookahead_depth,
                ml_model=wrapper,
                ml_weight=ml_weight,
                ml_debug=ml_debug,
                debug=False,
            )
        return _make
    return _factory


def _evaluate_controller_logged(
    cfg: HanabiConfig,
    runner: "PettingZooHanabiRunner",
    make_controller,
    num_episodes: int,
    seed: int,
    post_episode=None,
) -> Dict[str, float]:
    scores: List[int] = []
    for ep in range(num_episodes):
        controllers = {
            f"player_{i}": make_controller(i) for i in range(cfg.players)
        }
        out = runner.run_episode(
            controllers, seed=seed + ep, log_trajectory=True,
        )
        scores.append(int(out["score"]))
        if post_episode is not None:
            post_episode(controllers, int(out["score"]))
    return {
        "mean": sum(scores) / max(1, len(scores)),
        "min": min(scores) if scores else 0,
        "max": max(scores) if scores else 0,
        "scores": scores,
    }


def _phase10_5_model_column(label: str) -> str:
    s = label.lower()
    if "baseline" in s:
        return "Baseline"
    for arch in ("gru", "lstm", "transformer"):
        if arch in s:
            return arch.upper()
    return label[:18]


def build_phase10_5_table_rows(
    results: Dict[str, Dict[str, object]],
    *,
    rollout: str,
    ml_weight: float,
    depth: int,
    seed: int,
    num_games: int,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for label, st in results.items():
        if not isinstance(st, dict):
            continue
        st_d = st
        var = st_d.get("variance")
        if var is None:
            sd = float(st_d.get("std", 0.0))
            var = sd * sd
        out.append({
            "config_label": str(rollout),
            "ml_weight": float(ml_weight),
            "depth": int(depth),
            "model": _phase10_5_model_column(str(label)),
            "mean": float(st_d.get("mean", 0.0)),
            "std": float(st_d.get("std", 0.0)),
            "variance": float(var),
            "min": int(st_d.get("min", 0)),
            "max": int(st_d.get("max", 0)),
            "n": int(st_d.get("n", 0)),
            "time_s": float(st_d.get("wall_time_s", 0.0)),
            "seed": int(seed),
            "num_games": int(num_games),
        })
    return out


def print_phase10_5_table(rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    print()
    print("=" * 100)
    r0 = rows[0]
    print(
        "Phase 10.5 - Config | ML weight | depth | model | mean | std | min | max | time_s"
    )
    print(
        f"  (same seed sequence: base_seed={r0.get('seed')!r}  "
        f"games_per_controller={r0.get('num_games')!r}  "
        f"variance=std^2)"
    )
    print("=" * 100)
    h = (
        f"{'Config':<8}{'MLw':>6}{'D':>4}{'Model':<12}{'Mean':>8}{'Std':>8}"
        f"{'Min':>5}{'Max':>5}{'t_s':>8}"
    )
    print(h)
    print("-" * 100)
    for r in rows:
        line = (
            f"{r.get('config_label', ''):<8}"
            f"{r.get('ml_weight', 0.0):>6.1f}"
            f"{r.get('depth', 0):>4d}"
            f"{str(r.get('model', '')):<12}"
            f"{r.get('mean', 0.0):>8.2f}{r.get('std', 0.0):>8.2f}"
            f"{r.get('min', 0):>5d}{r.get('max', 0):>5d}"
            f"{r.get('time_s', 0.0):>8.1f}"
        )
        print(line)
    print("=" * 100)
    print()


def run_phase8_compare(
    num_games: int = 30,
    seed: int = 0,
    plot_path: Optional[str] = os.path.join(PNG_DIR, "phase8_controller_scores.png"),
    cfg: Optional[HanabiConfig] = None,
    verbose: bool = True,
    maximax_samples: int = 80,
    lookahead_depth: int = 1,
    ml_weight: float = 0.1,
    ml_debug: bool = False,
    model_paths: Optional[Dict[str, str]] = None,
    include_archs: Tuple[str, ...] = ("GRU", "LSTM", "Transformer"),
    report_label: str = "Phase 8",
    baseline_name: str = "Maximax (baseline)",
    ml_name_template: str = "Maximax + {arch}",
    rollout_weights: Optional[Dict[str, float]] = None,
    hint_score_tune: Optional[Dict[str, float]] = None,
    rollout_config_label: str = "default",
) -> Dict[str, Dict[str, object]]:
    if torch is None:
        raise RuntimeError("Phase 8 requires PyTorch")

    cfg = cfg or HanabiConfig()
    runner = PettingZooHanabiRunner(cfg)
    rw = dict(rollout_weights) if rollout_weights is not None else dict(ROLLOUT_WEIGHTS)

    model_paths = dict(model_paths or DEFAULT_ML_MODEL_PATHS)
    wrappers: Dict[str, object] = {}
    for arch in include_archs:
        key = _phase8_model_key(arch)
        hint = model_paths.get(key) or DEFAULT_ML_MODEL_PATHS.get(key)
        resolved = _phase8_resolve_path(arch, hint)
        if resolved is None:
            if verbose:
                print(
                    f"[{report_label}] WARN: skipping {arch} "
                    f"(no weights found at {hint!r})"
                )
            continue
        try:
            wrappers[arch] = load_phase8_wrapper(arch, path=resolved, cfg=cfg)
            if verbose:
                print(f"[{report_label}] loaded {arch} from {resolved}")
        except Exception as exc:
            print(f"[{report_label}] WARN: skipping {arch}: {exc}")

    def _make_baseline(i: int) -> MaximaxController:
        return MaximaxController(
            f"max_base_{i}", cfg,
            num_samples=maximax_samples,
            lookahead_depth=lookahead_depth,
            ml_model=None,
            rollout_weights=rw,
            hint_score_tune=hint_score_tune,
            debug=False,
        )

    def _make_ml(arch: str):
        wrapper = wrappers[arch]
        def _factory(i: int) -> MaximaxController:
            return MaximaxController(
                f"max_{arch.lower()}_{i}", cfg,
                num_samples=maximax_samples,
                lookahead_depth=lookahead_depth,
                ml_model=wrapper,
                ml_weight=ml_weight,
                ml_debug=ml_debug,
                rollout_weights=rw,
                hint_score_tune=hint_score_tune,
                debug=False,
            )
        return _factory

    configs: List[Tuple[str, object]] = [(baseline_name, _make_baseline)]
    for arch in include_archs:
        if arch in wrappers:
            configs.append((ml_name_template.format(arch=arch), _make_ml(arch)))

    results: Dict[str, Dict[str, object]] = {}
    if verbose:
        print(
            f"[{report_label}] evaluating {len(configs)} controllers over "
            f"{num_games} games each ({cfg.players} players)  "
            f"samples={maximax_samples}  lookahead_depth={lookahead_depth}  "
            f"ml_weight={ml_weight}  rollout_config={rollout_config_label!r}"
        )

    for pos, (label, factory) in enumerate(configs):
        if verbose:
            print(f"  running {label} ...")
        ml_agg = MLGuidanceStats() if pos > 0 else None

        def _hook(controllers, _score, _agg=ml_agg):
            if _agg is None:
                return
            for ctrl in controllers.values():
                if isinstance(ctrl, MaximaxController):
                    _agg.merge(ctrl.ml_stats)

        t0 = time.perf_counter()
        raw = _evaluate_controller_logged(
            cfg, runner,
            make_controller=factory,
            num_episodes=num_games,
            seed=seed,
            post_episode=_hook if ml_agg is not None else None,
        )
        wall = time.perf_counter() - t0
        stats = _score_stats(raw["scores"])
        stats["wall_time_s"] = float(wall)
        results[label] = stats
        if verbose:
            print(
                f"    {label:<24s} mean={stats['mean']:.2f}  "
                f"std={stats['std']:.2f}  min={stats['min']}  "
                f"max={stats['max']}  n={stats['n']}  "
                f"var={stats['variance']:.3f}  wall_s={wall:.2f}"
            )
        if ml_agg is not None:
            if verbose:
                print("    " + ml_agg.summary())
            n_dec = max(1, ml_agg.decisions)
            results[label]["ml_diagnostics"] = {
                "decisions": ml_agg.decisions,
                "candidates_scored": ml_agg.candidates_scored,
                "action_changes": ml_agg.action_changes,
                "action_change_rate": ml_agg.action_changes / n_dec,
                "avg_rollout_best": ml_agg.sum_rollout_best / n_dec,
                "avg_ml_best": ml_agg.sum_ml_best / n_dec,
                "avg_ml_mean": ml_agg.sum_ml_mean / n_dec,
            }

    if plot_path:
        plot_phase8_results(results, plot_path, cfg, report_label=report_label)
    return results


def run_phase10_compare(
    num_games: int = 30,
    seed: int = 0,
    plot_path: Optional[str] = os.path.join(PNG_DIR, "phase10_controller_scores.png"),
    cfg: Optional[HanabiConfig] = None,
    verbose: bool = True,
    maximax_samples: int = 80,
    lookahead_depth: int = 1,
    ml_weight: float = 0.1,
    ml_debug: bool = False,
    model_paths: Optional[Dict[str, str]] = None,
    include_archs: Tuple[str, ...] = ("GRU", "LSTM", "Transformer"),
    rollout_weights: Optional[Dict[str, float]] = None,
    hint_score_tune: Optional[Dict[str, float]] = None,
    rollout_config_label: str = "default",
) -> Dict[str, Dict[str, object]]:
    return run_phase8_compare(
        num_games=num_games,
        seed=seed,
        plot_path=plot_path,
        cfg=cfg,
        verbose=verbose,
        maximax_samples=maximax_samples,
        lookahead_depth=lookahead_depth,
        ml_weight=ml_weight,
        ml_debug=ml_debug,
        model_paths=model_paths,
        include_archs=include_archs,
        report_label="Phase 10",
        baseline_name="Improved Maximax (baseline)",
        ml_name_template="Improved Maximax + {arch}",
        rollout_weights=rollout_weights,
        hint_score_tune=hint_score_tune,
        rollout_config_label=rollout_config_label,
    )


def plot_phase8_results(
    results: Dict[str, Dict[str, object]],
    plot_path: str,
    cfg: HanabiConfig,
    report_label: str = "Phase 8",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[{report_label}] matplotlib not installed; skipping plot")
        return

    labels = list(results.keys())
    means = [float(results[l]["mean"]) for l in labels]
    stds = [float(results[l]["std"]) for l in labels]
    max_score = cfg.colors * cfg.ranks

    palette = ["#8ea8d8", "#a3c8a3", "#d89a6a", "#c8689e", "#8c8ad8"]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
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
    ax.set_xticklabels([l.replace(" + ", "\n+") for l in labels], fontsize=9)
    ax.set_ylabel("Final fireworks score")
    ax.set_ylim(0, max_score + 2)
    ax.axhline(max_score, color="gray", linestyle="--", linewidth=0.8,
               alpha=0.5, label=f"max ({max_score})")
    ax.grid(True, axis="y", alpha=0.3)

    any_stats = next(iter(results.values()), {})
    n = any_stats.get("n", "?")
    ax.set_title(f"{report_label} - Maximax++ (ML optional)  (N={n} games)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    parent = os.path.dirname(os.path.abspath(plot_path)) or "."
    os.makedirs(parent, exist_ok=True)
    fig.savefig(plot_path, dpi=120)
    print(f"[{report_label}] bar chart saved -> {plot_path}")
    plt.close(fig)


PHASE9_DEFAULT_HISTORY_PATH: str = "phase9_playstyle_history.pkl"


class PlaystyleTracker:
    def __init__(
        self,
        wrapper,
        human_agent: str,
        cfg: Optional[HanabiConfig] = None,
        style_window: int = 10,
        verbose: bool = True,
        save_path: Optional[str] = PHASE9_DEFAULT_HISTORY_PATH,
        model_tag: str = "transformer",
        debug: bool = True,
    ) -> None:
        if wrapper is None:
            raise ValueError("PlaystyleTracker needs an MLInferenceWrapper")
        if torch is None:
            raise RuntimeError("PlaystyleTracker requires PyTorch")
        self.wrapper = wrapper
        self.cfg: HanabiConfig = cfg or getattr(wrapper, "cfg", None) or HanabiConfig()
        self.human_agent = str(human_agent)
        self.style_window = max(1, int(style_window))
        self.verbose = bool(verbose)
        self.save_path = save_path
        self.model_tag = str(model_tag)
        self.debug = bool(debug)
        wrapper_map = dict(getattr(wrapper, "label_map", LABEL_TO_INDEX) or LABEL_TO_INDEX)
        if wrapper_map != LABEL_TO_INDEX:
            print(
                f"[Phase 9] WARN: wrapper.label_map {wrapper_map} differs from "
                f"training LABEL_TO_INDEX {LABEL_TO_INDEX}; using training order"
            )
        self.label_map: Dict[str, int] = dict(LABEL_TO_INDEX)
        self._feature_dim: int = int(feature_vector_size(self.cfg))
        self._model_input_dim: Optional[int] = self._probe_model_input_dim(wrapper.model)
        if (self._model_input_dim is not None
                and self._model_input_dim != self._feature_dim):
            raise RuntimeError(
                f"[Phase 9] feature dim mismatch: cfg yields F={self._feature_dim} "
                f"but model expects {self._model_input_dim}. "
                f"Re-train Phase 7 with the same HanabiConfig used here."
            )
        self._device = getattr(wrapper, "device", torch.device("cpu"))
        self._buffer: List[TrajectoryStep] = []
        self.history: List[Dict[str, object]] = []
        self._human_turn_idx: int = 0
        self._debug_banner_printed: bool = False

    @staticmethod
    def _probe_model_input_dim(model) -> Optional[int]:
        try:
            gru = getattr(model, "gru", None)
            if gru is not None and hasattr(gru, "input_size"):
                return int(gru.input_size)
            lstm = getattr(model, "lstm", None)
            if lstm is not None and hasattr(lstm, "input_size"):
                return int(lstm.input_size)
            proj = getattr(model, "proj", None)
            if proj is not None and hasattr(proj, "in_features"):
                return int(proj.in_features)
        except Exception:
            return None
        return None

    def reset(self) -> None:
        self._buffer.clear()
        self.history.clear()
        self._human_turn_idx = 0
        self._debug_banner_printed = False

    def _build_sequence_features(
        self, buffer: List[TrajectoryStep],
    ) -> np.ndarray:
        if not buffer:
            return np.zeros((0, self._feature_dim), dtype=np.float32)
        per_step = [step_to_features(s, self.cfg) for s in buffer]
        for t, vec in enumerate(per_step):
            if not isinstance(vec, np.ndarray):
                raise TypeError(
                    f"[Phase 9] step_to_features returned {type(vec)} for buffer[{t}]"
                )
            if vec.ndim != 1 or vec.shape[0] != self._feature_dim:
                raise ValueError(
                    f"[Phase 9] feature shape mismatch at buffer[{t}]: got "
                    f"{vec.shape}, expected ({self._feature_dim},)"
                )
        return np.stack(per_step, axis=0).astype(np.float32)

    def observe(self, step: TrajectoryStep) -> Optional[Dict[str, float]]:
        if step is None or step.agent != self.human_agent:
            return None

        self._buffer.append(step)
        if len(self._buffer) > self.style_window:
            self._buffer = self._buffer[-self.style_window:]
        self._human_turn_idx += 1
        turn = self._human_turn_idx

        try:
            seq = self._build_sequence_features(self._buffer)
        except Exception as exc:
            print(f"[Phase 9] feature extraction failed at turn {turn}: {exc}")
            return None

        T, F = seq.shape if seq.ndim == 2 else (0, 0)

        if self.debug and not self._debug_banner_printed:
            print(
                f"[Phase 9 DEBUG] cfg=(N={self.cfg.players}, H={self.cfg.hand_size}, "
                f"C={self.cfg.colors}, R={self.cfg.ranks})  "
                f"feature_vector_size(cfg)={self._feature_dim}  "
                f"model.input_dim={self._model_input_dim}  "
                f"label_map={self.label_map}  "
                f"window={self.style_window}  device={self._device}"
            )
            self._debug_banner_printed = True

        if T == 0 or F != self._feature_dim:
            print(
                f"[Phase 9] turn={turn} bad sequence shape ({T}, {F}); "
                f"expected (T>=1, {self._feature_dim}); skipping inference"
            )
            return None

        try:
            x = torch.from_numpy(seq).unsqueeze(0).to(self._device)
            lengths = torch.tensor([T], dtype=torch.long)
            with torch.no_grad():
                logits = self.wrapper.model(x, lengths=lengths)
                if logits.dim() != 2 or logits.size(0) != 1:
                    raise RuntimeError(
                        f"unexpected logits shape {tuple(logits.shape)}; "
                        f"expected (1, num_classes)"
                    )
                probs_t = torch.softmax(logits, dim=1)
                probs_arr = probs_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
        except Exception as exc:
            print(f"[Phase 9] inference failed at turn {turn}: {exc}")
            return None

        playstyle_probs: Dict[str, float] = {ps: 0.0 for ps in PLAYSTYLE_ORDER}
        for label, idx in self.label_map.items():
            ps = LABEL_TO_PLAYSTYLE.get(label)
            if ps is None:
                continue
            if 0 <= int(idx) < probs_arr.shape[0]:
                playstyle_probs[ps] = float(probs_arr[int(idx)])

        record: Dict[str, object] = {"turn": int(turn)}
        record.update({ps: float(playstyle_probs[ps]) for ps in PLAYSTYLE_ORDER})
        self.history.append(record)

        if self.debug:
            print(
                f"[Phase 9 DEBUG] turn={turn}  buffer_len={len(self._buffer)}  "
                f"seq.shape=({T}, {F})  "
                f"raw_probs=[random={probs_arr[0]:.3f}, "
                f"heuristic={probs_arr[1]:.3f}, "
                f"maximax={probs_arr[2]:.3f}]"
            )

        if self.verbose:
            print("[Human Playstyle]")
            print(f"turn={turn}")
            print(
                f"chaotic={playstyle_probs['chaotic']:.2f}  "
                f"cooperative={playstyle_probs['cooperative']:.2f}  "
                f"strategic={playstyle_probs['strategic']:.2f}"
            )

        return playstyle_probs

    def save(self, path: Optional[str] = None) -> Optional[str]:
        target = path if path is not None else self.save_path
        if target is None:
            return None
        if not self.history:
            if self.verbose:
                print("[Phase 9] no human turns recorded - skipping save")
            return None
        payload = {
            "human_agent": self.human_agent,
            "style_window": self.style_window,
            "model": self.model_tag,
            "playstyles": list(PLAYSTYLE_ORDER),
            "labels": list(CONTROLLER_LABELS),
            "label_to_playstyle": dict(LABEL_TO_PLAYSTYLE),
            "history": list(self.history),
        }
        try:
            parent = os.path.dirname(os.path.abspath(target)) or "."
            os.makedirs(parent, exist_ok=True)
            with open(target, "wb") as fh:
                pickle.dump(payload, fh)
        except OSError as exc:
            print(f"[Phase 9] failed to save playstyle history -> {target}: {exc}")
            return None
        if self.verbose:
            print(
                f"[Phase 9] playstyle history saved -> {target}  "
                f"({len(self.history)} entries, model={self.model_tag})"
            )
        return target


def build_phase9_tracker(
    model_type: str,
    human_agent: str,
    cfg: Optional[HanabiConfig] = None,
    style_window: int = 10,
    verbose: bool = True,
    save_path: Optional[str] = PHASE9_DEFAULT_HISTORY_PATH,
    debug: bool = True,
) -> Optional["PlaystyleTracker"]:
    if torch is None or load_phase8_wrapper is None:
        print("[Phase 9] PyTorch unavailable - playstyle tracking disabled")
        return None
    cfg = cfg or HanabiConfig()
    arch = _phase8_model_key(model_type)
    try:
        wrapper = load_phase8_wrapper(arch, cfg=cfg)
    except FileNotFoundError as exc:
        print(f"[Phase 9] {exc}")
        print("[Phase 9] playstyle tracking disabled for this run")
        return None
    except Exception as exc:
        print(f"[Phase 9] failed to load {arch} weights: {exc}")
        return None
    print(f"[Phase 9] tracking human '{human_agent}' with {arch} model "
          f"(window={style_window})")
    try:
        return PlaystyleTracker(
            wrapper=wrapper,
            human_agent=human_agent,
            cfg=cfg,
            style_window=style_window,
            verbose=verbose,
            save_path=save_path,
            model_tag=arch,
            debug=debug,
        )
    except RuntimeError as exc:
        print(f"[Phase 9] {exc}")
        print("[Phase 9] playstyle tracking disabled for this run")
        return None
