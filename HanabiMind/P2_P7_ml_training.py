from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional, Tuple
import os
import pickle
import random

import numpy as np

from P0_P1_project_config import HanabiConfig
from P0_observation_decoding import (
    ActionType,
    Card,
    DecodedObservation,
    LastAction,
    StructuredAction,
    copies_per_rank,
    total_copies_per_color,
)
from P0_P1_environment import PettingZooHanabiRunner
from P1_P5_P6_P8_P10_controllers import (
    HeuristicController,
    MaximaxController,
    PlaystyleBiasedController,
    RandomController,
)
from P2_P7_trajectory_schema import (
    CONTROLLER_LABELS,
    LABEL_TO_INDEX,
    LABEL_TO_PLAYSTYLE,
    TrajectoryStep,
)


PNG_DIR = "png"
TRAINED_MODEL_DIR = "trained models"
PHASE2_GRU_MODEL_PATH = os.path.join(TRAINED_MODEL_DIR, "phase2_gru.pt")
PHASE7_MODEL_PATHS: Dict[str, str] = {
    "GRU": os.path.join(TRAINED_MODEL_DIR, "phase7_gru.pt"),
    "LSTM": os.path.join(TRAINED_MODEL_DIR, "phase7_lstm.pt"),
    "Transformer": os.path.join(TRAINED_MODEL_DIR, "phase7_transformer.pt"),
}


class _LegacyDatasetUnpickler(pickle.Unpickler):
    _MAIN_CLASS_MAP = {
        "ActionType": ActionType,
        "Card": Card,
        "DecodedObservation": DecodedObservation,
        "HanabiConfig": HanabiConfig,
        "LastAction": LastAction,
        "StructuredAction": StructuredAction,
        "TrajectoryStep": TrajectoryStep,
    }

    def find_class(self, module, name):
        if module == "__main__" and name in self._MAIN_CLASS_MAP:
            return self._MAIN_CLASS_MAP[name]
        return super().find_class(module, name)


try:
    import torch
    import torch.nn as nn

    class GRUClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, num_classes=3):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, num_classes)

        def forward(self, x, lengths=None):
            if lengths is not None:
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False,
                )
                _, h = self.gru(packed)
            else:
                _, h = self.gru(x)
            return self.fc(h[-1])
except ImportError:
    torch = None
    nn = None
    GRUClassifier = None


def _build_controllers(
    label: str,
    cfg: HanabiConfig,
    style_biased: bool = False,
    rng_seed: int = 0,
    lookahead_depth: int = 1,
) -> Dict[str, object]:
    if label == "random":
        factory = lambda i: RandomController(f"rand{i}", cfg)
    elif label == "heuristic":
        factory = lambda i: HeuristicController(f"heur{i}", cfg)
    elif label == "maximax":
        factory = lambda i: MaximaxController(f"max{i}", cfg, lookahead_depth=lookahead_depth)
    else:
        raise ValueError(f"unknown controller label: {label}")
    controllers: Dict[str, object] = {}
    for i in range(cfg.players):
        base = factory(i)
        if style_biased:
            style = LABEL_TO_PLAYSTYLE.get(label, label)
            base = PlaystyleBiasedController(
                f"{style}_{i}",
                cfg,
                base=base,
                style=style,
                rng_seed=rng_seed + i,
                override_prob={
                    "chaotic": 0.78,
                    "cooperative": 0.72,
                    "strategic": 0.18,
                }.get(style, 0.5),
            )
        controllers[f"player_{i}"] = base
    return controllers


DATASET_SCHEMA_VERSION: int = 2
DATASET_TRAJECTORY_KIND: str = "per_agent"


def _split_history_by_agent(
    history: List["TrajectoryStep"],
    cfg: HanabiConfig,
) -> Dict[str, List["TrajectoryStep"]]:
    buckets: Dict[str, List["TrajectoryStep"]] = {
        f"player_{i}": [] for i in range(cfg.players)
    }
    for step in history:
        agent = getattr(step, "agent", None)
        if agent is None:
            continue
        buckets.setdefault(agent, []).append(step)
    return buckets


def generate_dataset(
    num_games: int = 40,
    labels: Tuple[str, ...] = CONTROLLER_LABELS,
    save_path: Optional[str] = "phase2_trajectories.pkl",
    seed: int = 0,
    cfg: Optional[HanabiConfig] = None,
    style_biased: bool = False,
    verbose: bool = True,
    lookahead_depth: int = 1,
) -> List[Dict[str, object]]:
    cfg = cfg or HanabiConfig()
    runner = PettingZooHanabiRunner(cfg)

    trajectories: List[Dict[str, object]] = []
    for g in range(num_games):
        label = labels[g % len(labels)]
        controllers = _build_controllers(
            label,
            cfg,
            style_biased=style_biased,
            rng_seed=seed + g * 101,
            lookahead_depth=lookahead_depth,
        )
        out = runner.run_episode(
            controllers,
            seed=seed + g,
            log_trajectory=True,
        )
        history = out["history"]
        score = int(out["score"])

        buckets = _split_history_by_agent(history, cfg)
        per_agent_trajs: List[Dict[str, object]] = []
        for agent_name in sorted(buckets.keys()):
            agent_steps = buckets[agent_name]
            if not agent_steps:
                continue
            per_agent_trajs.append({
                "controller_label": label,
                "score": score,
                "steps": agent_steps,
                "agent": agent_name,
                "game_idx": g,
            })

        trajectories.extend(per_agent_trajs)
        if verbose:
            agent_lens = [len(t["steps"]) for t in per_agent_trajs]
            mean_len = (sum(agent_lens) / len(agent_lens)) if agent_lens else 0.0
            print(
                f"  [dataset] game {g + 1:3d}/{num_games}  "
                f"label={label:<9s}  agents={len(per_agent_trajs)}  "
                f"steps_total={len(history):3d}  "
                f"steps/agent~{mean_len:.1f}  "
                f"score={score}"
            )

    if save_path:
        parent = os.path.dirname(os.path.abspath(save_path)) or "."
        os.makedirs(parent, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(
                {
                    "cfg": asdict(cfg),
                    "trajectories": trajectories,
                    "labels": list(labels),
                    "schema_version": DATASET_SCHEMA_VERSION,
                    "trajectory_kind": DATASET_TRAJECTORY_KIND,
                    "style_biased": bool(style_biased),
                    "num_games": int(num_games),
                },
                f,
            )
        if verbose:
            print(
                f"  [dataset] saved -> {save_path}  "
                f"({len(trajectories)} per-agent trajectories from "
                f"{num_games} games, style_biased={bool(style_biased)})"
            )

    return trajectories


def load_dataset(path: str) -> Dict[str, object]:
    with open(path, "rb") as f:
        data = _LegacyDatasetUnpickler(f).load()
    cfg_val = data.get("cfg")
    if isinstance(cfg_val, dict):
        data["cfg"] = HanabiConfig(**cfg_val)
    return data


def _load_or_generate_dataset(
    dataset_path: Optional[str],
    num_games: int,
    seed: int,
    cfg: HanabiConfig,
    labels: Tuple[str, ...] = CONTROLLER_LABELS,
    style_biased: bool = False,
    verbose: bool = True,
    log_prefix: str = "[dataset]",
    lookahead_depth: int = 1,
) -> List[Dict[str, object]]:
    if dataset_path and os.path.exists(dataset_path):
        cached = load_dataset(dataset_path)
        cached_labels = tuple(cached.get("labels") or [])
        cached_version = int(cached.get("schema_version", 1))
        cached_kind = str(cached.get("trajectory_kind", "mixed_agent"))
        cached_style_biased = bool(cached.get("style_biased", False))
        if (
            cached_labels == tuple(labels)
            and cached_version == DATASET_SCHEMA_VERSION
            and cached_kind == DATASET_TRAJECTORY_KIND
            and cached_style_biased == bool(style_biased)
        ):
            if verbose:
                print(
                    f"{log_prefix} loading cached dataset from {dataset_path}  "
                    f"(schema v{cached_version} / {cached_kind}, "
                    f"{len(cached.get('trajectories', []))} trajectories, "
                    f"style_biased={cached_style_biased})"
                )
            return cached["trajectories"]
        if verbose:
            reasons: List[str] = []
            if cached_labels != tuple(labels):
                reasons.append(f"labels {cached_labels} != {tuple(labels)}")
            if cached_version != DATASET_SCHEMA_VERSION:
                reasons.append(f"schema v{cached_version} != v{DATASET_SCHEMA_VERSION}")
            if cached_kind != DATASET_TRAJECTORY_KIND:
                reasons.append(f"kind={cached_kind!r} != {DATASET_TRAJECTORY_KIND!r}")
            if cached_style_biased != bool(style_biased):
                reasons.append(f"style_biased={cached_style_biased} != {bool(style_biased)}")
            print(
                f"{log_prefix} cache at {dataset_path} is stale "
                f"({'; '.join(reasons) or 'unknown'}); regenerating"
            )

    if verbose:
        print(f"{log_prefix} generating {num_games} self-play games ...")
    return generate_dataset(
        num_games=num_games,
        labels=labels,
        save_path=dataset_path,
        seed=seed,
        cfg=cfg,
        style_biased=style_biased,
        verbose=verbose,
        lookahead_depth=lookahead_depth,
    )


def build_windowed_playstyle_trajectories(
    trajectories: List[Dict[str, object]],
    window_size: int = 10,
    stride: int = 1,
    balance: bool = True,
    balance_lengths: bool = True,
    seed: int = 0,
    verbose: bool = True,
) -> List[Dict[str, object]]:
    window_size = max(1, int(window_size))
    stride = max(1, int(stride))
    by_label: Dict[str, List[Dict[str, object]]] = {
        label: [] for label in CONTROLLER_LABELS
    }

    for traj in trajectories:
        label = str(traj.get("controller_label", ""))
        steps = list(traj.get("steps") or [])
        if label not in by_label or not steps:
            continue
        for end in range(1, len(steps) + 1, stride):
            start = max(0, end - window_size)
            by_label[label].append({
                "controller_label": label,
                "score": traj.get("score"),
                "steps": steps[start:end],
                "agent": traj.get("agent"),
                "game_idx": traj.get("game_idx"),
                "window_start": start,
                "window_end": end,
                "window_size": window_size,
                "trajectory_kind": "windowed_playstyle",
            })

    rng = random.Random(seed)
    out: List[Dict[str, object]] = []
    if balance and balance_lengths:
        by_label_len: Dict[str, Dict[int, List[Dict[str, object]]]] = {
            label: {} for label in CONTROLLER_LABELS
        }
        all_lengths = set()
        for label in CONTROLLER_LABELS:
            for window in by_label[label]:
                length = len(window.get("steps") or [])
                all_lengths.add(length)
                by_label_len[label].setdefault(length, []).append(window)

        for length in sorted(all_lengths):
            buckets = [
                by_label_len[label].get(length, [])
                for label in CONTROLLER_LABELS
            ]
            if any(not bucket for bucket in buckets):
                continue
            target = min(len(bucket) for bucket in buckets)
            for label in CONTROLLER_LABELS:
                windows = by_label_len[label][length]
                rng.shuffle(windows)
                out.extend(windows[:target])
    else:
        if balance:
            non_empty = [len(v) for v in by_label.values() if v]
            target = min(non_empty) if non_empty else 0
        else:
            target = 0

        for label in CONTROLLER_LABELS:
            windows = by_label[label]
            rng.shuffle(windows)
            if balance and target > 0:
                windows = windows[:target]
            out.extend(windows)
    rng.shuffle(out)

    if verbose:
        counts = {label: len(by_label[label]) for label in CONTROLLER_LABELS}
        used = {label: 0 for label in CONTROLLER_LABELS}
        lengths: Dict[str, List[int]] = {label: [] for label in CONTROLLER_LABELS}
        for item in out:
            label = str(item.get("controller_label"))
            used[label] = used.get(label, 0) + 1
            lengths.setdefault(label, []).append(len(item.get("steps") or []))
        used_lengths = {
            label: sorted(set(vals))
            for label, vals in lengths.items() if vals
        }
        avg_len = {
            label: round(sum(vals) / len(vals), 2)
            for label, vals in lengths.items() if vals
        }
        print(
            "[windowed playstyle] raw_windows="
            f"{counts}  used={used}  avg_window_len={avg_len}  "
            f"lengths={used_lengths}"
        )

    return out


def feature_vector_size(cfg: HanabiConfig) -> int:
    N, H, C, R = cfg.players, cfg.hand_size, cfg.colors, cfg.ranks
    total_actions = 2 * H + (N - 1) * (C + R)
    return (
        C
        + 3
        + C * R
        + N
        + (N - 1) * H * (C * R + 1)
        + total_actions
    )


def step_to_features(step: TrajectoryStep, cfg: HanabiConfig) -> np.ndarray:
    N, H, C, R = cfg.players, cfg.hand_size, cfg.colors, cfg.ranks
    obs = step.decoded_obs
    total_actions = 2 * H + (N - 1) * (C + R)
    max_copies = max(copies_per_rank(r, R) for r in range(R))
    max_deck_size = total_copies_per_color(R) * C - N * H

    parts: List[np.ndarray] = []
    parts.append(np.asarray(obs.fireworks, dtype=np.float32) / float(R))
    parts.append(np.array(
        [
            obs.information_tokens / max(1.0, cfg.max_information_tokens),
            obs.life_tokens / max(1.0, cfg.max_life_tokens),
            obs.deck_size / max(1.0, max_deck_size),
        ],
        dtype=np.float32,
    ))
    parts.append((obs.discards.astype(np.float32) / float(max_copies)).reshape(-1))
    parts.append(np.asarray(obs.missing_card, dtype=np.float32))

    hands = np.zeros((N - 1, H, C * R + 1), dtype=np.float32)
    for p, hand in enumerate(obs.partner_hands):
        for s in range(H):
            card = hand[s] if s < len(hand) else None
            if card is None:
                hands[p, s, -1] = 1.0
            else:
                hands[p, s, card.color * R + card.rank] = 1.0
    parts.append(hands.reshape(-1))

    action = np.zeros(total_actions, dtype=np.float32)
    if 0 <= step.action_id < total_actions:
        action[step.action_id] = 1.0
    parts.append(action)

    return np.concatenate(parts, axis=0).astype(np.float32)


if torch is not None:
    from torch.utils.data import Dataset, DataLoader

    class HanabiTrajectoryDataset(Dataset):
        def __init__(
            self,
            trajectories: List[Dict[str, object]],
            cfg: HanabiConfig,
            label_map: Dict[str, int] = LABEL_TO_INDEX,
        ) -> None:
            self.cfg = cfg
            self.label_map = label_map
            self._items: List[Tuple[np.ndarray, int]] = []
            for traj in trajectories:
                label = traj.get("controller_label")
                steps = traj.get("steps") or []
                if label not in label_map or not steps:
                    continue
                feats = np.stack(
                    [step_to_features(s, cfg) for s in steps], axis=0
                ).astype(np.float32)
                self._items.append((feats, int(label_map[label])))

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
            return self._items[idx]

    def collate_trajectories(batch):
        feats = [torch.from_numpy(f) for f, _ in batch]
        labels = torch.tensor([l for _, l in batch], dtype=torch.long)
        lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
        padded = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
        return padded, lengths, labels


def _train_val_split(
    dataset_size: int,
    val_fraction: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    indices = list(range(dataset_size))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(round(dataset_size * val_fraction))) if dataset_size > 1 else 0
    return indices[val_size:], indices[:val_size]


def train_phase2_model(
    trajectories: Optional[List[Dict[str, object]]] = None,
    cfg: Optional[HanabiConfig] = None,
    num_games: int = 40,
    epochs: int = 15,
    batch_size: int = 8,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    seed: int = 0,
    save_model_path: Optional[str] = PHASE2_GRU_MODEL_PATH,
    save_history_path: Optional[str] = "phase2_training_history.pkl",
    plot_path: Optional[str] = os.path.join(PNG_DIR, "phase2_loss_curve.png"),
    dataset_path: Optional[str] = "phase2_trajectories.pkl",
    lookahead_depth: int = 1,
) -> Dict[str, object]:
    if torch is None:
        raise RuntimeError("PyTorch is required for Phase 2 training")

    cfg = cfg or HanabiConfig()

    if trajectories is None:
        trajectories = _load_or_generate_dataset(
            dataset_path=dataset_path,
            num_games=num_games,
            seed=seed,
            cfg=cfg,
            labels=CONTROLLER_LABELS,
            verbose=True,
            log_prefix="[Phase 2]",
            lookahead_depth=lookahead_depth,
        )

    dataset = HanabiTrajectoryDataset(trajectories, cfg, LABEL_TO_INDEX)
    if len(dataset) < 2:
        raise RuntimeError(
            f"Dataset too small to train ({len(dataset)} usable trajectories)"
        )

    train_idx, val_idx = _train_val_split(len(dataset), val_fraction, seed)
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_trajectories,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_trajectories,
    ) if len(val_ds) > 0 else None

    input_dim = feature_vector_size(cfg)
    model = GRUClassifier(input_dim=input_dim, hidden_dim=hidden_dim,
                          num_classes=len(CONTROLLER_LABELS))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    torch.manual_seed(seed)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }

    print(
        f"[Phase 2] training  input_dim={input_dim}  hidden={hidden_dim}  "
        f"train={len(train_ds)}  val={len(val_ds)}  epochs={epochs}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for padded, lengths, labels in train_loader:
            optimizer.zero_grad()
            logits = model(padded, lengths=lengths)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += labels.size(0)
        train_loss = total_loss / max(1, seen)
        train_acc = correct / max(1, seen)

        val_loss = float("nan")
        val_acc = float("nan")
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                vl_sum, vc_sum, vn = 0.0, 0, 0
                for padded, lengths, labels in val_loader:
                    logits = model(padded, lengths=lengths)
                    loss = loss_fn(logits, labels)
                    vl_sum += float(loss.item()) * labels.size(0)
                    vc_sum += int((logits.argmax(dim=1) == labels).sum().item())
                    vn += labels.size(0)
                val_loss = vl_sum / max(1, vn)
                val_acc = vc_sum / max(1, vn)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(
            f"  epoch {epoch:2d}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
        )

    if save_model_path:
        parent = os.path.dirname(os.path.abspath(save_model_path)) or "."
        os.makedirs(parent, exist_ok=True)
        torch.save(model.state_dict(), save_model_path)
        print(f"[Phase 2] model weights saved -> {save_model_path}")
    if save_history_path:
        with open(save_history_path, "wb") as f:
            pickle.dump(history, f)
        print(f"[Phase 2] training history saved -> {save_history_path}")
    if plot_path:
        plot_training_history(history, plot_path)

    return {"model": model, "history": history, "dataset": dataset}


def plot_training_history(
    history: Dict[str, List[float]],
    plot_path: Optional[str] = os.path.join(PNG_DIR, "phase2_loss_curve.png"),
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Phase 2] matplotlib not installed; skipping plot")
        return

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss.plot(epochs, history["train_loss"], label="train", marker="o")
    if any(not np.isnan(v) for v in history["val_loss"]):
        ax_loss.plot(epochs, history["val_loss"], label="val", marker="o")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    ax_acc.plot(epochs, history["train_acc"], label="train", marker="o")
    if any(not np.isnan(v) for v in history["val_acc"]):
        ax_acc.plot(epochs, history["val_acc"], label="val", marker="o")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_ylim(0.0, 1.05)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend()

    fig.suptitle("Phase 2 - GRU controller classifier")
    fig.tight_layout()

    if plot_path:
        parent = os.path.dirname(os.path.abspath(plot_path)) or "."
        os.makedirs(parent, exist_ok=True)
        fig.savefig(plot_path, dpi=120)
        print(f"[Phase 2] loss curve saved -> {plot_path}")
    plt.close(fig)


def run_phase2(
    num_games: int = 40,
    epochs: int = 15,
    batch_size: int = 8,
    seed: int = 0,
    lookahead_depth: int = 1,
) -> Dict[str, object]:
    if torch is None:
        print("[Phase 2] skipped: PyTorch is not installed")
        return {}
    cfg = HanabiConfig()
    return train_phase2_model(
        cfg=cfg,
        num_games=num_games,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        lookahead_depth=lookahead_depth,
    )


if torch is not None:
    import math

    class LSTMClassifier(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64,
                     num_classes: int = len(CONTROLLER_LABELS)) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, num_classes)

        def forward(self, x, lengths=None):
            if lengths is not None:
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False,
                )
                _, (h, _) = self.lstm(packed)
            else:
                _, (h, _) = self.lstm(x)
            return self.fc(h[-1])

    class _PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 1024) -> None:
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32)
                * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, : x.size(1)]

    class TransformerClassifier(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            num_classes: int = len(CONTROLLER_LABELS),
            num_heads: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 128,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            if hidden_dim % num_heads != 0:
                raise ValueError(
                    f"hidden_dim ({hidden_dim}) must be divisible by "
                    f"num_heads ({num_heads})"
                )
            self.proj = nn.Linear(input_dim, hidden_dim)
            self.pos = _PositionalEncoding(hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(hidden_dim, num_classes)

        def forward(self, x, lengths=None):
            x = self.proj(x)
            x = self.pos(x)
            key_padding_mask = None
            if lengths is not None:
                B, T, _ = x.shape
                idx = torch.arange(T, device=x.device).unsqueeze(0)
                key_padding_mask = idx >= lengths.to(x.device).unsqueeze(1)
            h = self.encoder(x, src_key_padding_mask=key_padding_mask)
            if key_padding_mask is not None:
                valid = (~key_padding_mask).unsqueeze(-1).float()
                pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
            else:
                pooled = h.mean(dim=1)
            return self.fc(pooled)
else:
    LSTMClassifier = None
    TransformerClassifier = None


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> Dict[str, object]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    try:
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            confusion_matrix,
        )
        labels = list(range(num_classes))
        acc = float(accuracy_score(y_true, y_pred))
        prec, rec, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0,
        )
        macro = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="macro", zero_division=0,
        )
        weighted = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return {
            "accuracy": acc,
            "precision": [float(p) for p in prec],
            "recall": [float(r) for r in rec],
            "f1": [float(f) for f in f1],
            "support": [int(s) for s in support],
            "macro_precision": float(macro[0]),
            "macro_recall": float(macro[1]),
            "macro_f1": float(macro[2]),
            "weighted_precision": float(weighted[0]),
            "weighted_recall": float(weighted[1]),
            "weighted_f1": float(weighted[2]),
            "confusion_matrix": cm,
            "backend": "sklearn",
        }
    except ImportError:
        pass

    n = len(y_true)
    acc = float((y_true == y_pred).sum() / max(1, n))
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    precision, recall, f1, support = [], [], [], []
    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        sup = int(cm[c, :].sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(f)
        support.append(sup)

    total_sup = max(1, sum(support))
    macro_p = float(np.mean(precision))
    macro_r = float(np.mean(recall))
    macro_f = float(np.mean(f1))
    weighted_p = float(sum(p * s for p, s in zip(precision, support)) / total_sup)
    weighted_r = float(sum(r * s for r, s in zip(recall, support)) / total_sup)
    weighted_f = float(sum(f * s for f, s in zip(f1, support)) / total_sup)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f,
        "confusion_matrix": cm,
        "backend": "numpy",
    }


def _format_classification_report(
    metrics: Dict[str, object],
    class_names: Tuple[str, ...],
) -> str:
    head = f"{'class':<14s} {'precision':>10s} {'recall':>10s} {'f1':>10s} {'support':>10s}"
    lines = [head, "-" * len(head)]
    support_total = sum(metrics["support"])
    for i, name in enumerate(class_names):
        lines.append(
            f"{name:<14s} {metrics['precision'][i]:>10.3f} "
            f"{metrics['recall'][i]:>10.3f} {metrics['f1'][i]:>10.3f} "
            f"{metrics['support'][i]:>10d}"
        )
    lines.append("")
    lines.append(
        f"{'accuracy':<14s} {'':>10s} {'':>10s} "
        f"{metrics['accuracy']:>10.3f} {support_total:>10d}"
    )
    lines.append(
        f"{'macro avg':<14s} {metrics['macro_precision']:>10.3f} "
        f"{metrics['macro_recall']:>10.3f} {metrics['macro_f1']:>10.3f} "
        f"{support_total:>10d}"
    )
    lines.append(
        f"{'weighted avg':<14s} {metrics['weighted_precision']:>10.3f} "
        f"{metrics['weighted_recall']:>10.3f} {metrics['weighted_f1']:>10.3f} "
        f"{support_total:>10d}"
    )
    return "\n".join(lines)


def _train_and_eval_one(
    model,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    label: str,
    verbose: bool = True,
) -> Dict[str, object]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for padded, lengths, labels in train_loader:
            optimizer.zero_grad()
            logits = model(padded, lengths=lengths)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += labels.size(0)
        train_loss = total_loss / max(1, seen)
        train_acc = correct / max(1, seen)

        val_loss = float("nan")
        val_acc = float("nan")
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                vl_sum, vc_sum, vn = 0.0, 0, 0
                for padded, lengths, labels in val_loader:
                    logits = model(padded, lengths=lengths)
                    loss = loss_fn(logits, labels)
                    vl_sum += float(loss.item()) * labels.size(0)
                    vc_sum += int((logits.argmax(dim=1) == labels).sum().item())
                    vn += labels.size(0)
                val_loss = vl_sum / max(1, vn)
                val_acc = vc_sum / max(1, vn)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        if verbose:
            print(
                f"  [{label:<11s}] epoch {epoch:2d}/{epochs}  "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )

    y_true: List[int] = []
    y_pred: List[int] = []
    if val_loader is not None:
        model.eval()
        with torch.no_grad():
            for padded, lengths, labels in val_loader:
                logits = model(padded, lengths=lengths)
                preds = logits.argmax(dim=1)
                y_true.extend(labels.tolist())
                y_pred.extend(preds.tolist())

    return {
        "model": model,
        "history": history,
        "y_true": np.asarray(y_true, dtype=int),
        "y_pred": np.asarray(y_pred, dtype=int),
    }


def run_phase7(
    trajectories: Optional[List[Dict[str, object]]] = None,
    cfg: Optional[HanabiConfig] = None,
    num_games: int = 40,
    epochs: int = 15,
    batch_size: int = 8,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    seed: int = 0,
    dataset_path: Optional[str] = "phase2_trajectories.pkl",
    plot_curves_path: Optional[str] = os.path.join(PNG_DIR, "phase7_loss_curves.png"),
    plot_cm_path: Optional[str] = os.path.join(PNG_DIR, "phase7_confusion_matrices.png"),
    save_history_path: Optional[str] = "phase7_history.pkl",
    save_model_paths: Optional[Dict[str, str]] = None,
    windowed_playstyle: bool = False,
    style_window: int = 10,
    balance_window_lengths: bool = True,
    style_biased_data: bool = False,
    verbose: bool = True,
    lookahead_depth: int = 1,
) -> Dict[str, object]:
    if torch is None:
        raise RuntimeError("PyTorch is required for Phase 7")

    cfg = cfg or HanabiConfig()
    if style_biased_data and dataset_path == "phase2_trajectories.pkl":
        dataset_path = "phase7_style_biased_trajectories.pkl"
        if verbose:
            print(
                "[Phase 7] style-biased data requested; using separate "
                f"dataset cache {dataset_path}"
            )

    if trajectories is None:
        trajectories = _load_or_generate_dataset(
            dataset_path=dataset_path,
            num_games=num_games,
            seed=seed,
            cfg=cfg,
            labels=CONTROLLER_LABELS,
            style_biased=style_biased_data,
            verbose=verbose,
            log_prefix="[Phase 7]",
            lookahead_depth=lookahead_depth,
        )

    if windowed_playstyle:
        trajectories = build_windowed_playstyle_trajectories(
            trajectories,
            window_size=style_window,
            stride=1,
            balance=True,
            balance_lengths=balance_window_lengths,
            seed=seed,
            verbose=verbose,
        )

    dataset = HanabiTrajectoryDataset(trajectories, cfg, LABEL_TO_INDEX)
    if len(dataset) < 2:
        raise RuntimeError(
            f"Dataset too small to train ({len(dataset)} usable trajectories)"
        )

    train_idx, val_idx = _train_val_split(len(dataset), val_fraction, seed)
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_trajectories,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_trajectories,
    ) if len(val_ds) > 0 else None

    input_dim = feature_vector_size(cfg)
    num_classes = len(CONTROLLER_LABELS)

    if verbose:
        print(
            f"[Phase 7] training 3 models  input_dim={input_dim}  "
            f"hidden={hidden_dim}  train={len(train_ds)}  val={len(val_ds)}  "
            f"epochs={epochs}  batch={batch_size}  lr={lr}"
        )

    def _build(name: str):
        torch.manual_seed(seed)
        if name == "GRU":
            return GRUClassifier(input_dim, hidden_dim, num_classes)
        if name == "LSTM":
            return LSTMClassifier(input_dim, hidden_dim, num_classes)
        if name == "Transformer":
            return TransformerClassifier(input_dim, hidden_dim, num_classes)
        raise ValueError(name)

    model_names = ("GRU", "LSTM", "Transformer")
    runs: Dict[str, Dict[str, object]] = {}

    if save_model_paths is None:
        save_model_paths = dict(PHASE7_MODEL_PATHS)

    for name in model_names:
        if verbose:
            print(f"\n[Phase 7] === training {name} ===")
        model = _build(name)
        out = _train_and_eval_one(
            model, train_loader, val_loader,
            epochs=epochs, lr=lr, label=name, verbose=verbose,
        )
        metrics = _compute_metrics(out["y_true"], out["y_pred"], num_classes)
        runs[name] = {
            "history": out["history"],
            "metrics": metrics,
            "y_true": out["y_true"],
            "y_pred": out["y_pred"],
        }

        if verbose:
            print(f"\n[Phase 7] {name} classification report:")
            print(_format_classification_report(metrics, CONTROLLER_LABELS))
            print(f"  confusion matrix (rows=true, cols=pred):")
            print(metrics["confusion_matrix"])

        model_path = save_model_paths.get(name) if save_model_paths else None
        if model_path:
            parent = os.path.dirname(os.path.abspath(model_path)) or "."
            os.makedirs(parent, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "arch": name,
                    "input_dim": input_dim,
                    "hidden_dim": hidden_dim,
                    "num_classes": num_classes,
                    "labels": list(CONTROLLER_LABELS),
                    "label_to_playstyle": dict(LABEL_TO_PLAYSTYLE),
                    "windowed_playstyle": bool(windowed_playstyle),
                    "style_window": int(style_window),
                    "balance_window_lengths": bool(balance_window_lengths),
                    "style_biased_data": bool(style_biased_data),
                    "dataset_path": dataset_path,
                },
                model_path,
            )
            if verbose:
                print(f"[Phase 7] {name} weights saved -> {model_path}")

    if verbose:
        print("\n[Phase 7] === side-by-side summary ===")
        header = (
            f"{'model':<13s} {'accuracy':>10s} {'macro_P':>10s} "
            f"{'macro_R':>10s} {'macro_F1':>10s} {'weighted_F1':>12s}"
        )
        print(header)
        print("-" * len(header))
        for name in model_names:
            m = runs[name]["metrics"]
            print(
                f"{name:<13s} {m['accuracy']:>10.3f} "
                f"{m['macro_precision']:>10.3f} {m['macro_recall']:>10.3f} "
                f"{m['macro_f1']:>10.3f} {m['weighted_f1']:>12.3f}"
            )

    if plot_curves_path:
        plot_phase7_curves(
            {n: runs[n]["history"] for n in model_names},
            plot_curves_path,
        )
    if plot_cm_path:
        plot_phase7_confusion_matrices(
            {n: runs[n]["metrics"]["confusion_matrix"] for n in model_names},
            CONTROLLER_LABELS,
            plot_cm_path,
        )
    if save_history_path:
        parent = os.path.dirname(os.path.abspath(save_history_path)) or "."
        os.makedirs(parent, exist_ok=True)
        to_save = {
            n: {
                "history": runs[n]["history"],
                "metrics": runs[n]["metrics"],
                "y_true": runs[n]["y_true"],
                "y_pred": runs[n]["y_pred"],
            } for n in model_names
        }
        to_save["_metadata"] = {
            "labels": list(CONTROLLER_LABELS),
            "label_to_playstyle": dict(LABEL_TO_PLAYSTYLE),
            "windowed_playstyle": bool(windowed_playstyle),
            "style_window": int(style_window),
            "balance_window_lengths": bool(balance_window_lengths),
            "style_biased_data": bool(style_biased_data),
            "dataset_path": dataset_path,
            "trajectory_count": int(len(trajectories)),
            "input_dim": int(input_dim),
            "hidden_dim": int(hidden_dim),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "seed": int(seed),
        }
        with open(save_history_path, "wb") as f:
            pickle.dump(to_save, f)
        if verbose:
            print(f"\n[Phase 7] history + metrics saved -> {save_history_path}")

    return runs


def plot_phase7_curves(
    histories: Dict[str, Dict[str, List[float]]],
    plot_path: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Phase 7] matplotlib not installed; skipping curves plot")
        return

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.2))
    color_cycle = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    for i, (name, h) in enumerate(histories.items()):
        c = color_cycle[i % len(color_cycle)]
        epochs = range(1, len(h["train_loss"]) + 1)
        ax_loss.plot(epochs, h["train_loss"], label=f"{name} train",
                     color=c, marker="o", linestyle="-")
        if any(not np.isnan(v) for v in h["val_loss"]):
            ax_loss.plot(epochs, h["val_loss"], label=f"{name} val",
                         color=c, marker="x", linestyle="--", alpha=0.85)
        ax_acc.plot(epochs, h["train_acc"], label=f"{name} train",
                    color=c, marker="o", linestyle="-")
        if any(not np.isnan(v) for v in h["val_acc"]):
            ax_acc.plot(epochs, h["val_acc"], label=f"{name} val",
                        color=c, marker="x", linestyle="--", alpha=0.85)

    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(fontsize=7, loc="upper right")

    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_ylim(0.0, 1.05)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(fontsize=7, loc="lower right")

    fig.suptitle("Phase 7 - GRU vs LSTM vs Transformer (controller classifier)")
    fig.tight_layout()

    parent = os.path.dirname(os.path.abspath(plot_path)) or "."
    os.makedirs(parent, exist_ok=True)
    fig.savefig(plot_path, dpi=120)
    print(f"[Phase 7] loss/accuracy curves saved -> {plot_path}")
    plt.close(fig)


def plot_phase7_confusion_matrices(
    matrices: Dict[str, np.ndarray],
    class_names: Tuple[str, ...],
    plot_path: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Phase 7] matplotlib not installed; skipping confusion plot")
        return

    names = list(matrices.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.0))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        cm = np.asarray(matrices[name])
        im = ax.imshow(cm, cmap="Blues", aspect="equal")
        ax.set_title(f"{name}")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, str(int(cm[i, j])),
                    ha="center", va="center",
                    color=("white" if cm[i, j] > thresh else "black"),
                    fontsize=10,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Phase 7 - confusion matrices (validation set)")
    fig.tight_layout()

    parent = os.path.dirname(os.path.abspath(plot_path)) or "."
    os.makedirs(parent, exist_ok=True)
    fig.savefig(plot_path, dpi=120)
    print(f"[Phase 7] confusion matrices saved -> {plot_path}")
    plt.close(fig)