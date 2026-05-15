from __future__ import annotations

from typing import Dict, List, Optional
import random

import numpy as np

from P0_P1_project_config import HanabiConfig
from P0_observation_decoding import KnowledgeDecoder, ObservationDecoder, agent_name_to_index
from P1_game_state_simulation import build_state_from_observation
from P5_belief_sampling import BeliefSampler, BeliefStats, build_own_hand_beliefs
from P0_P1_environment import PettingZooHanabiRunner
from P1_P5_P6_P8_P10_controllers import (
    Controller,
    MaximaxController,
    RandomController,
    legal_actions_from_mask,
)


# shared helper used by phase 1 and phase 4 to run a batch of self-play games with one controller type
def _evaluate_controller(
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
        out = runner.run_episode(controllers, seed=seed + ep)
        scores.append(int(out["score"]))
        if post_episode is not None:
            # hook for harvesting diagnostics like belief stats after each game
            post_episode(controllers, int(out["score"]))
    return {
        "mean": sum(scores) / max(1, len(scores)),
        "min": min(scores) if scores else 0,
        "max": max(scores) if scores else 0,
        "scores": scores,
    }


def run_phase1_compare(num_episodes: int = 20, seed: int = 0) -> None:
    cfg = HanabiConfig()
    runner = PettingZooHanabiRunner(cfg)

    rand_stats = _evaluate_controller(
        cfg, runner,
        make_controller=lambda i: RandomController(f"rand{i}", cfg),
        num_episodes=num_episodes, seed=seed,
    )
    max_stats = _evaluate_controller(
        cfg, runner,
        make_controller=lambda i: MaximaxController(f"max{i}", cfg),
        num_episodes=num_episodes, seed=seed,
    )

    print(f"[Phase 1] ran {num_episodes} episodes / {cfg.players} players each")
    print(
        f"  RandomController  mean={rand_stats['mean']:.2f} "
        f"min={rand_stats['min']} max={rand_stats['max']}"
    )
    print(
        f"  MaximaxController mean={max_stats['mean']:.2f} "
        f"min={max_stats['min']} max={max_stats['max']}"
    )
    if max_stats["mean"] <= rand_stats["mean"]:
        print("  (!) Maximax did not outperform Random on this sample.")


def run_phase5_belief_check(
    num_checks: int = 5,
    num_samples: int = 25,
    seed: int = 0,
) -> None:
    cfg = HanabiConfig()
    runner = PettingZooHanabiRunner(cfg)
    env = runner.make_env()
    env.reset(seed=seed)

    obs_decoder = ObservationDecoder(cfg)
    kn_decoder = KnowledgeDecoder(cfg)
    rng = random.Random(seed)
    stats = BeliefStats()
    checked = 0
    total_samples = 0

    try:
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            if termination or truncation:
                env.step(None)
                continue

            observer = agent_name_to_index(agent)
            vec = obs["observation"]
            decoded = obs_decoder.decode(vec, agent)
            knowledge = kn_decoder.decode(vec)
            state = build_state_from_observation(decoded, cfg, observer)
            beliefs = build_own_hand_beliefs(state, knowledge, observer)
            sampler = BeliefSampler(beliefs, rng=rng, stats=stats)
            samples = sampler.sample_many(num_samples)

            if beliefs.num_unknown and not samples:
                raise AssertionError(
                    f"Belief sampler produced no samples for {agent}"
                )

            for sample in samples:
                used = np.zeros_like(beliefs.remaining)
                for local_idx, card in enumerate(sample.cards):
                    if beliefs.slot_counts[local_idx][card.color, card.rank] <= 0:
                        raise AssertionError(
                            f"sampled impossible card {card} for {agent}"
                        )
                    used[card.color, card.rank] += 1
                # verify the sampled hand doesn't exceed the global remaining card budget
                if np.any(used > beliefs.remaining):
                    raise AssertionError(
                        f"sample exceeded remaining card budget for {agent}"
                    )

            checked += 1
            total_samples += len(samples)

            legal = legal_actions_from_mask(obs["action_mask"])
            env.step(rng.choice(legal) if legal else None)
            if checked >= num_checks:
                break
    finally:
        env.close()

    print(
        f"[Phase 5] checked {checked} live observations and "
        f"{total_samples} sampled hands"
    )
    print(f"[Phase 5] {stats.summary()}")
