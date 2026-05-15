from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from P0_P1_project_config import HanabiConfig
from P0_observation_decoding import ActionCodec, ObservationDecoder, run_phase0_selfcheck
from P1_P5_P6_P8_P10_controllers import Controller, RandomController
from P2_P7_trajectory_schema import TrajectoryStep


# pettingzoo is an optional dependency, so we handle the missing case gracefully
try:
    from pettingzoo.classic import hanabi_v5
except ImportError:
    hanabi_v5 = None


class PettingZooHanabiRunner:

    def __init__(self, cfg: HanabiConfig):
        self.cfg = cfg

    def make_env(self):
        if hanabi_v5 is None:
            raise RuntimeError(
                "PettingZoo with the Hanabi classic environment is required "
                "for game execution. Install the project dependencies before "
                "running simulation, training, evaluation, or UI phases."
            )
        return hanabi_v5.env(
            colors=self.cfg.colors,
            ranks=self.cfg.ranks,
            players=self.cfg.players,
            hand_size=self.cfg.hand_size,
            max_information_tokens=self.cfg.max_information_tokens,
            max_life_tokens=self.cfg.max_life_tokens,
            observation_type="card_knowledge",   # full card knowledge encoding, not vanilla
        )

    def run_episode(
        self,
        controllers: Dict[str, Controller],
        seed=None,
        log_trajectory: bool = False,
        controller_labels: Optional[Dict[str, str]] = None,
    ):
        env = self.make_env()
        env.reset(seed=seed)

        history: List = []
        last_obs_vec = None
        last_agent = None

        # only build decoders if we actually need the trajectory, saves some overhead
        decoder = ObservationDecoder(self.cfg) if log_trajectory else None
        codec = ActionCodec(self.cfg) if log_trajectory else None

        # fall back to using the class name as a short label if none provided
        labels = controller_labels or {
            name: ctrl.__class__.__name__.replace("Controller", "").lower()
            for name, ctrl in controllers.items()
        }

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            last_obs_vec = obs["observation"]
            last_agent = agent

            if not termination and not truncation:
                action = controllers[agent].select_action(
                    obs["observation"],
                    obs["action_mask"],
                    agent,
                    history,
                )
                if log_trajectory:
                    decoded_obs = decoder.decode(obs["observation"], agent)
                    structured = codec.decode(int(action))
                    history.append(TrajectoryStep(
                        agent=agent,
                        decoded_obs=decoded_obs,
                        action_mask=np.asarray(obs["action_mask"]).copy(),
                        action_id=int(action),
                        decoded_action=structured,
                        reward=float(reward),
                        controller_label=labels.get(agent, "unknown"),
                    ))
            else:
                # terminal steps still need a None step to advance the env
                action = None

            env.step(action)

        env.close()

        # we recover the true score from the final decoded observation because
        # pettingzoo's returns() call gives 0 for unfinished games
        final_decoder = decoder or ObservationDecoder(self.cfg)
        decoded = final_decoder.decode(last_obs_vec, last_agent)
        fireworks_score = int(sum(decoded.fireworks))
        # tournament scoring rule: blowing all lives counts as zero regardless of progress
        if decoded.life_tokens == 0:
            fireworks_score = 0
        return {"history": history, "score": fireworks_score}


def generate_games(num_games=10):
    cfg = HanabiConfig()
    runner = PettingZooHanabiRunner(cfg)

    controllers = {
        f"player_{i}": RandomController(f"p{i}", cfg)
        for i in range(cfg.players)
    }

    for i in range(num_games):
        runner.run_episode(controllers, seed=i)


def run_phase0_check(num_episodes: int = 5, seed: int = 0) -> None:
    cfg = HanabiConfig()
    runner = PettingZooHanabiRunner(cfg)
    report = run_phase0_selfcheck(cfg, runner, num_episodes=num_episodes, seed=seed)
    print(report.summary())
    if not report.ok:
        raise AssertionError("Phase 0 consistency check failed")
