from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from P0_observation_decoding import DecodedObservation, StructuredAction


# the three controller types the classifier is trained to distinguish
CONTROLLER_LABELS: Tuple[str, ...] = ("random", "heuristic", "maximax")

# maps each label string to an integer class index for the ML models
LABEL_TO_INDEX: Dict[str, int] = {label: i for i, label in enumerate(CONTROLLER_LABELS)}

# maps controller labels to human-friendly playstyle names shown in the ui
LABEL_TO_PLAYSTYLE: Dict[str, str] = {
    "random": "chaotic",
    "heuristic": "cooperative",
    "maximax": "strategic",
}

# the order playstyle values are displayed in the panel and stored in history
PLAYSTYLE_ORDER: Tuple[str, ...] = ("chaotic", "cooperative", "strategic")


# one recorded step from a self-play game, used as training data and for ml inference
@dataclass
class TrajectoryStep:
    agent: str                        # which seat took this action, e.g. "player_2"
    decoded_obs: DecodedObservation   # the observation this agent saw before acting
    action_mask: np.ndarray           # which action ids were legal at this step
    action_id: int                    # the flat integer action the agent chose
    decoded_action: StructuredAction  # the same action in structured form
    reward: float                     # reward signal from the environment
    controller_label: str             # which controller type produced this step
