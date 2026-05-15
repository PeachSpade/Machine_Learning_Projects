from __future__ import annotations

from dataclasses import dataclass


# central config object passed around to every module so all sizes stay consistent
@dataclass
class HanabiConfig:
    colors: int = 5           # number of card colors (suits) in the deck
    ranks: int = 5            # number of ranks per color (1 through 5)
    players: int = 5          # number of seats at the table
    hand_size: int = 4        # cards each player holds at a time
    max_information_tokens: int = 8   # max clue tokens available
    max_life_tokens: int = 3          # bombs before the game ends
