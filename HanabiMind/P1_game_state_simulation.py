from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from P0_observation_decoding import (
    ActionType,
    Card,
    DecodedObservation,
    StructuredAction,
    copies_per_rank,
)


@dataclass
class GameState:
    cfg: object
    fireworks: np.ndarray            # shape (colors,), how many ranks played per color
    discards: np.ndarray             # shape (colors, ranks), count of discarded cards
    hands: List[List[Optional[Card]]]  # hands[p][s] is None for unknown own-hand slots
    deck_size: int
    info_tokens: int
    life_tokens: int
    current_player: int
    turns_remaining: Optional[int] = None   # None until deck empties, then counts down

    @property
    def is_alive(self) -> bool:
        return self.life_tokens > 0

    @property
    def fireworks_sum(self) -> int:
        return int(self.fireworks.sum())

    @property
    def max_score(self) -> int:
        return int(self.cfg.colors * self.cfg.ranks)

    @property
    def is_complete(self) -> bool:
        return self.fireworks_sum == self.max_score

    def is_terminal(self) -> bool:
        if not self.is_alive:
            return True
        if self.is_complete:
            return True
        # turns_remaining hits zero after all players have had one more turn post-deck
        if self.turns_remaining is not None and self.turns_remaining <= 0:
            return True
        return False

    def final_score(self) -> int:
        # hanabi tournament rule: if you lose all lives the score is zero
        if not self.is_alive:
            return 0
        return self.fireworks_sum

    def clone(self) -> "GameState":
        return GameState(
            cfg=self.cfg,
            fireworks=self.fireworks.copy(),
            discards=self.discards.copy(),
            hands=[list(h) for h in self.hands],
            deck_size=self.deck_size,
            info_tokens=self.info_tokens,
            life_tokens=self.life_tokens,
            current_player=self.current_player,
            turns_remaining=self.turns_remaining,
        )

    def step(
        self,
        action: StructuredAction,
        acting: int,
    ) -> "GameState":
        # return a no-op clone if the game is already over
        if self.is_terminal():
            return self.clone()

        ns = self.clone()
        N = ns.cfg.players
        R = ns.cfg.ranks
        max_info = ns.cfg.max_information_tokens

        if action.type is ActionType.PLAY:
            card = ns.hands[acting][action.slot]
            if card is None:
                raise ValueError(
                    f"Cannot PLAY: slot {action.slot} has no committed card"
                )
            c, r = card.color, card.rank
            if ns.fireworks[c] == r:
                # successful play: advance this color's fireworks stack
                ns.fireworks[c] = r + 1
                # completing a full color stack returns an info token if there is room
                if r + 1 == R and ns.info_tokens < max_info:
                    ns.info_tokens += 1
            else:
                # wrong card: lose a life and send it to discards
                ns.life_tokens -= 1
                ns.discards[c, r] += 1
            ns._consume_and_draw(acting, action.slot)

        elif action.type is ActionType.DISCARD:
            card = ns.hands[acting][action.slot]
            if card is None:
                raise ValueError(
                    f"Cannot DISCARD: slot {action.slot} has no committed card"
                )
            c, r = card.color, card.rank
            ns.discards[c, r] += 1
            # discarding always returns an info token unless already at max
            if ns.info_tokens < max_info:
                ns.info_tokens += 1
            ns._consume_and_draw(acting, action.slot)

        elif action.type in (ActionType.REVEAL_COLOR, ActionType.REVEAL_RANK):
            if ns.info_tokens <= 0:
                raise ValueError("Cannot HINT: no information tokens")
            ns.info_tokens -= 1

        else:
            raise ValueError(f"Unknown action type: {action.type}")

        ns.current_player = (ns.current_player + 1) % N
        if ns.turns_remaining is not None:
            ns.turns_remaining -= 1
        return ns

    def _consume_and_draw(self, acting: int, slot: int) -> None:
        # remove the played or discarded card from the hand
        self.hands[acting].pop(slot)
        if self.deck_size > 0:
            self.deck_size -= 1
            # append None as a placeholder for the unknown drawn card
            self.hands[acting].append(None)
            # once the deck empties, start the final countdown of N more turns
            if self.deck_size == 0 and self.turns_remaining is None:
                self.turns_remaining = self.cfg.players


def build_state_from_observation(
    decoded: DecodedObservation,
    cfg,
    observer_index: int,
) -> GameState:
    N, H = cfg.players, cfg.hand_size

    hands: List[List[Optional[Card]]] = [[] for _ in range(N)]

    # the observer's own cards are unknown, so fill their slots with None
    own_size = H - (1 if decoded.missing_card[0] else 0)
    hands[observer_index] = [None] * own_size

    for p in range(N - 1):
        # convert relative partner offset to absolute seat index
        partner_idx = (observer_index + 1 + p) % N
        partner_hand: List[Optional[Card]] = list(decoded.partner_hands[p])
        # strip any trailing None entries that come from the encoding when hands are short
        while partner_hand and partner_hand[-1] is None:
            partner_hand.pop()
        hands[partner_idx] = partner_hand

    fireworks = np.asarray(decoded.fireworks, dtype=int)
    discards = decoded.discards.astype(int).copy()

    turns_remaining: Optional[int] = None
    # if the deck is already empty when we build the state, the countdown is already running
    if decoded.deck_size == 0:
        turns_remaining = cfg.players

    return GameState(
        cfg=cfg,
        fireworks=fireworks,
        discards=discards,
        hands=hands,
        deck_size=decoded.deck_size,
        info_tokens=decoded.information_tokens,
        life_tokens=decoded.life_tokens,
        current_player=observer_index,
        turns_remaining=turns_remaining,
    )


def pool_remaining(state: GameState, observer_index: int) -> np.ndarray:
    # counts how many of each card could still be in the deck or the observer's own hand
    C, R = state.cfg.colors, state.cfg.ranks
    remaining = np.zeros((C, R), dtype=int)
    for c in range(C):
        for r in range(R):
            total = copies_per_rank(r, R)
            played = 1 if state.fireworks[c] > r else 0
            discarded = int(state.discards[c, r])
            visible = 0
            for p in range(state.cfg.players):
                if p == observer_index:
                    continue   # skip own hand since those cards are hidden
                for card in state.hands[p]:
                    if card is not None and card.color == c and card.rank == r:
                        visible += 1
            remaining[c, r] = total - played - discarded - visible
    return remaining


def compute_wall(state: GameState) -> np.ndarray:
    # for each color, finds the lowest rank that is permanently unplayable due to full discards
    C, R = state.cfg.colors, state.cfg.ranks
    wall = np.full(C, R, dtype=int)  # default to R meaning no wall yet
    for c in range(C):
        for r in range(int(state.fireworks[c]), R):
            copies = copies_per_rank(r, R)
            if int(state.discards[c, r]) >= copies:
                # all copies of this rank are gone, the stack is permanently blocked here
                wall[c] = r
                break
    return wall


def count_dead_cards(state: GameState) -> int:
    # counts visible cards that can never be played, either already played or past the wall
    wall = compute_wall(state)
    count = 0
    for hand in state.hands:
        for card in hand:
            if card is None:
                continue
            c, r = card.color, card.rank
            if r < int(state.fireworks[c]) or r >= int(wall[c]):
                count += 1
    return count


@dataclass
class EvalWeights:
    # tunable multipliers for the static board evaluation function
    fireworks: float = 10.0          # reward per card played
    information: float = 0.5         # small bonus for having tokens available
    life: float = 4.0                # cost per life token already spent
    dead_card_penalty: float = 0.3   # mild penalty for holding cards that can't be played
    loss_penalty: float = 100.0      # heavy penalty for losing all lives
    completion_bonus: float = 50.0   # bonus for achieving a perfect game


def evaluate(state: GameState, weights: EvalWeights = EvalWeights()) -> float:
    fsum = state.fireworks_sum

    if not state.is_alive:
        # still give partial credit for how far the fireworks got before losing
        return -weights.loss_penalty + weights.fireworks * fsum

    score = weights.fireworks * fsum
    score += weights.information * state.info_tokens
    score += weights.life * state.life_tokens
    score -= weights.dead_card_penalty * count_dead_cards(state)

    if state.is_complete:
        score += weights.completion_bonus

    return float(score)


__all__ = [
    "GameState",
    "build_state_from_observation",
    "pool_remaining",
    "compute_wall",
    "count_dead_cards",
    "EvalWeights",
    "evaluate",
]
