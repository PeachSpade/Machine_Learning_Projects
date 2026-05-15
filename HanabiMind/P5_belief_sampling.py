from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from P0_observation_decoding import Card, KnowledgeView
from P1_game_state_simulation import GameState, pool_remaining


@dataclass
class BeliefStats:
    # diagnostic counters for tracking how the sampler resolves each request
    sample_calls: int = 0
    rejection_success: int = 0   # how many samples were found via rejection sampling
    greedy_fallback: int = 0     # how many fell through to the greedy path
    failures: int = 0            # how many could not produce any sample at all
    rejection_attempts: int = 0  # total loop iterations across all rejection runs
    empty_samples: int = 0       # calls where the observer had no hidden cards to sample

    def merge(self, other: "BeliefStats") -> None:
        self.sample_calls += other.sample_calls
        self.rejection_success += other.rejection_success
        self.greedy_fallback += other.greedy_fallback
        self.failures += other.failures
        self.rejection_attempts += other.rejection_attempts
        self.empty_samples += other.empty_samples

    def summary(self) -> str:
        n = max(1, self.sample_calls)
        attempts_per_call = self.rejection_attempts / n
        return (
            f"sample_calls={self.sample_calls}  "
            f"rejection={self.rejection_success} ({self.rejection_success / n:.1%})  "
            f"greedy={self.greedy_fallback} ({self.greedy_fallback / n:.1%})  "
            f"failed={self.failures} ({self.failures / n:.1%})  "
            f"empty={self.empty_samples}  "
            f"avg_attempts={attempts_per_call:.2f}"
        )


@dataclass
class OwnHandBeliefs:
    # per-slot (colors x ranks) count arrays, already masked by the possibility hints
    slot_counts: List[np.ndarray]
    remaining: np.ndarray    # global unaccounted card counts visible from this seat
    unknown_slots: List[int] # indices of own hand slots that are still hidden

    @property
    def num_unknown(self) -> int:
        return len(self.unknown_slots)

    def feasible(self) -> bool:
        # returns false if any slot has zero compatible cards, meaning no sample is possible
        if not self.unknown_slots:
            return True
        for counts in self.slot_counts:
            if not counts.any():
                return False
        return True


@dataclass
class SampledHand:
    cards: List[Card]
    source: str = "rejection"   # "rejection", "greedy", or "empty"

    def key(self) -> Tuple:
        # hashable representation so we can count unique samples across a batch
        return tuple((c.color, c.rank) for c in self.cards)


def build_own_hand_beliefs(
    state: GameState,
    knowledge: KnowledgeView,
    observer: int,
) -> OwnHandBeliefs:
    own_hand = state.hands[observer]
    remaining = pool_remaining(state, observer)

    # find which of the observer's slots still have unknown cards
    unknown_slots = [s for s, card in enumerate(own_hand) if card is None]

    slot_counts: List[np.ndarray] = []
    for slot in unknown_slots:
        possibility = knowledge.per_player[0][slot].possibility if slot < len(knowledge.per_player[0]) else None
        if possibility is None:
            counts = remaining.copy()
        else:
            # intersect the hint possibility mask with the global remaining pool
            counts = remaining * possibility.astype(int)
        slot_counts.append(counts)

    return OwnHandBeliefs(
        slot_counts=slot_counts,
        remaining=remaining.copy(),
        unknown_slots=unknown_slots,
    )


def commit_sample(
    base_state: GameState,
    sample: SampledHand,
    unknown_slots: List[int],
    observer: int,
) -> GameState:
    # writes a sampled hand hypothesis into a cloned state so we can simulate from it
    hypothesis = base_state.clone()
    for local_idx, slot in enumerate(unknown_slots):
        if local_idx < len(sample.cards) and slot < len(hypothesis.hands[observer]):
            hypothesis.hands[observer][slot] = sample.cards[local_idx]
    return hypothesis


class BeliefSampler:
    def __init__(
        self,
        beliefs: OwnHandBeliefs,
        rng: Optional[random.Random] = None,
        max_attempts: int = 200,
        stats: Optional[BeliefStats] = None,
    ) -> None:
        self.beliefs = beliefs
        self.rng = rng or random.Random()
        self.max_attempts = int(max_attempts)
        self.stats = stats

    def sample_many(self, n: int) -> List[SampledHand]:
        if not self.beliefs.num_unknown:
            if self.stats:
                self.stats.empty_samples += n
            return [SampledHand(cards=[], source="empty")] * n
        results: List[SampledHand] = []
        for _ in range(n):
            s = self.sample()
            if s is not None:
                results.append(s)
        return results

    def sample(self) -> Optional[SampledHand]:
        if self.stats:
            self.stats.sample_calls += 1
        if not self.beliefs.num_unknown:
            if self.stats:
                self.stats.empty_samples += 1
            return SampledHand(cards=[], source="empty")

        # try rejection sampling first since it gives jointly consistent hands
        result = self._try_rejection()
        if result is not None:
            if self.stats:
                self.stats.rejection_success += 1
            return result

        # fall back to greedy if rejection fails after max_attempts tries
        result = self._try_greedy()
        if result is not None:
            if self.stats:
                self.stats.greedy_fallback += 1
            return result

        if self.stats:
            self.stats.failures += 1
        return None

    def _try_rejection(self) -> Optional[SampledHand]:
        # rejection sampling: draw cards slot by slot and restart if the budget runs out
        budget = self.beliefs.remaining.copy()
        n = self.beliefs.num_unknown

        for attempt in range(self.max_attempts):
            if self.stats:
                self.stats.rejection_attempts += 1

            cards: List[Card] = []
            temp_budget = budget.copy()
            failed = False

            for i in range(n):
                # multiply slot possibilities by remaining budget to get valid weighted candidates
                counts = self.beliefs.slot_counts[i] * temp_budget
                total = int(counts.sum())
                if total <= 0:
                    failed = True
                    break

                flat_idx = self._weighted_choice(counts, total)
                c, r = flat_idx // counts.shape[1], flat_idx % counts.shape[1]
                cards.append(Card(color=c, rank=r))
                # reduce the budget so later slots can't pick the same physical card
                temp_budget[c, r] = max(0, temp_budget[c, r] - 1)

            if not failed and len(cards) == n:
                return SampledHand(cards=cards, source="rejection")

        return None

    def _try_greedy(self) -> Optional[SampledHand]:
        # greedy path: commit each slot in order without backtracking, lower quality but always fast
        budget = self.beliefs.remaining.copy()
        cards: List[Card] = []

        for i in range(self.beliefs.num_unknown):
            counts = self.beliefs.slot_counts[i] * budget
            total = int(counts.sum())
            if total <= 0:
                return None

            flat_idx = self._weighted_choice(counts, total)
            c, r = flat_idx // counts.shape[1], flat_idx % counts.shape[1]
            cards.append(Card(color=c, rank=r))
            budget[c, r] = max(0, budget[c, r] - 1)

        if len(cards) == self.beliefs.num_unknown:
            return SampledHand(cards=cards, source="greedy")
        return None

    def _weighted_choice(self, counts: np.ndarray, total: int) -> int:
        # samples a flat index proportional to counts using a single random draw
        target = self.rng.randint(0, total - 1)
        flat = counts.reshape(-1)
        cumsum = 0
        for idx, v in enumerate(flat):
            cumsum += int(v)
            if cumsum > target:
                return idx
        # fallback to last nonzero index in case of floating point edge cases
        return int(np.flatnonzero(flat)[-1])
