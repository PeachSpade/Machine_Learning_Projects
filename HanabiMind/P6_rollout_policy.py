from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from P0_observation_decoding import (
    ActionType,
    Card,
    KnowledgeView,
    StructuredAction,
)
from P1_game_state_simulation import (
    GameState,
    compute_wall,
    pool_remaining,
)


@dataclass
class RolloutKnowledge:
    # per-player, per-slot possibility masks that get updated as simulated actions happen
    cfg: object
    masks: List[List[np.ndarray]]   # masks[abs_player][slot] is a (colors, ranks) bool array

    @classmethod
    def from_view(
        cls,
        knowledge: KnowledgeView,
        observer: int,
        state: GameState,
        cfg,
    ) -> "RolloutKnowledge":
        N = cfg.players
        masks: List[Optional[List[np.ndarray]]] = [None] * N
        for rel in range(N):
            # convert relative seat offset to absolute player index
            abs_idx = (observer + rel) % N
            hand_len = len(state.hands[abs_idx])
            slots = knowledge.per_player[rel]
            masks[abs_idx] = [
                slots[s].possibility.copy() for s in range(min(hand_len, len(slots)))
            ]
            # pad with full possibility masks if the hand is longer than the encoded knowledge
            while len(masks[abs_idx]) < hand_len:
                masks[abs_idx].append(_full_mask(cfg))
        return cls(cfg=cfg, masks=[m for m in masks if m is not None])

    def clone(self) -> "RolloutKnowledge":
        return RolloutKnowledge(
            cfg=self.cfg,
            masks=[[m.copy() for m in slots] for slots in self.masks],
        )

    def apply_action(
        self,
        state_before: GameState,
        action: StructuredAction,
        acting: int,
    ) -> None:
        if action.type in (ActionType.REVEAL_COLOR, ActionType.REVEAL_RANK):
            target = (acting + action.target_offset) % self.cfg.players
            # pass the pre-action hand so we can check which cards the hint touched
            self._apply_hint(target, state_before.hands[target], action)
        elif action.type in (ActionType.PLAY, ActionType.DISCARD):
            drew_new = state_before.deck_size > 0
            self._apply_play_discard(acting, action.slot, drew_new)

    def _apply_hint(
        self,
        target: int,
        target_hand: List[Optional[Card]],
        action: StructuredAction,
    ) -> None:
        slots = self.masks[target]
        for slot, card in enumerate(target_hand):
            if slot >= len(slots):
                break
            if card is None:
                continue
            m = slots[slot]
            if action.type is ActionType.REVEAL_COLOR:
                c = int(action.color)
                if int(card.color) == c:
                    # positive match: collapse mask to only the hinted color row
                    keep = np.zeros_like(m)
                    keep[c, :] = True
                    np.bitwise_and(m, keep, out=m)
                else:
                    # negative match: zero out the hinted color row
                    m[c, :] = False
            elif action.type is ActionType.REVEAL_RANK:
                r = int(action.rank)
                if int(card.rank) == r:
                    # positive match: collapse mask to only the hinted rank column
                    keep = np.zeros_like(m)
                    keep[:, r] = True
                    np.bitwise_and(m, keep, out=m)
                else:
                    # negative match: zero out the hinted rank column
                    m[:, r] = False

    def _apply_play_discard(
        self,
        acting: int,
        slot: int,
        drew_new: bool,
    ) -> None:
        slots = self.masks[acting]
        if 0 <= slot < len(slots):
            slots.pop(slot)
        if drew_new:
            # newly drawn card is completely unknown so it gets a full possibility mask
            slots.append(_full_mask(self.cfg))


def _full_mask(cfg) -> np.ndarray:
    # all cards are possible until hints eliminate them
    return np.ones((cfg.colors, cfg.ranks), dtype=bool)


def _playable_matrix(state: GameState) -> np.ndarray:
    # returns a (colors, ranks) bool mask where True means this card is the next needed one
    C, R = state.cfg.colors, state.cfg.ranks
    mat = np.zeros((C, R), dtype=bool)
    for c in range(C):
        r = int(state.fireworks[c])
        if r < R:
            mat[c, r] = True
    return mat


def _dead_matrix(state: GameState) -> np.ndarray:
    # true for any card that has either already been played or is past the discard wall
    C, R = state.cfg.colors, state.cfg.ranks
    fireworks = state.fireworks
    wall = compute_wall(state)
    mat = np.zeros((C, R), dtype=bool)
    for c in range(C):
        f = int(fireworks[c])
        w = int(wall[c])
        for r in range(R):
            if r < f or r >= w:
                mat[c, r] = True
    return mat


def _critical_matrix(state: GameState, acting: int) -> np.ndarray:
    # true for any (color, rank) that has exactly one copy left unaccounted for
    return pool_remaining(state, acting) == 1


def _is_certainly_playable(mask: np.ndarray, playable: np.ndarray) -> bool:
    # every card still in the possibility set must be the next needed card
    return bool(mask.any()) and not bool((mask & ~playable).any())


def _is_certainly_dead(mask: np.ndarray, dead: np.ndarray) -> bool:
    # every card still in the possibility set must be permanently unplayable
    return bool(mask.any()) and not bool((mask & ~dead).any())


def _candidate_hints(target_hand: List[Optional[Card]]) -> List[Tuple[ActionType, int]]:
    # builds the list of all hints that would touch at least one card in the target hand
    colors, ranks = set(), set()
    for card in target_hand:
        if card is not None:
            colors.add(int(card.color))
            ranks.add(int(card.rank))
    out: List[Tuple[ActionType, int]] = []
    for c in sorted(colors):
        out.append((ActionType.REVEAL_COLOR, c))
    for r in sorted(ranks):
        out.append((ActionType.REVEAL_RANK, r))
    return out


def _simulate_hint(
    target_masks: List[np.ndarray],
    target_hand: List[Optional[Card]],
    hint_type: ActionType,
    hint_value: int,
) -> List[np.ndarray]:
    # returns new mask list after applying a hint, without modifying the originals
    new_masks = [m.copy() for m in target_masks]
    for slot, card in enumerate(target_hand):
        if slot >= len(new_masks):
            break
        if card is None:
            continue
        m = new_masks[slot]
        if hint_type is ActionType.REVEAL_COLOR:
            if int(card.color) == hint_value:
                keep = np.zeros_like(m)
                keep[hint_value, :] = True
                np.bitwise_and(m, keep, out=m)
            else:
                m[hint_value, :] = False
        else:
            if int(card.rank) == hint_value:
                keep = np.zeros_like(m)
                keep[:, hint_value] = True
                np.bitwise_and(m, keep, out=m)
            else:
                m[:, hint_value] = False
    return new_masks


# tunable hint scoring weights used by the phase 10 hint evaluator
DEFAULT_HINT_SCORE_TUNE: Dict[str, float] = {
    # extra reward when a rank hint positively touches the card currently needed by a color
    "w_rank_immediate": 2.5,
    # penalty when a color hint matches many cards (broad hints carry less information)
    "w_color_breadth": 0.35,
}


def _hint_score(
    target_masks: List[np.ndarray],
    target_hand: List[Optional[Card]],
    hint_type: ActionType,
    hint_value: int,
    playable: np.ndarray,
    old_certain: List[bool],
) -> Tuple[int, int]:
    new_certain = 0
    touched = 0
    n = len(target_masks)
    for slot in range(n):
        if slot >= len(target_hand):
            break
        card = target_hand[slot]
        if card is None:
            continue
        m = target_masks[slot]
        was_certain = old_certain[slot]

        if hint_type is ActionType.REVEAL_COLOR:
            c = hint_value
            if int(card.color) == c:
                row = m[c, :]
                if not row.any():
                    continue
                if int(m.sum()) != int(row.sum()):
                    touched += 1
                row_playable = playable[c, :]
                certain_after = bool(row.any()) and not bool((row & ~row_playable).any())
            else:
                if bool(m[c, :].any()):
                    touched += 1
                rest_any = bool(m.any()) and (
                    int(m.sum()) - int(m[c, :].sum()) > 0
                )
                if not rest_any:
                    certain_after = False
                else:
                    bad = m & ~playable
                    bad[c, :] = False
                    certain_after = not bool(bad.any())
        else:
            r = hint_value
            if int(card.rank) == r:
                col = m[:, r]
                if not col.any():
                    continue
                if int(m.sum()) != int(col.sum()):
                    touched += 1
                col_playable = playable[:, r]
                certain_after = bool(col.any()) and not bool((col & ~col_playable).any())
            else:
                if bool(m[:, r].any()):
                    touched += 1
                rest_any = bool(m.any()) and (
                    int(m.sum()) - int(m[:, r].sum()) > 0
                )
                if not rest_any:
                    certain_after = False
                else:
                    bad = m & ~playable
                    bad[:, r] = False
                    certain_after = not bool(bad.any())

        if certain_after and not was_certain:
            new_certain += 1
    return new_certain, touched


def score_hint_candidate_phase10(
    target_masks: List[np.ndarray],
    target_hand: List[Optional[Card]],
    hint_type: ActionType,
    hint_value: int,
    playable: np.ndarray,
    fireworks: Optional[np.ndarray] = None,
    hint_tune: Optional[Dict[str, float]] = None,
) -> float:
    if not target_masks or not any(m.any() for m in target_masks):
        return -1e9
    not_playable = ~playable
    new_masks = _simulate_hint(target_masks, target_hand, hint_type, hint_value)
    touched = 0
    for a, b in zip(target_masks, new_masks):
        if not np.array_equal(a, b):
            touched += 1
    # redundant hint: nothing changed, not worth a token
    if touched == 0:
        return -1e9

    def is_cert(m: np.ndarray) -> bool:
        return bool(m.any()) and not bool((m & ~playable).any())

    new_c = 0
    for i, nm in enumerate(new_masks):
        om = target_masks[i] if i < len(target_masks) else nm
        if is_cert(nm) and not is_cert(om):
            new_c += 1

    # measure how much uncertainty was removed (non-playable bits eliminated by the hint)
    old_bad = 0
    new_bad = 0
    for m in target_masks:
        old_bad += int((m & not_playable).sum())
    for m in new_masks:
        new_bad += int((m & not_playable).sum())
    fut = max(0, old_bad - new_bad)

    # count slots that remain ambiguous after the hint (useful but not certain)
    amb = 0
    for m in new_masks:
        if m.any() and (m & playable).any() and (m & not_playable).any():
            amb += 1

    t = dict(DEFAULT_HINT_SCORE_TUNE)
    if hint_tune:
        t.update(hint_tune)

    rank_bonus = 0.0
    color_breadth_pen = 0.0
    if fireworks is not None:
        f = np.asarray(fireworks, dtype=int).reshape(-1)
        if hint_type is ActionType.REVEAL_RANK:
            r = int(hint_value)
            w_imm = float(t.get("w_rank_immediate", 0.0))
            for card in target_hand:
                if card is None:
                    continue
                c = int(card.color)
                if c >= f.shape[0]:
                    continue
                # bonus if this rank hint touches the exact card currently needed for this color
                if int(card.rank) == r and int(f[c]) == int(card.rank):
                    rank_bonus += w_imm
        elif hint_type is ActionType.REVEAL_COLOR:
            c0 = int(hint_value)
            w_br = float(t.get("w_color_breadth", 0.0))
            n_match = sum(
                1 for card in target_hand
                if card is not None and int(card.color) == c0
            )
            # penalise color hints that touch many cards since they give less targeted info
            if n_match > 1 and w_br > 0.0:
                color_breadth_pen += w_br * float(n_match - 1)

    w_play = 3.0
    w_fut = 1.0
    w_touch = 0.5
    w_amb = 0.4
    return float(
        w_play * new_c
        + w_fut * fut
        + w_touch * touched
        - w_amb * amb
        + rank_bonus
        - color_breadth_pen
    )


def _hint_touch_count(
    target_masks: List[np.ndarray],
    target_hand: List[Optional[Card]],
    hint_type: ActionType,
    hint_value: int,
) -> int:
    # counts how many slots would have their possibility mask changed by this hint
    touched = 0
    n = len(target_masks)
    for slot in range(n):
        if slot >= len(target_hand):
            break
        card = target_hand[slot]
        if card is None:
            continue
        m = target_masks[slot]
        if hint_type is ActionType.REVEAL_COLOR:
            c = hint_value
            if int(card.color) == c:
                if int(m.sum()) != int(m[c, :].sum()):
                    touched += 1
            else:
                if bool(m[c, :].any()):
                    touched += 1
        else:
            r = hint_value
            if int(card.rank) == r:
                if int(m.sum()) != int(m[:, r].sum()):
                    touched += 1
            else:
                if bool(m[:, r].any()):
                    touched += 1
    return touched


@dataclass
class _SlotStats:
    # precomputed scalar statistics for a single hand slot, used to avoid repeated numpy work
    total_sum: int
    bad_sum: int           # bits set in mask that are not currently playable
    row_sum: List[int]     # total bits per color row
    row_bad_sum: List[int] # non-playable bits per color row
    col_sum: List[int]     # total bits per rank column
    col_bad_sum: List[int] # non-playable bits per rank column
    is_certain_playable: bool


def _has_uncertain_playable_target(
    target_masks: List[np.ndarray],
    playable: np.ndarray,
    not_playable: np.ndarray,
) -> bool:
    # quick pre-check: returns false if no hint could possibly flip a slot to certainly playable
    for m in target_masks:
        if not m.any():
            continue
        if (m & not_playable).any() and (m & playable).any():
            return True
    return False


def _build_slot_stats(
    target_masks: List[np.ndarray],
    playable: np.ndarray,
    not_playable: Optional[np.ndarray] = None,
) -> List[Optional[_SlotStats]]:
    # precomputes row/column sums for all slots at once using a stacked numpy array
    if not target_masks:
        return []
    if not_playable is None:
        not_playable = ~playable

    non_empty: List[np.ndarray] = []
    indices: List[int] = []
    for slot, m in enumerate(target_masks):
        if m.any():
            non_empty.append(m)
            indices.append(slot)

    out: List[Optional[_SlotStats]] = [None] * len(target_masks)
    if not non_empty:
        return out

    # stack all non-empty masks into a single array so numpy does one pass over all of them
    mask_arr = np.stack(non_empty, axis=0)
    bad_arr = mask_arr & not_playable
    row_sum_arr = mask_arr.sum(axis=2)
    col_sum_arr = mask_arr.sum(axis=1)
    row_bad_arr = bad_arr.sum(axis=2)
    col_bad_arr = bad_arr.sum(axis=1)
    total_arr = row_sum_arr.sum(axis=1)
    bad_total_arr = row_bad_arr.sum(axis=1)

    for i, slot in enumerate(indices):
        total = int(total_arr[i])
        bad_total = int(bad_total_arr[i])
        out[slot] = _SlotStats(
            total_sum=total,
            bad_sum=bad_total,
            row_sum=row_sum_arr[i].tolist(),
            row_bad_sum=row_bad_arr[i].tolist(),
            col_sum=col_sum_arr[i].tolist(),
            col_bad_sum=col_bad_arr[i].tolist(),
            is_certain_playable=(total > 0 and bad_total == 0),
        )
    return out


def _fast_strong_hint_score(
    slot_stats: List[Optional[_SlotStats]],
    target_hand: List[Optional[Card]],
    hint_type: ActionType,
    hint_value: int,
) -> Tuple[int, int]:
    # fast integer-only version of the hint evaluator, avoids any numpy allocations
    new_certain = 0
    touched = 0
    n = min(len(slot_stats), len(target_hand))
    for slot in range(n):
        s = slot_stats[slot]
        if s is None:
            continue
        card = target_hand[slot]
        if card is None:
            continue

        if hint_type is ActionType.REVEAL_COLOR:
            c = hint_value
            if int(card.color) == c:
                row_sum = s.row_sum[c]
                if row_sum == 0:
                    continue
                if row_sum != s.total_sum:
                    touched += 1
                if not s.is_certain_playable and s.row_bad_sum[c] == 0:
                    new_certain += 1
            else:
                row_sum = s.row_sum[c]
                if row_sum > 0:
                    touched += 1
                rest_sum = s.total_sum - row_sum
                if rest_sum > 0 and not s.is_certain_playable:
                    rest_bad = s.bad_sum - s.row_bad_sum[c]
                    if rest_bad == 0:
                        new_certain += 1
        else:
            r = hint_value
            if int(card.rank) == r:
                col_sum = s.col_sum[r]
                if col_sum == 0:
                    continue
                if col_sum != s.total_sum:
                    touched += 1
                if not s.is_certain_playable and s.col_bad_sum[r] == 0:
                    new_certain += 1
            else:
                col_sum = s.col_sum[r]
                if col_sum > 0:
                    touched += 1
                rest_sum = s.total_sum - col_sum
                if rest_sum > 0 and not s.is_certain_playable:
                    rest_bad = s.bad_sum - s.col_bad_sum[r]
                    if rest_bad == 0:
                        new_certain += 1
    return new_certain, touched


def _ordered_promising_candidates(
    target_hand: List[Optional[Card]],
    fireworks,
) -> List[Tuple[ActionType, int]]:
    # puts hints that positively touch a currently playable card at the front of the list
    colors_present, ranks_present = set(), set()
    promising_colors, promising_ranks = set(), set()
    for card in target_hand:
        if card is None:
            continue
        c = int(card.color)
        r = int(card.rank)
        colors_present.add(c)
        ranks_present.add(r)
        if r == int(fireworks[c]):
            promising_colors.add(c)
            promising_ranks.add(r)

    promising: List[Tuple[ActionType, int]] = []
    rest: List[Tuple[ActionType, int]] = []
    for c in sorted(colors_present):
        bucket = promising if c in promising_colors else rest
        bucket.append((ActionType.REVEAL_COLOR, c))
    for r in sorted(ranks_present):
        bucket = promising if r in promising_ranks else rest
        bucket.append((ActionType.REVEAL_RANK, r))
    return promising + rest


class RolloutPolicy:
    # deterministic policy used inside the rollout simulation to play out future turns

    DEFAULT_HINT_EVAL_BUDGET = 12

    def __init__(
        self,
        risky_play_threshold: float = 1.0,   # 1.0 means only guaranteed plays, lower allows guesses
        hint_eval_budget: int = DEFAULT_HINT_EVAL_BUDGET,
        early_exit_strong_hint: bool = True,  # when true, return the first hint that creates a new certain play
        hint_score_tune: Optional[Dict[str, float]] = None,
    ) -> None:
        self.risky_play_threshold = float(risky_play_threshold)
        self.hint_eval_budget = max(1, int(hint_eval_budget))
        self.early_exit_strong_hint = bool(early_exit_strong_hint)
        self.hint_score_tune: Optional[Dict[str, float]] = (
            dict(hint_score_tune) if hint_score_tune else None
        )

    def select(
        self,
        state: GameState,
        knowledge: RolloutKnowledge,
        acting: int,
    ) -> Optional[StructuredAction]:
        if state.is_terminal():
            return None
        if not state.hands[acting]:
            return None

        max_info = state.cfg.max_information_tokens
        playable = _playable_matrix(state)

        masks = knowledge.masks[acting]

        # priority 1: play a card we are guaranteed to know is playable
        play_slot = self._find_guaranteed_play(masks, playable)
        if play_slot is not None:
            return StructuredAction(ActionType.PLAY, slot=play_slot)

        # priority 2: give a hint that unlocks a new guaranteed play for a partner
        if state.info_tokens > 0:
            hint = self._find_strong_hint(state, knowledge, acting, playable)
            if hint is not None:
                return hint

        # priority 3: discard the safest card (dead first, then non-critical, then non-5)
        if state.info_tokens < max_info:
            dead = _dead_matrix(state)
            critical = _critical_matrix(state, acting)
            slot = self._find_safe_discard(state, masks, dead, critical)
            if slot is not None:
                return StructuredAction(ActionType.DISCARD, slot=slot)

        # priority 4: give any non-redundant hint if we are token-locked
        if state.info_tokens > 0:
            hint = self._find_useful_hint(state, knowledge, acting)
            if hint is not None:
                return hint

        # priority 5: last resort, play the slot with highest probability of being playable
        play_slot = self._find_probable_play(state, masks, playable, acting)
        if play_slot is not None:
            return StructuredAction(ActionType.PLAY, slot=play_slot)

        if state.info_tokens < max_info and len(state.hands[acting]) > 0:
            return StructuredAction(ActionType.DISCARD, slot=0)
        return StructuredAction(ActionType.PLAY, slot=0)

    @staticmethod
    def _find_guaranteed_play(
        masks: List[np.ndarray],
        playable: np.ndarray,
    ) -> Optional[int]:
        for slot, mask in enumerate(masks):
            if _is_certainly_playable(mask, playable):
                return slot
        return None

    def _find_strong_hint(
        self,
        state: GameState,
        knowledge: RolloutKnowledge,
        acting: int,
        playable: np.ndarray,
    ) -> Optional[StructuredAction]:
        N = state.cfg.players
        budget = self.hint_eval_budget
        early_exit = self.early_exit_strong_hint
        evaluated = 0
        fireworks = state.fireworks
        not_playable = ~playable

        best: Optional[StructuredAction] = None
        best_float: float = -1e18

        for off in range(1, N):
            target = (acting + off) % N
            hand = state.hands[target]
            if not hand:
                continue
            target_masks = knowledge.masks[target]
            if not target_masks or not any(m.any() for m in target_masks):
                continue

            if early_exit:
                # skip this partner entirely if they have no slots where a hint could flip to certain
                if not _has_uncertain_playable_target(
                    target_masks, playable, not_playable,
                ):
                    continue
                slot_stats = _build_slot_stats(target_masks, playable, not_playable)
            else:
                slot_stats = None

            for hint_type, hint_value in _ordered_promising_candidates(hand, fireworks):
                if evaluated >= budget:
                    return best
                evaluated += 1
                if early_exit:
                    new_certain, touched = _fast_strong_hint_score(
                        slot_stats, hand, hint_type, hint_value,
                    )
                    if new_certain == 0:
                        continue
                    return self._build_hint(hint_type, off, hint_value)
                s = score_hint_candidate_phase10(
                    target_masks, hand, hint_type, hint_value, playable,
                    fireworks=fireworks, hint_tune=self.hint_score_tune,
                )
                if s > best_float:
                    best_float = s
                    best = self._build_hint(hint_type, off, hint_value)
        return best

    @staticmethod
    def _find_safe_discard(
        state: GameState,
        masks: List[np.ndarray],
        dead: np.ndarray,
        critical: np.ndarray,
    ) -> Optional[int]:
        if not masks:
            return None
        R = state.cfg.ranks

        # prefer definitely dead cards first
        for slot, mask in enumerate(masks):
            if _is_certainly_dead(mask, dead):
                return slot

        # then non-critical non-5 cards
        for slot, mask in enumerate(masks):
            if not mask.any():
                continue
            if (mask & critical).any():
                continue
            if mask[:, R - 1].any():   # R-1 is the top rank (5 in standard hanabi)
                continue
            return slot

        # then any non-critical card, even if it might be a 5
        for slot, mask in enumerate(masks):
            if not mask.any():
                continue
            if (mask & critical).any():
                continue
            return slot

        # last resort: just take the first non-empty slot
        for slot, mask in enumerate(masks):
            if mask.any():
                return slot
        return 0

    def _find_useful_hint(
        self,
        state: GameState,
        knowledge: RolloutKnowledge,
        acting: int,
    ) -> Optional[StructuredAction]:
        N = state.cfg.players
        budget = self.hint_eval_budget
        evaluated = 0
        playable = _playable_matrix(state)
        fw = state.fireworks
        best: Optional[StructuredAction] = None
        best_f = -1e18
        for off in range(1, N):
            target = (acting + off) % N
            hand = state.hands[target]
            if not hand:
                continue
            target_masks = knowledge.masks[target]
            for hint_type, hint_value in _candidate_hints(hand):
                if evaluated >= budget:
                    return best
                evaluated += 1
                s = score_hint_candidate_phase10(
                    target_masks, hand, hint_type, hint_value, playable,
                    fireworks=fw, hint_tune=self.hint_score_tune,
                )
                if s > best_f:
                    best_f = s
                    best = self._build_hint(hint_type, off, hint_value)
        return best

    def _find_probable_play(
        self,
        state: GameState,
        masks: List[np.ndarray],
        playable: np.ndarray,
        acting: int,
    ) -> Optional[int]:
        # threshold > 1.0 means this last-resort path is disabled in the default config
        if self.risky_play_threshold > 1.0:
            return None
        pool = pool_remaining(state, acting)
        best_slot = None
        best_prob = self.risky_play_threshold - 1e-9
        for slot, mask in enumerate(masks):
            weights = mask.astype(int) * pool
            total = int(weights.sum())
            if total <= 0:
                continue
            prob = float((weights * playable).sum()) / total
            if prob > best_prob:
                best_prob = prob
                best_slot = slot
        return best_slot

    @staticmethod
    def _build_hint(
        hint_type: ActionType,
        target_offset: int,
        hint_value: int,
    ) -> StructuredAction:
        if hint_type is ActionType.REVEAL_COLOR:
            return StructuredAction(
                ActionType.REVEAL_COLOR,
                target_offset=target_offset,
                color=hint_value,
            )
        return StructuredAction(
            ActionType.REVEAL_RANK,
            target_offset=target_offset,
            rank=hint_value,
        )


def simulate_rollout(
    state: GameState,
    knowledge: RolloutKnowledge,
    policy: RolloutPolicy,
    num_turns: int,
) -> GameState:
    # simulates num_turns of play using the rollout policy, mutating knowledge in place
    cur = state
    for _ in range(max(0, int(num_turns))):
        if cur.is_terminal():
            break
        acting = cur.current_player

        if not cur.hands[acting]:
            # skip this seat if they somehow have no cards (shouldn't happen but handled safely)
            cur = _advance_empty_turn(cur)
            continue

        action = policy.select(cur, knowledge, acting)
        if action is None:
            break

        try:
            # knowledge must be updated before stepping so hint matching uses the pre-step hand
            knowledge.apply_action(cur, action, acting)
            cur = cur.step(action, acting)
        except ValueError:
            break
    return cur


def _advance_empty_turn(state: GameState) -> GameState:
    # moves to the next player without taking any action
    ns = state.clone()
    ns.current_player = (ns.current_player + 1) % ns.cfg.players
    if ns.turns_remaining is not None:
        ns.turns_remaining -= 1
    return ns


__all__ = [
    "DEFAULT_HINT_SCORE_TUNE",
    "RolloutKnowledge",
    "RolloutPolicy",
    "simulate_rollout",
    "score_hint_candidate_phase10",
]
