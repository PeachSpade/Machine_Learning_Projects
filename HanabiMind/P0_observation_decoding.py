from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# color letter codes in the order the hle encoder uses them
COLOR_NAMES: Tuple[str, ...] = ("R", "Y", "G", "W", "B")


def copies_per_rank(rank: int, num_ranks: int) -> int:
    # hanabi deck distribution: three 1s, one top-rank card, two of everything else
    if rank == 0:
        return 3
    if rank == num_ranks - 1:
        return 1
    return 2


def total_copies_per_color(num_ranks: int) -> int:
    return sum(copies_per_rank(r, num_ranks) for r in range(num_ranks))


def card_index(color: int, rank: int, num_ranks: int) -> int:
    # flattens a (color, rank) pair into a single integer index
    return color * num_ranks + rank


def index_to_card(idx: int, num_ranks: int) -> Tuple[int, int]:
    # reverses card_index back to (color, rank)
    return idx // num_ranks, idx % num_ranks


def format_card(color: int, rank: int) -> str:
    # returns a short readable string like "R3" for display purposes
    return f"{COLOR_NAMES[color]}{rank + 1}"


def agent_name_to_index(agent_name: str) -> int:
    # pettingzoo names seats as "player_0", "player_1" etc, this extracts the number
    return int(agent_name.rsplit("_", 1)[-1])


class ActionType(str, Enum):
    DISCARD = "DISCARD"
    PLAY = "PLAY"
    REVEAL_COLOR = "REVEAL_COLOR"
    REVEAL_RANK = "REVEAL_RANK"


@dataclass(frozen=True)
class StructuredAction:
    type: ActionType
    slot: Optional[int] = None            # hand position for play and discard moves
    target_offset: Optional[int] = None  # how many seats ahead the hint target sits
    color: Optional[int] = None          # color index for reveal color hints
    rank: Optional[int] = None           # rank index for reveal rank hints

    def describe(self) -> str:
        if self.type is ActionType.DISCARD:
            return f"Discard slot {self.slot}"
        if self.type is ActionType.PLAY:
            return f"Play slot {self.slot}"
        if self.type is ActionType.REVEAL_COLOR:
            return f"Reveal +{self.target_offset} color {COLOR_NAMES[self.color]}"
        if self.type is ActionType.REVEAL_RANK:
            return f"Reveal +{self.target_offset} rank {self.rank + 1}"
        raise ValueError(self.type)


class ActionCodec:
    # action ids are laid out in order: discards, plays, color hints, rank hints
    def __init__(self, cfg) -> None:
        self.num_players = cfg.players
        self.hand_size = cfg.hand_size
        self.num_colors = cfg.colors
        self.num_ranks = cfg.ranks

        H, N, C, R = self.hand_size, self.num_players, self.num_colors, self.num_ranks
        self._discard_start = 0
        self._play_start = H                      # plays start after H discards
        self._color_start = 2 * H                 # color hints start after H plays
        self._rank_start = 2 * H + (N - 1) * C   # rank hints start after color hints
        self._total = 2 * H + (N - 1) * (C + R)  # total number of possible actions

    @property
    def total_actions(self) -> int:
        return self._total

    def encode(self, action: StructuredAction) -> int:
        H, C, R = self.hand_size, self.num_colors, self.num_ranks
        if action.type is ActionType.DISCARD:
            self._check_slot(action.slot)
            return self._discard_start + action.slot
        if action.type is ActionType.PLAY:
            self._check_slot(action.slot)
            return self._play_start + action.slot
        if action.type is ActionType.REVEAL_COLOR:
            self._check_offset(action.target_offset)
            if not (0 <= (action.color or 0) < C):
                raise ValueError(f"color out of range: {action.color}")
            # stride by C for each target offset so hints to different seats don't overlap
            return self._color_start + (action.target_offset - 1) * C + action.color
        if action.type is ActionType.REVEAL_RANK:
            self._check_offset(action.target_offset)
            if not (0 <= (action.rank or 0) < R):
                raise ValueError(f"rank out of range: {action.rank}")
            return self._rank_start + (action.target_offset - 1) * R + action.rank
        raise ValueError(action.type)

    def decode(self, action_id: int) -> StructuredAction:
        if not (0 <= action_id < self._total):
            raise ValueError(f"action_id {action_id} out of range [0, {self._total})")

        H, C, R = self.hand_size, self.num_colors, self.num_ranks
        if action_id < self._play_start:
            return StructuredAction(ActionType.DISCARD, slot=action_id - self._discard_start)
        if action_id < self._color_start:
            return StructuredAction(ActionType.PLAY, slot=action_id - self._play_start)
        if action_id < self._rank_start:
            rel = action_id - self._color_start
            # divmod splits the flat offset into (which target, which color)
            offset, color = divmod(rel, C)
            return StructuredAction(
                ActionType.REVEAL_COLOR,
                target_offset=offset + 1,
                color=color,
            )
        rel = action_id - self._rank_start
        offset, rank = divmod(rel, R)
        return StructuredAction(
            ActionType.REVEAL_RANK,
            target_offset=offset + 1,
            rank=rank,
        )

    def describe(self, action_id: int) -> str:
        return self.decode(action_id).describe()

    def _check_slot(self, slot: Optional[int]) -> None:
        if slot is None or not (0 <= slot < self.hand_size):
            raise ValueError(f"slot out of range: {slot}")

    def _check_offset(self, offset: Optional[int]) -> None:
        if offset is None or not (1 <= offset <= self.num_players - 1):
            raise ValueError(f"target_offset out of range: {offset}")


@dataclass(frozen=True)
class SectionOffsets:
    # each tuple is (start_index, length) within the flat observation vector
    hands: Tuple[int, int]
    board: Tuple[int, int]
    discards: Tuple[int, int]
    last_action: Tuple[int, int]
    card_knowledge: Tuple[int, int]
    total: int


def compute_section_offsets(cfg) -> SectionOffsets:
    N, H, C, R = cfg.players, cfg.hand_size, cfg.colors, cfg.ranks
    per_color = total_copies_per_color(R)
    total_deck = per_color * C
    # the deck starts with all cards minus what players hold in their hands
    max_deck_size = total_deck - N * H

    # each partner hand is H slots, each slot is a C*R one-hot card vector
    hands_len = (N - 1) * H * (C * R) + N   # the +N is the missing-card bit per player
    board_len = max_deck_size + C * R + cfg.max_information_tokens + cfg.max_life_tokens
    discards_len = C * per_color
    # last action section encodes who acted, what they did, and the outcome
    last_len = N + 4 + N + C + R + H + H + (C * R) + 2
    kn_len = N * H * (C * R + C + R)

    # build offsets by walking forward through the vector layout
    off = 0
    hands = (off, hands_len); off += hands_len
    board = (off, board_len); off += board_len
    discards = (off, discards_len); off += discards_len
    last_action = (off, last_len); off += last_len
    card_knowledge = (off, kn_len); off += kn_len
    return SectionOffsets(hands, board, discards, last_action, card_knowledge, off)


@dataclass(frozen=True)
class Card:
    color: int
    rank: int

    def __str__(self) -> str:
        return format_card(self.color, self.rank)


@dataclass
class LastAction:
    actor_offset: int              # seat offset from the observer, 0 means the observer themselves acted
    type: ActionType
    target_offset: Optional[int] = None   # only set for hint moves
    color: Optional[int] = None
    rank: Optional[int] = None
    revealed_slots: Optional[List[int]] = None   # which hand slots the hint matched
    slot: Optional[int] = None
    card: Optional[Card] = None
    play_successful: Optional[bool] = None
    info_token_added: Optional[bool] = None      # true when completing a color stack returns a token

    def describe(self) -> str:
        if self.type is ActionType.DISCARD:
            return f"actor+{self.actor_offset} discarded {self.card} from slot {self.slot}"
        if self.type is ActionType.PLAY:
            tag = "played" if self.play_successful else "misplayed"
            extra = " (+1 info)" if self.info_token_added else ""
            return f"actor+{self.actor_offset} {tag} {self.card} from slot {self.slot}{extra}"
        if self.type is ActionType.REVEAL_COLOR:
            return (
                f"actor+{self.actor_offset} revealed color {COLOR_NAMES[self.color]} "
                f"to +{self.target_offset}, slots={self.revealed_slots}"
            )
        if self.type is ActionType.REVEAL_RANK:
            return (
                f"actor+{self.actor_offset} revealed rank {self.rank + 1} "
                f"to +{self.target_offset}, slots={self.revealed_slots}"
            )
        raise ValueError(self.type)


@dataclass
class DecodedObservation:
    observer_index: int
    partner_hands: List[List[Optional[Card]]]   # indexed by relative seat offset (0 = next player)
    missing_card: List[bool]                    # true for any seat that is short a card
    fireworks: List[int]                        # how many ranks have been played per color
    information_tokens: int
    life_tokens: int
    deck_size: int
    discards: np.ndarray          # shape (colors, ranks), counts how many of each card were discarded
    last_action: Optional[LastAction]


@dataclass
class CardKnowledge:
    possibility: np.ndarray    # (colors, ranks) bool mask of cards still consistent with all hints
    revealed_color: Optional[int]   # set if this slot received a positive color hint
    revealed_rank: Optional[int]    # set if this slot received a positive rank hint

    def possible_colors(self) -> List[int]:
        return [c for c in range(self.possibility.shape[0]) if self.possibility[c].any()]

    def possible_ranks(self) -> List[int]:
        return [r for r in range(self.possibility.shape[1]) if self.possibility[:, r].any()]


@dataclass
class KnowledgeView:
    # per_player[p][s] is the knowledge of the player at relative offset p about their slot s
    per_player: List[List[CardKnowledge]]


class ObservationDecoder:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.offsets = compute_section_offsets(cfg)
        self._per_color = total_copies_per_color(cfg.ranks)
        self._max_deck_size = self._per_color * cfg.colors - cfg.players * cfg.hand_size
        # precompute copies per rank so the discard decoder doesn't recalculate each time
        self._copies = [copies_per_rank(r, cfg.ranks) for r in range(cfg.ranks)]

    @property
    def expected_length(self) -> int:
        return self.offsets.total

    def decode(
        self,
        observation_vec: Sequence[float],
        observer: "int | str",
    ) -> DecodedObservation:
        v = np.asarray(observation_vec)
        if v.shape != (self.offsets.total,):
            raise ValueError(
                f"observation has shape {v.shape}, expected ({self.offsets.total},)"
            )
        if isinstance(observer, str):
            observer_index = agent_name_to_index(observer)
        else:
            observer_index = int(observer)

        partner_hands, missing_card = self._decode_hands(v)
        fireworks, info_tokens, life_tokens, deck_size = self._decode_board(v)
        discards = self._decode_discards(v)
        last_action = self._decode_last_action(v)

        return DecodedObservation(
            observer_index=observer_index,
            partner_hands=partner_hands,
            missing_card=missing_card,
            fireworks=fireworks,
            information_tokens=info_tokens,
            life_tokens=life_tokens,
            deck_size=deck_size,
            discards=discards,
            last_action=last_action,
        )

    def _decode_hands(self, v: np.ndarray) -> Tuple[List[List[Optional[Card]]], List[bool]]:
        start, length = self.offsets.hands
        N, H, C, R = (self.cfg.players, self.cfg.hand_size, self.cfg.colors, self.cfg.ranks)
        bits_per_card = C * R
        hand_block_len = (N - 1) * H * bits_per_card
        # reshape into (partners, slots, card_bits) for easy indexing
        hand_bits = v[start : start + hand_block_len].reshape(N - 1, H, bits_per_card)

        partner_hands: List[List[Optional[Card]]] = []
        for p in range(N - 1):
            slots: List[Optional[Card]] = []
            for s in range(H):
                idx_arr = np.flatnonzero(hand_bits[p, s])
                if idx_arr.size == 0:
                    # no bit set means the slot is empty (deck ran out)
                    slots.append(None)
                elif idx_arr.size == 1:
                    color, rank = index_to_card(int(idx_arr[0]), R)
                    slots.append(Card(color, rank))
                else:
                    raise ValueError(
                        f"hand slot has {idx_arr.size} bits set, expected 0 or 1"
                    )
            partner_hands.append(slots)

        # the missing-card bits follow the hand one-hots, one bit per player including the observer
        missing_card = [
            bool(v[start + hand_block_len + p]) for p in range(N)
        ]
        return partner_hands, missing_card

    def _decode_board(self, v: np.ndarray) -> Tuple[List[int], int, int, int]:
        start, _ = self.offsets.board
        N, H, C, R = (self.cfg.players, self.cfg.hand_size, self.cfg.colors, self.cfg.ranks)

        off = start
        deck_bits = v[off : off + self._max_deck_size]
        # deck size is a thermometer encoding: n bits set means n cards remain
        deck_size = int(deck_bits.sum())
        off += self._max_deck_size

        fireworks: List[int] = []
        for _ in range(C):
            color_bits = v[off : off + R]
            idx = np.flatnonzero(color_bits)
            if idx.size == 0:
                fireworks.append(0)
            elif idx.size == 1:
                # bit position tells us which rank was last played, +1 converts to count
                fireworks.append(int(idx[0]) + 1)
            else:
                raise ValueError("fireworks encoding has more than one bit set per color")
            off += R

        # info and life tokens are also thermometer encoded
        info_tokens = int(v[off : off + self.cfg.max_information_tokens].sum())
        off += self.cfg.max_information_tokens
        life_tokens = int(v[off : off + self.cfg.max_life_tokens].sum())
        return fireworks, info_tokens, life_tokens, deck_size

    def _decode_discards(self, v: np.ndarray) -> np.ndarray:
        start, _ = self.offsets.discards
        C, R = self.cfg.colors, self.cfg.ranks
        discards = np.zeros((C, R), dtype=int)
        off = start
        for c in range(C):
            for r in range(R):
                copies = self._copies[r]
                # each card type gets a thermometer of length equal to its copy count
                discards[c, r] = int(v[off : off + copies].sum())
                off += copies
        return discards

    def _decode_last_action(self, v: np.ndarray) -> Optional[LastAction]:
        start, length = self.offsets.last_action
        N, H, C, R = (self.cfg.players, self.cfg.hand_size, self.cfg.colors, self.cfg.ranks)
        section = v[start : start + length]
        # if the whole section is zeros, no action has happened yet (first turn)
        if not section.any():
            return None

        off = 0
        actor_bits = section[off : off + N]; off += N
        move_bits = section[off : off + 4]; off += 4     # 4 move types: play, discard, color, rank
        target_bits = section[off : off + N]; off += N
        color_bits = section[off : off + C]; off += C
        rank_bits = section[off : off + R]; off += R
        outcome_bits = section[off : off + H]; off += H   # which slots the hint matched
        position_bits = section[off : off + H]; off += H  # which slot was played or discarded
        card_bits = section[off : off + C * R]; off += C * R
        play_flags = section[off : off + 2]               # [play_successful, info_token_added]

        actor_offset = int(np.flatnonzero(actor_bits)[0])
        move_idx = int(np.flatnonzero(move_bits)[0])

        if move_idx == 0:
            pos = int(np.flatnonzero(position_bits)[0])
            card_idx = int(np.flatnonzero(card_bits)[0])
            color, rank = index_to_card(card_idx, R)
            return LastAction(
                actor_offset=actor_offset,
                type=ActionType.PLAY,
                slot=pos,
                card=Card(color, rank),
                play_successful=bool(play_flags[0]),
                info_token_added=bool(play_flags[1]),
            )
        if move_idx == 1:
            pos = int(np.flatnonzero(position_bits)[0])
            card_idx = int(np.flatnonzero(card_bits)[0])
            color, rank = index_to_card(card_idx, R)
            return LastAction(
                actor_offset=actor_offset,
                type=ActionType.DISCARD,
                slot=pos,
                card=Card(color, rank),
            )
        if move_idx == 2:
            target = int(np.flatnonzero(target_bits)[0])
            color = int(np.flatnonzero(color_bits)[0])
            slots = [int(s) for s in np.flatnonzero(outcome_bits)]
            return LastAction(
                actor_offset=actor_offset,
                type=ActionType.REVEAL_COLOR,
                target_offset=target,
                color=color,
                revealed_slots=slots,
            )
        if move_idx == 3:
            target = int(np.flatnonzero(target_bits)[0])
            rank = int(np.flatnonzero(rank_bits)[0])
            slots = [int(s) for s in np.flatnonzero(outcome_bits)]
            return LastAction(
                actor_offset=actor_offset,
                type=ActionType.REVEAL_RANK,
                target_offset=target,
                rank=rank,
                revealed_slots=slots,
            )
        raise ValueError(f"unknown move type index {move_idx}")


class KnowledgeDecoder:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.offsets = compute_section_offsets(cfg)

    def decode(self, observation_vec: Sequence[float]) -> KnowledgeView:
        v = np.asarray(observation_vec)
        start, length = self.offsets.card_knowledge
        N, H, C, R = (self.cfg.players, self.cfg.hand_size, self.cfg.colors, self.cfg.ranks)
        # each slot uses C*R bits for the possibility mask plus C and R bits for revealed color and rank
        slot_bits = C * R + C + R
        block = v[start : start + length].reshape(N, H, slot_bits)

        per_player: List[List[CardKnowledge]] = []
        for p in range(N):
            slots: List[CardKnowledge] = []
            for s in range(H):
                possibility = block[p, s, : C * R].reshape(C, R).astype(bool)
                rc_bits = block[p, s, C * R : C * R + C]      # revealed color indicator
                rr_bits = block[p, s, C * R + C : C * R + C + R]  # revealed rank indicator
                rc = int(np.flatnonzero(rc_bits)[0]) if rc_bits.any() else None
                rr = int(np.flatnonzero(rr_bits)[0]) if rr_bits.any() else None
                slots.append(CardKnowledge(possibility=possibility,
                                           revealed_color=rc,
                                           revealed_rank=rr))
            per_player.append(slots)
        return KnowledgeView(per_player=per_player)


@dataclass
class CheckReport:
    episodes: int = 0
    steps: int = 0
    action_decodes: int = 0
    observation_decodes: int = 0
    knowledge_decodes: int = 0
    failures: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        status = "OK" if self.ok else f"FAILED ({len(self.failures)})"
        lines = [
            f"[Phase 0 check] {status}",
            f"  episodes              : {self.episodes}",
            f"  steps                 : {self.steps}",
            f"  action decodes        : {self.action_decodes}",
            f"  observation decodes   : {self.observation_decodes}",
            f"  knowledge decodes     : {self.knowledge_decodes}",
        ]
        for msg in self.failures[:20]:
            lines.append(f"  ! {msg}")
        if len(self.failures) > 20:
            lines.append(f"  ... and {len(self.failures) - 20} more")
        return "\n".join(lines)


class ConsistencyChecker:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.codec = ActionCodec(cfg)
        self.obs_decoder = ObservationDecoder(cfg)
        self.kn_decoder = KnowledgeDecoder(cfg)
        self._total_deck = total_copies_per_color(cfg.ranks) * cfg.colors

    def check_action_codec(self, report: CheckReport) -> None:
        # verifies that encoding then decoding every action id gives back the same id
        for aid in range(self.codec.total_actions):
            action = self.codec.decode(aid)
            round_trip = self.codec.encode(action)
            report.action_decodes += 1
            if round_trip != aid:
                report.failures.append(
                    f"action codec round-trip failed for id {aid} "
                    f"(got {round_trip} via {action})"
                )

    def check_step(
        self,
        observer: str,
        obs_vec: np.ndarray,
        action_mask: np.ndarray,
        chosen_action: Optional[int],
        report: CheckReport,
    ) -> None:
        prefix = f"[{observer}] "
        try:
            decoded = self.obs_decoder.decode(obs_vec, observer)
        except Exception as exc:
            report.failures.append(prefix + f"obs decode raised: {exc!r}")
            return
        report.observation_decodes += 1

        try:
            knowledge = self.kn_decoder.decode(obs_vec)
        except Exception as exc:
            report.failures.append(prefix + f"knowledge decode raised: {exc!r}")
            return
        report.knowledge_decodes += 1

        if not (0 <= decoded.information_tokens <= self.cfg.max_information_tokens):
            report.failures.append(prefix + f"info tokens out of range: {decoded.information_tokens}")
        if not (0 <= decoded.life_tokens <= self.cfg.max_life_tokens):
            report.failures.append(prefix + f"life tokens out of range: {decoded.life_tokens}")
        max_deck = self._total_deck - self.cfg.players * self.cfg.hand_size
        if not (0 <= decoded.deck_size <= max_deck):
            report.failures.append(prefix + f"deck size out of range: {decoded.deck_size}")

        for c, f in enumerate(decoded.fireworks):
            if not (0 <= f <= self.cfg.ranks):
                report.failures.append(
                    prefix + f"fireworks[{COLOR_NAMES[c]}]={f} out of range"
                )

        for c in range(self.cfg.colors):
            for r in range(self.cfg.ranks):
                limit = copies_per_rank(r, self.cfg.ranks)
                n = int(decoded.discards[c, r])
                if not (0 <= n <= limit):
                    report.failures.append(
                        prefix + f"discards[{COLOR_NAMES[c]}{r + 1}]={n} exceeds {limit}"
                    )

        # card conservation: every card in the deck must be accounted for
        fireworks_count = sum(decoded.fireworks)
        discards_count = int(decoded.discards.sum())
        visible_cards = sum(
            1
            for hand in decoded.partner_hands
            for card in hand
            if card is not None
        )
        # the observer's own hand size is known even though card identities are not
        own_hand_size = self.cfg.hand_size - (1 if decoded.missing_card[0] else 0)
        accounted = (
            decoded.deck_size
            + fireworks_count
            + discards_count
            + visible_cards
            + own_hand_size
        )
        if accounted != self._total_deck:
            report.failures.append(
                prefix
                + f"card count mismatch: deck={decoded.deck_size} "
                f"fire={fireworks_count} disc={discards_count} "
                f"visible={visible_cards} own={own_hand_size} "
                f"sum={accounted} expected={self._total_deck}"
            )

        for p, hand in enumerate(decoded.partner_hands, start=1):
            empty = any(card is None for card in hand)
            if decoded.missing_card[p] != empty:
                report.failures.append(
                    prefix
                    + f"partner +{p} missing-card bit={decoded.missing_card[p]} "
                    f"but hand has empty slot: {empty}"
                )

        if chosen_action is not None:
            if not (0 <= chosen_action < self.codec.total_actions):
                report.failures.append(prefix + f"action {chosen_action} out of range")
            elif int(action_mask[chosen_action]) != 1:
                report.failures.append(prefix + f"action {chosen_action} is not legal in mask")

        legal = np.flatnonzero(action_mask).tolist()
        for a in legal:
            try:
                self.codec.decode(int(a))
            except Exception as exc:
                report.failures.append(prefix + f"legal action {a} failed to decode: {exc!r}")
                break

        for p, slots in enumerate(knowledge.per_player):
            for s, slot in enumerate(slots):
                if not slot.possibility.any():
                    report.failures.append(
                        prefix + f"card knowledge empty for player+{p} slot {s}"
                    )
                if slot.revealed_color is not None:
                    # after a color hint, no other color row should still be possible
                    other_rows = np.delete(slot.possibility, slot.revealed_color, axis=0)
                    if other_rows.any():
                        report.failures.append(
                            prefix
                            + f"player+{p} slot {s} revealed color "
                            f"{COLOR_NAMES[slot.revealed_color]} but "
                            "possibility mask includes other colors"
                        )
                if slot.revealed_rank is not None:
                    # after a rank hint, no other rank column should still be possible
                    other_cols = np.delete(slot.possibility, slot.revealed_rank, axis=1)
                    if other_cols.any():
                        report.failures.append(
                            prefix
                            + f"player+{p} slot {s} revealed rank "
                            f"{slot.revealed_rank + 1} but possibility mask "
                            "includes other ranks"
                        )

    def run(self, runner, num_episodes: int = 5, seed: int = 0) -> CheckReport:
        import random

        report = CheckReport()
        self.check_action_codec(report)

        for ep in range(num_episodes):
            env = runner.make_env()
            env.reset(seed=seed + ep)
            report.episodes += 1

            for agent in env.agent_iter():
                obs, reward, termination, truncation, info = env.last()
                if termination or truncation:
                    env.step(None)
                    continue

                obs_vec = np.asarray(obs["observation"])
                mask = np.asarray(obs["action_mask"])
                legal = np.flatnonzero(mask).tolist()
                if not legal:
                    report.failures.append(f"[{agent}] empty action mask mid-game")
                    env.step(None)
                    continue

                # use a fresh seeded rng per step so checks are reproducible
                chosen = int(random.Random(seed + ep + report.steps).choice(legal))
                self.check_step(agent, obs_vec, mask, chosen, report)
                report.steps += 1
                env.step(chosen)

            env.close()

        return report


def run_phase0_selfcheck(cfg, runner, num_episodes: int = 5, seed: int = 0) -> CheckReport:
    checker = ConsistencyChecker(cfg)
    return checker.run(runner, num_episodes=num_episodes, seed=seed)


__all__ = [
    "COLOR_NAMES",
    "copies_per_rank",
    "total_copies_per_color",
    "card_index",
    "index_to_card",
    "format_card",
    "agent_name_to_index",
    "ActionType",
    "StructuredAction",
    "ActionCodec",
    "SectionOffsets",
    "compute_section_offsets",
    "Card",
    "LastAction",
    "DecodedObservation",
    "CardKnowledge",
    "KnowledgeView",
    "ObservationDecoder",
    "KnowledgeDecoder",
    "CheckReport",
    "ConsistencyChecker",
    "run_phase0_selfcheck",
]
