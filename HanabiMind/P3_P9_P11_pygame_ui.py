"""
Pygame UI for interactive Hanabi play and explainability.

Phase 3: minimal board from the human's POV (keyboard to act on your turn).

Phase 11 (``ui_verbose``): larger window, right-hand panel for ML class
probabilities, human playstyle bars, and short AI action explanations. Heavy
inference and text generation run in the game loop, not in ``render``.

Layout: a single ``_compute_layout`` pass builds rects for each section
(top bar, partners, discards, hand, bottom panel, right panel). All draw
methods read from those rects so sections can never overlap. Card size is
auto-scaled to fit the available vertical budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from P0_observation_decoding import (
    COLOR_NAMES,
    ActionCodec,
    ActionType,
    Card,
    DecodedObservation,
    KnowledgeView,
    StructuredAction,
    agent_name_to_index,
)
from P1_game_state_simulation import build_state_from_observation, pool_remaining
from P6_rollout_policy import _dead_matrix, _playable_matrix


# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

CARD_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (220,  70,  70),   # Red
    (220, 200,  50),   # Yellow
    ( 70, 180,  90),   # Green
    (235, 235, 235),   # White
    ( 70, 120, 220),   # Blue
)

BG_COLOR        = ( 26,  28,  40)
PANEL_COLOR     = ( 48,  50,  68)
CARD_BACK_COLOR = ( 90,  90, 115)
TEXT_LIGHT      = (235, 235, 240)
TEXT_DIM        = (165, 165, 175)
TEXT_DARK       = ( 15,  15,  20)
HIGHLIGHT       = (240, 200,  80)
DANGER          = (230,  85,  85)
HINT_GLOW       = (120, 200, 255)
PLAY_RING       = ( 90, 255, 140)
DISC_RING       = (255, 160,  90)
ACCENT1         = (130, 180, 255)
ML_PANEL        = ( 38,  40,  56)
SECTION_BG      = ( 36,  38,  54)
SECTION_BORDER  = ( 70,  74, 100)
HAND_BORDER     = (180, 150,  60)
DEFAULT_WINDOW_BASIC = (1200, 780)
DEFAULT_WINDOW_VERBOSE = (1320, 860)
PANEL_W_FRAC     = 0.19
PANEL_W_MIN      = 220
PANEL_W_MAX      = 300
FULL_COLOR_NAME  = ("Red", "Yellow", "Green", "White", "Blue")


# ---------------------------------------------------------------------------
# Phase 11 — view model
# ---------------------------------------------------------------------------

@dataclass
class Phase11Highlights:
    active_agent: str = ""
    own_slot: Optional[int] = None
    hint_offset: Optional[int] = None
    hint_is_color: bool = False
    hint_c: Optional[int] = None
    hint_r: Optional[int] = None


@dataclass
class Phase11ViewModel:
    show_panel: bool = False
    model_name: str = "—"
    p_random: float = 0.0
    p_heuristic: float = 0.0
    p_maximax: float = 0.0
    chaotic: float = 0.0
    cooperative: float = 0.0
    strategic: float = 0.0
    playstyle_history: List[Tuple[float, float, float]] = field(default_factory=list)
    has_playstyle: bool = False
    last_ai_name: str = ""
    last_action_kind: str = ""
    last_explanation: str = ""
    highlight: Phase11Highlights = field(default_factory=Phase11Highlights)


def short_action_kind(action: StructuredAction) -> str:
    if action.type in (ActionType.DISCARD, ActionType.PLAY):
        if action.type is ActionType.PLAY:
            return "PLAY"
        return "DISCARD"
    return "HINT"


def _p_playable_marginal(knowledge, base_state, observer: int, slot: int) -> float:
    own = knowledge.per_player[0]
    if slot < 0 or slot >= len(own):
        return 0.0
    pmask = own[slot].possibility
    pool = pool_remaining(base_state, observer).astype(np.float64)
    w = pmask.astype(np.float64) * pool
    tot = float(w.sum())
    if tot <= 0:
        return 0.0
    pl = _playable_matrix(base_state)
    return float((w * pl).sum() / tot)


def _p_dead_marginal(knowledge, base_state, observer: int, slot: int) -> float:
    own = knowledge.per_player[0]
    if slot < 0 or slot >= len(own):
        return 0.0
    pmask = own[slot].possibility
    pool = pool_remaining(base_state, observer).astype(np.float64)
    w = pmask.astype(np.float64) * pool
    tot = float(w.sum())
    if tot <= 0:
        return 0.0
    dead = _dead_matrix(base_state)
    return float((w * dead).sum() / tot)


def explain_ai_action(
    action: StructuredAction,
    agent_name: str,
    cfg,
    knowledge: KnowledgeView,
    decoded: DecodedObservation,
) -> str:
    observer = agent_name_to_index(agent_name)
    try:
        base = build_state_from_observation(decoded, cfg, observer)
    except Exception:
        return "Could not rate this action from the current view"

    if action.type is ActionType.PLAY:
        p = _p_playable_marginal(knowledge, base, observer, int(action.slot or 0))
        if p >= 0.78:
            return "This play is likely a good one"
        if p >= 0.45:
            return "Reasonable play — some uncertainty"
        if p >= 0.2:
            return "Risky play — card may not match"
        return "Speculative play"

    if action.type is ActionType.DISCARD:
        p_dead = _p_dead_marginal(knowledge, base, observer, int(action.slot or 0))
        if p_dead >= 0.65:
            return "Safe discard — card was probably useless"
        if p_dead >= 0.35:
            return "Cautious discard"
        return "Discarding a card that still might help"

    if action.type in (ActionType.REVEAL_COLOR, ActionType.REVEAL_RANK):
        return "A helpful hint for the team"

    return ""


def phase11_from_ai_step(
    agent_name: str,
    struct: StructuredAction,
    cfg,
    knowledge: KnowledgeView,
    decoded: DecodedObservation,
) -> Tuple[str, str, Phase11Highlights]:
    kind = short_action_kind(struct)
    expl = explain_ai_action(struct, agent_name, cfg, knowledge, decoded)
    hi = Phase11Highlights(active_agent=agent_name)
    if struct.type is ActionType.PLAY or struct.type is ActionType.DISCARD:
        hi.own_slot = int(struct.slot or 0)
    elif struct.type is ActionType.REVEAL_COLOR:
        hi.hint_offset = int(struct.target_offset or 0)
        hi.hint_is_color = True
        hi.hint_c = int(struct.color or 0)
    elif struct.type is ActionType.REVEAL_RANK:
        hi.hint_offset = int(struct.target_offset or 0)
        hi.hint_is_color = False
        hi.hint_r = int(struct.rank or 0)
    return kind, expl, hi


class UIClosed(Exception):
    """Raised when the user closes the pygame window or presses Q."""


# ---------------------------------------------------------------------------
# HanabiUI
# ---------------------------------------------------------------------------

class HanabiUI:
    """Pygame-based renderer + keyboard input loop, from one fixed observer's
    perspective (the human seat).

    All drawing is driven by ``_compute_layout`` which produces a set of
    pygame.Rect regions stacked vertically. Sections never overlap by
    construction, and card size is auto-scaled if the window is too small.
    """

    # Class defaults; overridden per-frame by _compute_layout.
    CARD_W   = 58
    CARD_H   = 84
    CARD_GAP = 10
    HAND_GAP = 10
    _STATUS_H_MIN = 70

    def __init__(
        self,
        cfg,
        human_agent: str,
        window_size: Optional[Tuple[int, int]] = None,
        ui_verbose: bool = False,
    ) -> None:
        import pygame
        self.pygame = pygame
        if window_size is None:
            window_size = (
                DEFAULT_WINDOW_VERBOSE if ui_verbose else DEFAULT_WINDOW_BASIC
            )
        pygame.init()
        cap = "Hanabi — you + 4 AI"
        if ui_verbose:
            cap += "  (ML + playstyle panel)"
        pygame.display.set_caption(cap)
        self.screen = pygame.display.set_mode(window_size)
        self.clock = pygame.time.Clock()

        self.font        = pygame.font.SysFont("arial", 15)
        self.font_big    = pygame.font.SysFont("arial", 20, bold=True)
        self.font_huge   = pygame.font.SysFont("arial", 28, bold=True)
        self.font_small  = pygame.font.SysFont("arial", 12)
        self.font_tiny   = pygame.font.SysFont("arial", 11)
        self.font_title  = pygame.font.SysFont("arial", 17, bold=True)
        self.font_status_head   = pygame.font.SysFont("arial", 17, bold=True)
        self.font_status_line   = pygame.font.SysFont("arial", 15)
        self.font_status_key    = pygame.font.SysFont("arial", 15)
        self.font_status_bright = pygame.font.SysFont("arial", 13)
        self.font_section       = pygame.font.SysFont("arial", 16, bold=True)

        self.cfg          = cfg
        self.human_agent  = human_agent
        self.human_index  = agent_name_to_index(human_agent)
        self.codec        = ActionCodec(cfg)
        self.window_w, self.window_h = window_size
        self.ui_verbose   = bool(ui_verbose)

        self._phase11: Optional[Phase11ViewModel] = None
        self._current_status_h: int = self._STATUS_H_MIN

        # Layout rectangles, populated by _compute_layout each frame.
        self._lay_top = None
        self._lay_partners = None
        self._lay_discard = None
        self._lay_hand = None
        self._lay_bottom = None
        self._lay_right = None
        self._lay_partner_row_h = 0
        self._lay_partner_row_gap = 8
        self._cell_w_calc = 30
        self._cell_h_calc = 22

    def set_phase11(self, view: Optional[Phase11ViewModel]) -> None:
        self._phase11 = view

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _pad(self) -> int:
        return max(10, int(self.window_w * 0.012))

    def _section_gap(self) -> int:
        return max(10, int(self.window_h * 0.018))

    def _right_panel_w(self) -> int:
        if not self.ui_verbose:
            return 0
        w = int(self.window_w * PANEL_W_FRAC)
        return max(PANEL_W_MIN, min(PANEL_W_MAX, w))

    def _board_right(self) -> int:
        m = self._pad()
        if self.ui_verbose:
            return self.window_w - self._right_panel_w() - m
        return self.window_w - m

    def _status_h_for_lines(self, n_lines: int) -> int:
        n = max(1, min(8, n_lines))
        line = 19
        return max(self._STATUS_H_MIN, 14 + n * line)

    # Legacy helpers — kept for any external callers but unused by the
    # new layout system.
    def _content_left(self) -> int:
        return self._pad()

    def _partner_row_width(self) -> int:
        return self.cfg.hand_size * (self.CARD_W + self.CARD_GAP) - self.CARD_GAP

    def _discard_cell_w(self) -> int:
        return self._cell_w_calc

    def _discard_grid_width(self) -> int:
        return self.cfg.ranks * self._cell_w_calc + 8

    def _cluster_gap(self) -> int:
        return 18

    def _cluster_pair_x0(self) -> int:
        if self._lay_partners is not None:
            row_w = self._partner_row_width()
            return self._lay_partners.x + (self._lay_partners.w - row_w) // 2
        return self._pad()

    # ------------------------------------------------------------------
    # Layout computation: builds all section rects up-front.
    # ------------------------------------------------------------------
    def _compute_layout(self) -> None:
        pygame = self.pygame
        pad = self._pad()
        sg = self._section_gap()

        # --- right column (independent) ---
        rp_w = self._right_panel_w()
        if rp_w > 0:
            right_x = self.window_w - rp_w
            board_right = right_x - pad
        else:
            right_x = self.window_w
            board_right = self.window_w - pad

        board_left = pad
        board_w = max(240, board_right - board_left)

        # --- bottom status panel (anchored to bottom) ---
        bottom_h = self._current_status_h
        bottom_y = self.window_h - bottom_h - pad
        bottom_rect = pygame.Rect(
            pad, bottom_y, self.window_w - 2 * pad, bottom_h,
        )

        # --- top bar (fixed-ish, scales with window) ---
        top_h = max(72, min(96, int(self.window_h * 0.105)))
        top_rect = pygame.Rect(board_left, pad, board_w, top_h)

        # --- middle area available for partners + discard + hand ---
        middle_top = top_rect.bottom + sg
        middle_bot = bottom_rect.top - sg
        middle_h = max(120, middle_bot - middle_top)

        n_partners = max(1, self.cfg.players - 1)
        n_colors = self.cfg.colors

        # Constants used to compose section heights.
        cap_h = 18         # partner row caption
        cap_gap = 4
        slot_label_h = 14  # "[N]" under partner card
        row_gap = 8        # between partner rows
        disc_title_h = 22
        disc_header_h = 14
        hand_title_h = 22
        hand_card_gap = 4
        hand_below_h = 32  # 2 short lines under your hand cards

        def heights_for(ch: int) -> Tuple[int, int, int, int, int]:
            partner_row = cap_h + cap_gap + ch + slot_label_h
            partner_sec = (
                n_partners * partner_row + max(0, n_partners - 1) * row_gap
            )
            cell_h = max(16, int(ch * 0.30))
            disc_sec = (
                disc_title_h + 4 + disc_header_h + n_colors * cell_h + 6
            )
            hand_sec = hand_title_h + hand_card_gap + ch + hand_below_h
            total = partner_sec + sg + disc_sec + sg + hand_sec
            return partner_row, partner_sec, disc_sec, hand_sec, total

        # Auto-scale: pick the largest card_h whose layout fits.
        card_h = 36
        for ch in (84, 78, 72, 66, 60, 54, 48, 44, 40, 36):
            _, _, _, _, total = heights_for(ch)
            if total <= middle_h:
                card_h = ch
                break

        card_w = max(34, int(card_h * 0.72))
        cell_h = max(16, int(card_h * 0.30))
        cell_w = max(28, min(42, int(card_w * 0.62) + 6))

        self.CARD_H = card_h
        self.CARD_W = card_w
        self._cell_w_calc = cell_w
        self._cell_h_calc = cell_h

        partner_row_h, partner_sec_h, disc_sec_h, hand_sec_h, _ = heights_for(card_h)

        # Width sanity: ensure partner row of cards fits in board_w.
        row_card_w = self.cfg.hand_size * (card_w + self.CARD_GAP) - self.CARD_GAP
        if row_card_w + 16 > board_w:
            shrink = (row_card_w + 16) - board_w
            new_card_w = max(28, card_w - (shrink // self.cfg.hand_size + 1))
            self.CARD_W = new_card_w
            card_w = new_card_w

        # Allocate vertical regions inside middle area.
        y = middle_top
        partners_rect = pygame.Rect(board_left, y, board_w, partner_sec_h)
        y += partner_sec_h + sg
        discard_rect = pygame.Rect(board_left, y, board_w, disc_sec_h)
        y += disc_sec_h + sg
        hand_rect = pygame.Rect(board_left, y, board_w, hand_sec_h)

        # Final clamp: if hand still spills over (extreme small windows),
        # squeeze the section gaps and re-stack.
        if hand_rect.bottom > middle_bot:
            overflow = hand_rect.bottom - middle_bot
            tight_sg = max(4, sg - overflow // 2 - 1)
            y = middle_top
            partners_rect = pygame.Rect(board_left, y, board_w, partner_sec_h)
            y += partner_sec_h + tight_sg
            discard_rect = pygame.Rect(board_left, y, board_w, disc_sec_h)
            y += disc_sec_h + tight_sg
            hand_rect = pygame.Rect(board_left, y, board_w, hand_sec_h)

        # --- right panel (full vertical column, ends above status) ---
        if rp_w > 0:
            rp_top = pad
            rp_bot = bottom_rect.top - sg
            rp_rect = pygame.Rect(
                right_x + 4, rp_top, rp_w - 8, max(80, rp_bot - rp_top),
            )
        else:
            rp_rect = None

        self._lay_top = top_rect
        self._lay_partners = partners_rect
        self._lay_discard = discard_rect
        self._lay_hand = hand_rect
        self._lay_bottom = bottom_rect
        self._lay_right = rp_rect
        self._lay_partner_row_h = partner_row_h
        self._lay_partner_row_gap = row_gap

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        try:
            self.pygame.quit()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _text(self, surf, text, pos, font=None, color=TEXT_LIGHT):
        f = font or self.font
        surf.blit(f.render(text, True, color), pos)

    def _truncate(self, text: str, font, max_w: int) -> str:
        if font.size(text)[0] <= max_w:
            return text
        ell = "…"
        while text and font.size(text + ell)[0] > max_w:
            text = text[:-1]
        return text + ell if text else ""

    def _pump_quit(self) -> None:
        for event in self.pygame.event.get(self.pygame.QUIT):
            raise UIClosed()

    def _player_display_name(self, agent: str) -> str:
        if agent.lower().startswith("player_"):
            return "Player " + agent.split("_", 1)[-1]
        return agent

    def _partner_row_caption(
        self, absolute_idx: int, ring_offset: int, is_them: bool,
    ) -> str:
        n = int(ring_offset) + 1
        if n == 1:
            dist = "next at the table"
        else:
            dist = f"{n} seats away (clockwise)"
        cap = f"Player {absolute_idx} ({dist})"
        if is_them:
            cap += "  ·  their turn"
        return cap

    def _format_card_short(self, card: Card) -> str:
        return f"{COLOR_NAMES[card.color]}{card.rank + 1}"

    def _friendly_last_action(self, decoded: DecodedObservation) -> str:
        la = decoded.last_action
        if la is None:
            return "No move yet (game is starting)"
        t = la.type
        if t is ActionType.REVEAL_COLOR and la.color is not None:
            cname = FULL_COLOR_NAME[la.color]
            return f"Gave a color hint ({cname})"
        if t is ActionType.REVEAL_RANK and la.rank is not None:
            return f"Gave a number hint ({la.rank + 1})"
        if t in (ActionType.PLAY, ActionType.DISCARD) and la.card is not None:
            c = self._format_card_short(la.card)
            if t is ActionType.PLAY:
                if la.play_successful is False:
                    return f"Misplayed {c}"
                extra = " — gained a hint token" if la.info_token_added else ""
                return f"Successfully played {c}{extra}"
            return f"Discarded {c}"
        if t is ActionType.DISCARD:
            return f"Discarded a card (slot {1 + (la.slot or 0)})"
        if t is ActionType.PLAY:
            st = "Successful play" if la.play_successful else "Misplay"
            return f"{st} (slot {1 + (la.slot or 0)})"
        return la.describe()

    def wait_ms(self, ms: int) -> None:
        if ms <= 0:
            return
        pygame = self.pygame
        end_tick = pygame.time.get_ticks() + ms
        while pygame.time.get_ticks() < end_tick:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise UIClosed()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    raise UIClosed()
            self.clock.tick(60)

    # ------------------------------------------------------------------
    # Card drawing primitive
    # ------------------------------------------------------------------
    def _draw_card(
        self,
        rect,
        card: Optional[Card],
        *,
        face_down: bool = False,
        revealed_color: Optional[int] = None,
        revealed_rank: Optional[int] = None,
        highlight: bool = False,
        ring: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        pygame = self.pygame
        surf = self.screen

        if face_down:
            pygame.draw.rect(surf, CARD_BACK_COLOR, rect, border_radius=7)
            if revealed_color is not None:
                stripe = pygame.Rect(rect.x + 4, rect.y + 4, rect.w - 8, 8)
                pygame.draw.rect(
                    surf, CARD_COLORS[revealed_color], stripe, border_radius=3,
                )
            q_font = self.font_huge if rect.h >= 60 else self.font_big
            q = q_font.render("?", True, TEXT_LIGHT)
            surf.blit(q, (
                rect.x + rect.w // 2 - q.get_width() // 2,
                rect.y + rect.h // 2 - q.get_height() // 2,
            ))
            if revealed_rank is not None:
                self._text(
                    surf, str(revealed_rank + 1),
                    (rect.x + 5, rect.y + rect.h - 18),
                    self.font_small, HIGHLIGHT,
                )
        else:
            if card is None:
                pygame.draw.rect(
                    surf, (40, 40, 55), rect, border_radius=7, width=2,
                )
                self._text(
                    surf, "—",
                    (rect.x + rect.w // 2 - 4, rect.y + rect.h // 2 - 9),
                    self.font_big, TEXT_DIM,
                )
            else:
                pygame.draw.rect(
                    surf, CARD_COLORS[card.color], rect, border_radius=7,
                )
                label = f"{COLOR_NAMES[card.color]}{card.rank + 1}"
                glyph_font = self.font_big if rect.h >= 50 else self.font
                glyph = glyph_font.render(label, True, TEXT_DARK)
                surf.blit(glyph, (
                    rect.x + rect.w // 2 - glyph.get_width() // 2,
                    rect.y + rect.h // 2 - glyph.get_height() // 2,
                ))

        if ring is not None:
            pygame.draw.rect(surf, ring, rect, width=3, border_radius=7)
        elif highlight:
            pygame.draw.rect(surf, HIGHLIGHT, rect, width=3, border_radius=7)

    def _status_idle_choices(self) -> List[str]:
        return [
            "Your turn — choose an action:",
            "  [P] Play a card",
            "  [D] Discard a card",
            "  [H] Give a hint",
        ]

    # ------------------------------------------------------------------
    # Section: TOP BAR
    # ------------------------------------------------------------------
    def _draw_top_bar(
        self, decoded: DecodedObservation, current_agent: str,
    ) -> None:
        pygame = self.pygame
        rect = self._lay_top

        pygame.draw.rect(self.screen, SECTION_BG, rect, border_radius=8)
        pygame.draw.rect(
            self.screen, SECTION_BORDER, rect, width=1, border_radius=8,
        )

        inner = 12
        x_left = rect.x + inner
        x_right_max = rect.right - inner

        # Vertical layout inside top bar: two lightly spaced rows.
        avail_h = rect.h - 2 * 6
        row_h = avail_h // 2
        y1 = rect.y + 6
        y2 = y1 + row_h

        # ---- Row 1: Fireworks (left) | Stats (right) ----
        self._text(self.screen, "Fireworks", (x_left, y1 + 4), self.font_big)
        fw_label_w = self.font_big.size("Fireworks")[0]
        chip_x0 = x_left + fw_label_w + 14

        # Reserve right side for the stats block.
        stats_block_w = 220
        stats_x = max(chip_x0 + 60, x_right_max - stats_block_w)
        chip_zone_w = max(60, stats_x - chip_x0 - 12)
        chip_w = max(
            32,
            min(54, (chip_zone_w - 4 * (self.cfg.colors - 1)) // self.cfg.colors),
        )
        chip_gap = 4
        chip_h = min(28, row_h - 6)

        chip_y = y1 + (row_h - chip_h) // 2
        for c in range(self.cfg.colors):
            cx = chip_x0 + c * (chip_w + chip_gap)
            chip = pygame.Rect(cx, chip_y, chip_w, chip_h)
            pygame.draw.rect(self.screen, CARD_COLORS[c], chip, border_radius=5)
            txt = f"{decoded.fireworks[c]}/{self.cfg.ranks}"
            glyph = self.font.render(txt, True, TEXT_DARK)
            self.screen.blit(glyph, (
                chip.x + chip.w // 2 - glyph.get_width() // 2,
                chip.y + chip.h // 2 - glyph.get_height() // 2,
            ))

        # Stats: three values, vertically stacked but compact.
        stat_y = y1 + 4
        s_clues = (
            f"Clues  {decoded.information_tokens}/"
            f"{self.cfg.max_information_tokens}"
        )
        s_lives = (
            f"Lives  {decoded.life_tokens}/{self.cfg.max_life_tokens}"
        )
        s_deck = f"Deck   {decoded.deck_size}"
        life_color = TEXT_LIGHT if decoded.life_tokens > 1 else DANGER

        # Try a single line first to save vertical space.
        single = f"{s_clues}    {s_deck}"
        if self.font.size(single)[0] <= stats_block_w:
            self._text(self.screen, single, (stats_x, stat_y))
            self._text(
                self.screen, s_lives, (stats_x, stat_y + 18),
                color=life_color,
            )
        else:
            self._text(self.screen, s_clues, (stats_x, stat_y))
            self._text(
                self.screen, s_lives, (stats_x, stat_y + 16),
                color=life_color,
            )
            self._text(self.screen, s_deck, (stats_x, stat_y + 32))

        # ---- Row 2: Turn (left, prominent) | Last action (right, dim) ----
        turn_line = f"Turn: {self._player_display_name(current_agent)}"
        if current_agent == self.human_agent:
            turn_line += "  (you!)"
        color = HIGHLIGHT if current_agent == self.human_agent else TEXT_LIGHT
        self._text(
            self.screen, turn_line, (x_left, y2 + 2), self.font_big, color,
        )
        turn_w = self.font_big.size(turn_line)[0]

        last_txt = "Last: " + self._friendly_last_action(decoded)
        max_last_w = max(80, x_right_max - (x_left + turn_w + 24))
        last_txt = self._truncate(last_txt, self.font_small, max_last_w)
        last_w = self.font_small.size(last_txt)[0]
        last_x = x_right_max - last_w
        self._text(
            self.screen, last_txt, (last_x, y2 + 6),
            self.font_small, TEXT_DIM,
        )

    # ------------------------------------------------------------------
    # Section: PARTNER HANDS
    # ------------------------------------------------------------------
    def _draw_partner_hands(
        self, decoded: DecodedObservation, current_agent: str,
    ) -> None:
        pygame = self.pygame
        rect = self._lay_partners
        p11 = self._phase11 if self.ui_verbose else None
        hi = p11.highlight if (p11 and p11.show_panel) else None
        last_kind = (p11.last_action_kind or "") if p11 else ""

        hint_tgt_abs: Optional[int] = None
        if hi is not None and hi.hint_offset is not None and hi.active_agent:
            act_i = agent_name_to_index(hi.active_agent)
            hint_tgt_abs = (act_i + int(hi.hint_offset)) % self.cfg.players

        cap_h = 18
        cap_gap = 4
        slot_label_h = 14
        row_h = self._lay_partner_row_h
        row_gap = self._lay_partner_row_gap

        row_card_w = (
            self.cfg.hand_size * (self.CARD_W + self.CARD_GAP) - self.CARD_GAP
        )
        cards_x0 = rect.x + (rect.w - row_card_w) // 2

        for p, hand in enumerate(decoded.partner_hands):
            absolute_idx = (self.human_index + 1 + p) % self.cfg.players
            name = f"player_{absolute_idx}"
            row_y = rect.y + p * (row_h + row_gap)

            if row_y + row_h > rect.bottom:
                break  # safety: never draw outside section

            is_current = (name == current_agent)
            tag_color = HIGHLIGHT if is_current else TEXT_DIM
            cap = self._partner_row_caption(absolute_idx, p, is_current)
            cap = self._truncate(cap, self.font, rect.w - 16)
            self._text(self.screen, cap, (cards_x0, row_y), self.font, tag_color)

            cards_y = row_y + cap_h + cap_gap
            row_bg = (
                hint_tgt_abs is not None and absolute_idx == hint_tgt_abs
            )
            if row_bg:
                band = pygame.Rect(
                    cards_x0 - 8, cards_y - 3,
                    row_card_w + 16, self.CARD_H + 6,
                )
                pygame.draw.rect(self.screen, (52, 58, 84), band, border_radius=8)

            for s in range(self.cfg.hand_size):
                card = hand[s] if s < len(hand) else None
                x = cards_x0 + s * (self.CARD_W + self.CARD_GAP)
                crect = pygame.Rect(x, cards_y, self.CARD_W, self.CARD_H)
                ring = None
                if hi and name == hi.active_agent and hi.own_slot is not None:
                    if s == hi.own_slot:
                        ring = PLAY_RING if last_kind == "PLAY" else DISC_RING
                if (
                    hi and row_bg and card is not None
                    and hi.hint_offset is not None
                ):
                    if hi.hint_is_color and hi.hint_c is not None:
                        if card.color == hi.hint_c:
                            ring = HINT_GLOW
                    elif not hi.hint_is_color and hi.hint_r is not None:
                        if card.rank == hi.hint_r:
                            ring = HINT_GLOW

                self._draw_card(crect, card, face_down=False, ring=ring)
                slot_y = crect.y + crect.h + 1
                if slot_y + slot_label_h <= rect.bottom:
                    glyph = self.font_small.render(
                        f"[{s + 1}]", True, TEXT_DIM,
                    )
                    self.screen.blit(glyph, (
                        crect.x + crect.w // 2 - glyph.get_width() // 2,
                        slot_y,
                    ))

    # ------------------------------------------------------------------
    # Section: DISCARDS
    # ------------------------------------------------------------------
    def _draw_discards(self, discards: np.ndarray) -> None:
        pygame = self.pygame
        rect = self._lay_discard
        cell_w = self._cell_w_calc
        cell_h = self._cell_h_calc

        # Title.
        self._text(
            self.screen, "Discards",
            (rect.x + 6, rect.y), self.font_big, TEXT_LIGHT,
        )
        sub = "(by color × number)"
        title_w = self.font_big.size("Discards")[0]
        self._text(
            self.screen, sub,
            (rect.x + 6 + title_w + 8, rect.y + 5),
            self.font_small, TEXT_DIM,
        )

        # Center the grid (label column + ranks columns).
        label_col_w = 30
        grid_w = self.cfg.ranks * cell_w
        total_w = label_col_w + grid_w
        grid_x = rect.x + (rect.w - total_w) // 2 + label_col_w
        header_y = rect.y + 22 + 4
        grid_y = header_y + 14

        # Rank header.
        for r in range(self.cfg.ranks):
            cx = grid_x + r * cell_w + cell_w // 2
            glyph = self.font_small.render(str(r + 1), True, TEXT_DIM)
            self.screen.blit(
                glyph, (cx - glyph.get_width() // 2, header_y),
            )

        # Color rows.
        for c in range(self.cfg.colors):
            cy = grid_y + c * cell_h
            if cy + cell_h > rect.bottom:
                break  # safety
            cname = COLOR_NAMES[c]
            label_glyph = self.font_small.render(cname, True, CARD_COLORS[c])
            self.screen.blit(label_glyph, (
                grid_x - label_col_w + 2,
                cy + (cell_h - label_glyph.get_height()) // 2,
            ))
            for r in range(self.cfg.ranks):
                cell = pygame.Rect(
                    grid_x + r * cell_w + 1,
                    cy + 1,
                    cell_w - 2, cell_h - 2,
                )
                count = int(discards[c, r])
                if count > 0:
                    pygame.draw.rect(
                        self.screen, CARD_COLORS[c], cell, border_radius=3,
                    )
                    txt_color = TEXT_DARK
                else:
                    pygame.draw.rect(
                        self.screen, (45, 45, 58), cell, border_radius=3,
                    )
                    txt_color = TEXT_DIM
                num_glyph = self.font_small.render(
                    str(count), True, txt_color,
                )
                self.screen.blit(num_glyph, (
                    cell.x + cell.w // 2 - num_glyph.get_width() // 2,
                    cell.y + cell.h // 2 - num_glyph.get_height() // 2,
                ))

    # ------------------------------------------------------------------
    # Section: YOUR HAND
    # ------------------------------------------------------------------
    def _draw_human_hand(self, knowledge: KnowledgeView) -> None:
        pygame = self.pygame
        rect = self._lay_hand

        # Subtle frame around your hand area.
        bg = pygame.Rect(rect.x, rect.y, rect.w, rect.h)
        pygame.draw.rect(self.screen, (40, 42, 60), bg, border_radius=10)
        pygame.draw.rect(
            self.screen, HAND_BORDER, bg, width=1, border_radius=10,
        )

        # Title at top of section.
        title = (
            f"Your hand (hidden) — {self._player_display_name(self.human_agent)}"
        )
        title = self._truncate(title, self.font_big, rect.w - 16)
        self._text(
            self.screen, title, (rect.x + 10, rect.y + 4),
            self.font_big, HIGHLIGHT,
        )

        # Centered card row.
        card_gap = 12
        row_w = self.cfg.hand_size * (self.CARD_W + card_gap) - card_gap
        cards_x = rect.x + (rect.w - row_w) // 2
        cards_y = rect.y + 22 + 4

        # Below-text region: 2 short lines per card.
        below_y = cards_y + self.CARD_H + 2
        line_h = 14
        max_text_w = self.CARD_W + card_gap - 4

        for s in range(self.cfg.hand_size):
            x = cards_x + s * (self.CARD_W + card_gap)
            crect = pygame.Rect(x, cards_y, self.CARD_W, self.CARD_H)
            kn = knowledge.per_player[0][s]
            self._draw_card(
                crect, None, face_down=True,
                revealed_color=kn.revealed_color,
                revealed_rank=kn.revealed_rank,
            )

            slot_label = f"Slot {s + 1}"
            self._text(
                self.screen, slot_label,
                (crect.x + 2, below_y),
                self.font_small, HIGHLIGHT,
            )

            pc = kn.possible_colors()
            pr = kn.possible_ranks()
            pc_s = "".join(COLOR_NAMES[c] for c in pc) or "-"
            pr_s = "".join(str(r + 1) for r in pr) or "-"

            line2 = f"C:{pc_s}  N:{pr_s}"
            line2 = self._truncate(line2, self.font_small, max_text_w)
            line2_y = below_y + line_h
            if line2_y + 12 <= rect.bottom:
                self._text(
                    self.screen, line2,
                    (crect.x + 2, line2_y),
                    self.font_small, TEXT_DIM,
                )

    # ------------------------------------------------------------------
    # Section: BOTTOM STATUS PANEL
    # ------------------------------------------------------------------
    def _draw_status(self, status_lines: List[str]) -> None:
        pygame = self.pygame
        rect = self._lay_bottom

        pygame.draw.rect(self.screen, PANEL_COLOR, rect, border_radius=10)
        pygame.draw.rect(
            self.screen, SECTION_BORDER, rect, width=1, border_radius=10,
        )

        n = min(7, len(status_lines))
        if n == 0:
            return

        inner_pad = 10
        line_h = 19
        avail_h = rect.h - 2 * inner_pad
        if n * line_h > avail_h:
            line_h = max(15, avail_h // n)

        x = rect.x + 16
        y0 = rect.y + inner_pad
        max_w = rect.w - 32

        for i, line in enumerate(status_lines[:n]):
            stripped = line.strip()
            is_head = (i == 0)
            is_back = stripped.startswith("Back:") or "Quit:" in stripped
            is_key = (not is_head) and (
                stripped.startswith("[")
                or stripped.startswith("Shortcuts")
                or stripped.startswith("Number keys")
                or stripped.startswith("Color:")
                or stripped.startswith("C for")
            )
            if is_head:
                fnt, col = self.font_status_head, HIGHLIGHT
            elif is_back:
                fnt, col = self.font_status_bright, TEXT_DIM
            elif is_key:
                fnt, col = self.font_status_key, (220, 225, 250)
            else:
                fnt, col = self.font_status_line, TEXT_LIGHT

            text = self._truncate(line, fnt, max_w)
            self._text(self.screen, text, (x, y0 + i * line_h), fnt, col)

    # ------------------------------------------------------------------
    # Section: RIGHT PANEL (verbose only)
    # ------------------------------------------------------------------
    def _horiz_bars(
        self,
        x: int, y: int, width: int,
        labels_vals: List[Tuple[str, float, Tuple[int, int, int]]],
    ) -> int:
        pygame = self.pygame
        yp = y
        row_h = 22
        for text, val, bcol in labels_vals:
            v = max(0.0, min(1.0, val))
            self._text(
                self.screen, text, (x, yp + 1), self.font_small, TEXT_DIM,
            )
            self._text(
                self.screen, f"{v:.0%}",
                (x + width - 36, yp + 1), self.font_small, TEXT_LIGHT,
            )
            bg = pygame.Rect(x, yp + 14, width, 6)
            pygame.draw.rect(self.screen, (35, 36, 48), bg, border_radius=3)
            fill_w = int(width * v)
            if fill_w > 0:
                fr = pygame.Rect(x, yp + 14, fill_w, 6)
                pygame.draw.rect(self.screen, bcol, fr, border_radius=3)
            yp += row_h
        return yp

    def _draw_playstyle_history(
        self,
        x: int,
        y: int,
        width: int,
        history: List[Tuple[float, float, float]],
    ) -> int:
        pygame = self.pygame
        if not history:
            return y

        h = 76
        rect = pygame.Rect(x, y, width, h)
        pygame.draw.rect(self.screen, (31, 33, 46), rect, border_radius=6)
        pygame.draw.rect(self.screen, SECTION_BORDER, rect, width=1, border_radius=6)

        inner = 8
        plot = pygame.Rect(
            rect.x + inner,
            rect.y + inner + 10,
            max(1, rect.w - 2 * inner),
            max(1, rect.h - 2 * inner - 10),
        )

        self._text(
            self.screen, "trend", (rect.x + inner, rect.y + 3),
            self.font_small, TEXT_DIM,
        )
        pygame.draw.line(
            self.screen, (70, 74, 100),
            (plot.x, plot.bottom), (plot.right, plot.bottom), 1,
        )
        pygame.draw.line(
            self.screen, (70, 74, 100),
            (plot.x, plot.y), (plot.x, plot.bottom), 1,
        )

        series = list(history)[-20:]
        if len(series) < 2:
            return rect.bottom + 8

        colors = [
            (200, 90, 120),
            (90, 180, 130),
            (120, 150, 230),
        ]
        for idx, col in enumerate(colors):
            pts = []
            for i, triplet in enumerate(series):
                val = max(0.0, min(1.0, float(triplet[idx])))
                px = plot.x + int(i * plot.w / max(1, len(series) - 1))
                py = plot.bottom - int(val * plot.h)
                pts.append((px, py))
            if len(pts) >= 2:
                pygame.draw.lines(self.screen, col, False, pts, 2)

        return rect.bottom + 8

    def _draw_right_panel(self) -> None:
        p = self._phase11
        rect = self._lay_right
        if rect is None or not self.ui_verbose or p is None or not p.show_panel:
            return
        pygame = self.pygame
        pygame.draw.rect(self.screen, ML_PANEL, rect, border_radius=10)
        pygame.draw.rect(
            self.screen, SECTION_BORDER, rect, width=1, border_radius=10,
        )

        inner_m = 12
        xin = rect.x + inner_m
        max_w = rect.w - 2 * inner_m
        y = rect.y + inner_m
        gap = 12

        # ML model section
        self._text(
            self.screen, "ML model", (xin, y), self.font_section, TEXT_LIGHT,
        )
        y += 22
        model_line = self._truncate(
            f"Active: {p.model_name}", self.font, max_w,
        )
        self._text(self.screen, model_line, (xin, y), self.font, ACCENT1)
        y += 20
        self._text(
            self.screen, "Confidence (0.00–1.00):",
            (xin, y), self.font_small, TEXT_DIM,
        )
        y += 18
        self._text(
            self.screen, f"Random:    {p.p_random:5.2f}",
            (xin, y), self.font, TEXT_LIGHT,
        )
        y += 18
        self._text(
            self.screen, f"Heuristic: {p.p_heuristic:5.2f}",
            (xin, y), self.font, TEXT_LIGHT,
        )
        y += 18
        self._text(
            self.screen, f"Maximax:   {p.p_maximax:5.2f}",
            (xin, y), self.font, TEXT_LIGHT,
        )
        y += gap

        # Playstyle section
        self._text(
            self.screen, "Playstyle", (xin, y),
            self.font_section, TEXT_LIGHT,
        )
        y += 22
        if p.has_playstyle:
            y = self._horiz_bars(
                xin, y, max_w,
                [
                    ("Chaotic", p.chaotic, (200, 90, 120)),
                    ("Cooperative", p.cooperative, (90, 180, 130)),
                    ("Strategic", p.strategic, (120, 150, 230)),
                ],
            )
            if p.playstyle_history and y + 86 <= rect.bottom:
                y = self._draw_playstyle_history(
                    xin, y + 4, max_w, p.playstyle_history,
                )
        else:
            self._text(
                self.screen,
                "Play a few turns so we can read your style.",
                (xin, y), self.font_small, TEXT_DIM,
            )
            y += 22
        y += gap

        # AI decision section
        if y + 60 > rect.bottom:
            return
        self._text(
            self.screen, "AI decision", (xin, y),
            self.font_section, TEXT_LIGHT,
        )
        y += 22
        if p.last_ai_name and p.last_action_kind:
            an = self._player_display_name(p.last_ai_name)
            line = self._truncate(
                f"{an} · {p.last_action_kind}", self.font, max_w,
            )
            self._text(self.screen, line, (xin, y), self.font, HIGHLIGHT)
            y += 20
        if p.last_explanation:
            ex = self._truncate(p.last_explanation, self.font_small, max_w)
            if y + 14 <= rect.bottom:
                self._text(
                    self.screen, ex, (xin, y),
                    self.font_small, TEXT_LIGHT,
                )
                y += 18
        if not p.last_ai_name:
            self._text(self.screen, "—", (xin, y), self.font, TEXT_DIM)

    # ------------------------------------------------------------------
    # Top-level render
    # ------------------------------------------------------------------
    def render(
        self,
        decoded: DecodedObservation,
        knowledge: KnowledgeView,
        current_agent: str,
        status_lines: List[str],
    ) -> None:
        self._pump_quit()
        self._current_status_h = self._status_h_for_lines(len(status_lines))
        self._compute_layout()

        self.screen.fill(BG_COLOR)
        self._draw_top_bar(decoded, current_agent)
        self._draw_partner_hands(decoded, current_agent)
        self._draw_discards(decoded.discards)
        self._draw_human_hand(knowledge)
        self._draw_right_panel()
        self._draw_status(status_lines)
        self.pygame.display.flip()

    # ------------------------------------------------------------------
    # Human input loop
    # ------------------------------------------------------------------
    def wait_for_human_action(
        self,
        decoded: DecodedObservation,
        knowledge: KnowledgeView,
        action_mask: np.ndarray,
        current_agent: str,
    ) -> int:
        pygame = self.pygame
        state = "IDLE"
        hint_target: Optional[int] = None
        hint_kind: Optional[str] = None
        msg = ""

        while True:
            if state == "IDLE":
                status = self._status_idle_choices()
            else:
                status = [self._prompt_for(state, hint_target, hint_kind)]
            if msg:
                status.append(msg)
            status.append(self._keys_help(state))
            status.append("Back: Esc  ·  Quit: Q")
            self.render(decoded, knowledge, current_agent, status)
            self.clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise UIClosed()
                if event.type != pygame.KEYDOWN:
                    continue
                k = event.key

                if k == pygame.K_q:
                    raise UIClosed()
                if k == pygame.K_ESCAPE:
                    state = "IDLE"; hint_target = None; hint_kind = None
                    msg = "Cancelled — back to the main action list."
                    continue

                if state == "IDLE":
                    if k == pygame.K_p:
                        state = "PLAY_SLOT"; msg = ""
                    elif k == pygame.K_d:
                        if decoded.information_tokens >= self.cfg.max_information_tokens:
                            msg = "Clue bank is full, but you may still discard if you need to."
                        state = "DISC_SLOT"
                    elif k == pygame.K_h:
                        if decoded.information_tokens <= 0:
                            msg = "You need a clue token to give a hint — try play or discard first."
                        else:
                            state = "HINT_TGT"; msg = ""
                    else:
                        msg = "Use P, D, or H to pick an action."

                elif state in ("PLAY_SLOT", "DISC_SLOT"):
                    slot = self._digit(k)
                    if slot is None or not (1 <= slot <= self.cfg.hand_size):
                        msg = f"Press a number 1 to {self.cfg.hand_size} for a card in your hand."
                        continue
                    atype = ActionType.PLAY if state == "PLAY_SLOT" else ActionType.DISCARD
                    try:
                        action_id = self.codec.encode(
                            StructuredAction(atype, slot=slot - 1)
                        )
                    except ValueError as exc:
                        msg = f"That choice is not valid: {exc}"
                        continue
                    if not self._legal(action_mask, action_id):
                        msg = (
                            f"You cannot {atype.name.lower()} that card right now "
                            f"(try another slot)."
                        )
                        state = "IDLE"; continue
                    return action_id

                elif state == "HINT_TGT":
                    off = self._digit(k)
                    if off is None or not (1 <= off <= self.cfg.players - 1):
                        msg = (
                            f"Press 1 for the next player, 2 for the one after, "
                            f"up to {self.cfg.players - 1}."
                        )
                        continue
                    hint_target = off
                    state = "HINT_TYPE"; msg = ""

                elif state == "HINT_TYPE":
                    if k == pygame.K_c:
                        hint_kind = "color"; state = "HINT_VAL"; msg = ""
                    elif k == pygame.K_r:
                        hint_kind = "rank"; state = "HINT_VAL"; msg = ""
                    else:
                        msg = "C = color hint, R = number (rank) hint"

                elif state == "HINT_VAL":
                    if hint_kind == "color":
                        color_idx = self._color_key(k)
                        if color_idx is None or not (0 <= color_idx < self.cfg.colors):
                            msg = "pick a color key: R Y G W B"; continue
                        structured = StructuredAction(
                            ActionType.REVEAL_COLOR,
                            target_offset=hint_target, color=color_idx,
                        )
                    else:
                        rank = self._digit(k)
                        if rank is None or not (1 <= rank <= self.cfg.ranks):
                            msg = f"pick a rank 1..{self.cfg.ranks}"; continue
                        structured = StructuredAction(
                            ActionType.REVEAL_RANK,
                            target_offset=hint_target, rank=rank - 1,
                        )
                    try:
                        action_id = self.codec.encode(structured)
                    except ValueError as exc:
                        msg = f"That hint is not valid: {exc}"
                        state = "IDLE"; hint_target = None; hint_kind = None
                        continue
                    if not self._legal(action_mask, action_id):
                        msg = (
                            "That hint is not allowed (no matching card, or not enough clues)."
                        )
                        state = "IDLE"; hint_target = None; hint_kind = None
                        continue
                    return action_id

    # ------------------------------------------------------------------
    # Prompts / key help
    # ------------------------------------------------------------------
    def _prompt_for(self, state, hint_target, hint_kind) -> str:
        if state == "IDLE":
            return "Your turn — choose an action (see list above)."
        if state == "PLAY_SLOT":
            return f"Play — which card? (number 1–{self.cfg.hand_size}, left to right)"
        if state == "DISC_SLOT":
            return f"Discard — which card? (number 1–{self.cfg.hand_size}, left to right)"
        if state == "HINT_TGT":
            return (
                "Hint — who should get the clue? "
                f"(1 = next player, then 2, …, up to {self.cfg.players - 1})"
            )
        if state == "HINT_TYPE":
            return "Color hint or number hint?  [C] color   [R] number (rank)"
        if state == "HINT_VAL":
            if hint_kind == "color":
                return (
                    "Pick the color to point at: "
                    "R Red · Y Yellow · G Green · W White · B Blue"
                )
            return (
                f"Pick the number to point at: 1–{self.cfg.ranks} on the keyboard"
            )
        return str(state)

    def _keys_help(self, state) -> str:
        if state == "IDLE":
            return "Shortcuts:  P  ·  D  ·  H"
        if state in ("PLAY_SLOT", "DISC_SLOT"):
            return f"Number keys:  1 … {self.cfg.hand_size}"
        if state == "HINT_TGT":
            return f"Number keys:  1 … {self.cfg.players - 1}"
        if state == "HINT_TYPE":
            return "C for color   ·   R for rank (number)"
        if state == "HINT_VAL":
            return "Color: R Y G W B   ·   Number: 1 … 5"
        return ""

    # ------------------------------------------------------------------
    # Key mapping
    # ------------------------------------------------------------------
    def _digit(self, k) -> Optional[int]:
        pygame = self.pygame
        table = {
            pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3, pygame.K_4: 4,
            pygame.K_5: 5, pygame.K_6: 6, pygame.K_7: 7, pygame.K_8: 8,
            pygame.K_9: 9,
            pygame.K_KP1: 1, pygame.K_KP2: 2, pygame.K_KP3: 3, pygame.K_KP4: 4,
            pygame.K_KP5: 5, pygame.K_KP6: 6, pygame.K_KP7: 7, pygame.K_KP8: 8,
            pygame.K_KP9: 9,
        }
        return table.get(k)

    def _color_key(self, k) -> Optional[int]:
        pygame = self.pygame
        return {
            pygame.K_r: 0,
            pygame.K_y: 1,
            pygame.K_g: 2,
            pygame.K_w: 3,
            pygame.K_b: 4,
        }.get(k)

    def _legal(self, action_mask, action_id: int) -> bool:
        if 0 <= action_id < len(action_mask):
            return int(action_mask[action_id]) == 1
        return False

    # ------------------------------------------------------------------
    # End-of-game screen
    # ------------------------------------------------------------------
    def wait_for_ack(
        self,
        decoded: DecodedObservation,
        knowledge: KnowledgeView,
        current_agent: str,
        headline: str,
        details: List[str],
    ) -> None:
        pygame = self.pygame
        while True:
            status = [headline, *details, "Press any key to close..."]
            self.render(decoded, knowledge, current_agent, status)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    return
            self.clock.tick(30)


__all__ = [
    "HanabiUI",
    "UIClosed",
    "CARD_COLORS",
    "Phase11ViewModel",
    "Phase11Highlights",
    "explain_ai_action",
    "phase11_from_ai_step",
    "short_action_kind",
    "DEFAULT_WINDOW_BASIC",
    "DEFAULT_WINDOW_VERBOSE",
]
