from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import random

import numpy as np

from P0_P1_project_config import HanabiConfig
from P0_observation_decoding import (
    ActionCodec,
    ActionType,
    Card,
    DecodedObservation,
    KnowledgeDecoder,
    ObservationDecoder,
    StructuredAction,
    agent_name_to_index,
)
from P1_game_state_simulation import (
    EvalWeights,
    GameState,
    build_state_from_observation,
    compute_wall,
    pool_remaining,
)
from P5_belief_sampling import (
    BeliefStats,
    BeliefSampler,
    SampledHand,
    build_own_hand_beliefs,
    commit_sample,
)
from P6_rollout_policy import (
    DEFAULT_HINT_SCORE_TUNE,
    RolloutKnowledge,
    RolloutPolicy,
    _playable_matrix,
    simulate_rollout,
)
from P2_P7_trajectory_schema import LABEL_TO_INDEX, LABEL_TO_PLAYSTYLE, TrajectoryStep


# default scoring weights used when no preset or override is provided
ROLLOUT_WEIGHTS: Dict[str, float] = {
    "play": 5.0,      # reward per fireworks card successfully played
    "life": 3.0,      # cost per life token spent
    "info": 1.0,      # small bonus for available info tokens
    "discard": 2.0,   # penalty for discarding cards that were still useful
    "progress": 2.0,  # bonus for overall fireworks progress as a fraction of max score
}

# named presets for easy experimentation without hardcoding numbers at call sites
ROLLOUT_CONFIGS: Dict[str, Dict[str, float]] = {
    "A": {   # baseline, same as the defaults above
        "play": 5.0,
        "life": 3.0,
        "info": 1.0,
        "discard": 2.0,
        "progress": 2.0,
    },
    "B": {   # aggressive: values progress over life preservation
        "play": 5.5,
        "life": 2.5,
        "info": 1.0,
        "discard": 1.8,
        "progress": 3.2,
    },
    "C": {   # conservative: punishes risky discards and values lives more
        "play": 4.5,
        "life": 4.0,
        "info": 1.0,
        "discard": 2.8,
        "progress": 1.3,
    },
}


def resolve_rollout_weights(
    config: Optional[str] = None,
    override: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    if override is not None:
        return dict(override)
    if config is not None and str(config).strip():
        k = str(config).strip().upper()
        if k not in ROLLOUT_CONFIGS:
            valid = ", ".join(sorted(ROLLOUT_CONFIGS))
            raise ValueError(
                f"Unknown rollout config {k!r}; use one of: {valid}"
            )
        return dict(ROLLOUT_CONFIGS[k])
    return dict(ROLLOUT_WEIGHTS)


def _count_relevant_wasted_discards(state: GameState) -> int:
    # counts discarded cards that were still in the playable window, i.e. cards we actually needed
    C, R = state.cfg.colors, state.cfg.ranks
    wall = compute_wall(state)
    total = 0
    for c in range(C):
        w = int(wall[c])
        f = int(state.fireworks[c])
        for r in range(f, min(R, w)):
            total += int(state.discards[c, r])
    return total


def evaluate_rollout_phase10(
    state: GameState,
    cfg: HanabiConfig,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    w = dict(weights) if weights is not None else dict(ROLLOUT_WEIGHTS)
    w_play = float(w["play"])
    w_life = float(w["life"])
    w_info = float(w["info"])
    w_disc = float(w["discard"])
    w_prog = float(w["progress"])
    max_s = int(cfg.colors * cfg.ranks)
    fs = int(state.fireworks_sum)
    # progress is a 0-1 fraction of the perfect score, used as a normalised bonus
    progress = float(fs) / float(max(1, max_s))
    if not state.is_alive:
        # heavily penalise losing all lives but still acknowledge how far the game got
        return -80.0 + w_play * float(fs) + w_prog * progress
    life_lost = int(cfg.max_life_tokens) - int(state.life_tokens)
    crit = _count_relevant_wasted_discards(state)
    return float(
        w_play * float(fs)
        - w_life * float(life_lost)
        + w_info * float(state.info_tokens)
        - w_disc * float(crit)
        + w_prog * progress
    )


def playstyle_action_alignment(
    act,
    p_play: float,
    playstyle: Optional[Dict[str, float]],
    lookahead_depth: int = 1,
) -> float:
    # returns 0 when playstyle is None so the score is unchanged in pure ai self-play mode
    if not playstyle:
        return 0.0
    chaos = float(playstyle.get("chaotic", 0.0))
    coop = float(playstyle.get("cooperative", 0.0))
    strat = float(playstyle.get("strategic", 0.0))
    align = 0.0
    if act.type in (ActionType.REVEAL_COLOR, ActionType.REVEAL_RANK):
        # hints are cooperative by nature and mildly strategic, chaotic players still give some
        align += 0.55 * coop + 0.35 * strat + 0.2 * chaos
    elif act.type is ActionType.PLAY:
        # strategic players get rewarded for high-confidence plays, chaotic players penalised
        align += 0.45 * strat * p_play
        align -= 0.65 * chaos * max(0.0, 0.85 - p_play)
    elif act.type is ActionType.DISCARD:
        align += 0.15 * chaos + 0.1 * strat
    if lookahead_depth > 1:
        # at deeper search the strategic bonus grows slightly since planning further ahead is strategic
        align += 0.1 * strat * min(1.0, 0.35 * (lookahead_depth - 1))
    return float(align)


def legal_actions_from_mask(action_mask: np.ndarray) -> List[int]:
    # extracts the list of legal action ids from the binary action mask
    return np.flatnonzero(np.asarray(action_mask)).astype(int).tolist()


class Controller(ABC):
    def __init__(self, name: str, cfg: HanabiConfig):
        self.name = name
        self.cfg = cfg

    @abstractmethod
    def select_action(self, observation_vec, action_mask, agent_name, history) -> int:
        pass


class RandomController(Controller):
    def select_action(self, observation_vec, action_mask, agent_name, history):
        legal = legal_actions_from_mask(action_mask)
        return int(random.choice(legal))


def _adaptive_sample_count(requested: int, depth: int) -> int:
    # at depth 4+ we halve the sample budget for each extra depth step to stay tractable
    if depth <= 3:
        return requested
    reduction = 2 ** (depth - 3)
    return max(10, requested // reduction)


@dataclass
class MaximaxStats:
    # accumulated counters across all decisions made by one controller instance
    decisions: int = 0
    decisions_marginal_fallback: int = 0    # decisions where sampling failed and we used per-slot marginals
    decisions_with_samples: int = 0
    samples_drawn: int = 0
    unique_samples_sum: int = 0             # sum of unique sample counts, used to compute average
    greedy_samples: int = 0
    rejection_samples: int = 0
    belief: BeliefStats = field(default_factory=BeliefStats)

    def merge(self, other: "MaximaxStats") -> None:
        self.decisions += other.decisions
        self.decisions_marginal_fallback += other.decisions_marginal_fallback
        self.decisions_with_samples += other.decisions_with_samples
        self.samples_drawn += other.samples_drawn
        self.unique_samples_sum += other.unique_samples_sum
        self.greedy_samples += other.greedy_samples
        self.rejection_samples += other.rejection_samples
        self.belief.merge(other.belief)

    def summary(self) -> str:
        n = max(1, self.decisions)
        d_with = max(1, self.decisions_with_samples)
        marg_rate = self.decisions_marginal_fallback / n
        avg_samples = self.samples_drawn / d_with if self.decisions_with_samples else 0.0
        avg_unique = self.unique_samples_sum / d_with if self.decisions_with_samples else 0.0
        total_samples = max(1, self.samples_drawn)
        greedy_rate = self.greedy_samples / total_samples
        rejection_rate = self.rejection_samples / total_samples
        return (
            f"[MaximaxStats] decisions={self.decisions}  "
            f"marginal_fallback={self.decisions_marginal_fallback} ({marg_rate:.1%})  "
            f"avg_samples/decision={avg_samples:.1f}  "
            f"avg_unique/decision={avg_unique:.1f}  "
            f"rejection={rejection_rate:.1%}  "
            f"greedy={greedy_rate:.1%}\n"
            f"    belief: {self.belief.summary()}"
        )


@dataclass
class MLGuidanceStats:
    # tracks how much influence the ml model had on actual action choices
    decisions: int = 0
    candidates_scored: int = 0
    action_changes: int = 0        # how many times ml shifted the final pick away from the rollout best
    sum_rollout_best: float = 0.0  # accumulated for computing averages later
    sum_ml_best: float = 0.0
    sum_ml_mean: float = 0.0

    def merge(self, other: "MLGuidanceStats") -> None:
        self.decisions += other.decisions
        self.candidates_scored += other.candidates_scored
        self.action_changes += other.action_changes
        self.sum_rollout_best += other.sum_rollout_best
        self.sum_ml_best += other.sum_ml_best
        self.sum_ml_mean += other.sum_ml_mean

    def summary(self) -> str:
        n = max(1, self.decisions)
        chg_rate = self.action_changes / n
        avg_rollout = self.sum_rollout_best / n
        avg_ml_best = self.sum_ml_best / n
        avg_ml_mean = self.sum_ml_mean / n
        return (
            f"[MLGuidanceStats] decisions={self.decisions}  "
            f"action_changes={self.action_changes} ({chg_rate:.1%})  "
            f"avg_rollout_best={avg_rollout:.3f}  "
            f"avg_ml_best={avg_ml_best:.3f}  avg_ml_mean={avg_ml_mean:.3f}"
        )


class MaximaxController(Controller):

    def __init__(
        self,
        name: str,
        cfg: HanabiConfig,
        weights: Optional[EvalWeights] = None,
        num_samples: int = 80,             # how many own-hand hypotheses to draw per decision
        max_sample_attempts: int = 200,    # rejection sampling budget before falling back to greedy
        rng_seed: Optional[int] = None,
        debug: bool = False,
        lookahead_depth: int = 1,          # 1 means one-step, higher values add rollout turns
        rollout_policy: Optional[RolloutPolicy] = None,
        ml_model: Optional[object] = None,
        ml_weight: float = 0.1,            # how much to scale the ml score relative to the rollout score
        ml_target_class: str = "maximax",  # which controller label the ml model should favour
        ml_debug: bool = False,
        rollout_weights: Optional[Dict[str, float]] = None,
        hint_score_tune: Optional[Dict[str, float]] = None,
        play_confidence_threshold: float = 0.8,  # below this p(playable) a risk penalty kicks in
        playstyle_weight: float = 0.12,
        playstyle_probs: Optional[Dict[str, float]] = None,
        risk_penalty_scale: float = 3.5,
    ) -> None:
        super().__init__(name, cfg)
        self.codec = ActionCodec(cfg)
        self.obs_decoder = ObservationDecoder(cfg)
        self.kn_decoder = KnowledgeDecoder(cfg)
        self.weights = weights or EvalWeights()
        self._rollout_w: Dict[str, float] = dict(rollout_weights or ROLLOUT_WEIGHTS)
        self.num_samples = max(0, int(num_samples))
        self.max_sample_attempts = int(max_sample_attempts)
        self.debug = bool(debug)
        self.stats = MaximaxStats()
        self.lookahead_depth = max(1, int(lookahead_depth))
        if rollout_policy is not None:
            self.rollout_policy = rollout_policy
        else:
            self.rollout_policy = RolloutPolicy(
                hint_eval_budget=48,
                early_exit_strong_hint=False,
                hint_score_tune=hint_score_tune,
            )
        # per-instance rng keeps results reproducible when a seed is given
        self._rng = random.Random(rng_seed) if rng_seed is not None else random.Random()
        self.ml_model = ml_model
        self.ml_weight = float(ml_weight)
        self.ml_target_class = ml_target_class
        self.ml_debug = bool(ml_debug)
        self.ml_stats = MLGuidanceStats()
        self.play_confidence_threshold = float(play_confidence_threshold)
        self.playstyle_weight = float(playstyle_weight)
        self.playstyle_probs = playstyle_probs
        self.risk_penalty_scale = float(risk_penalty_scale)

    @property
    def _num_rollout_turns(self) -> int:
        # depth 1 means evaluate only the root action, so zero additional simulated turns
        return max(0, self.lookahead_depth - 1)

    def _rollout_after(
        self,
        pre_state: GameState,
        next_state: GameState,
        action,
        observer: int,
        rollout_kn_init: Optional[RolloutKnowledge],
    ) -> GameState:
        n = self._num_rollout_turns
        if n <= 0 or rollout_kn_init is None or next_state.is_terminal():
            return next_state
        # clone the knowledge so each candidate action gets a fresh copy to mutate
        kn = rollout_kn_init.clone()
        kn.apply_action(pre_state, action, observer)
        return simulate_rollout(next_state, kn, self.rollout_policy, n)

    def score_action_sampled(
        self,
        base_state: GameState,
        action,
        observer: int,
        samples,
        unknown_slots: List[int],
        rollout_kn_init: Optional[RolloutKnowledge] = None,
    ) -> float:
        if action.type in (ActionType.REVEAL_COLOR, ActionType.REVEAL_RANK):
            if base_state.info_tokens <= 0:
                return -float("inf")

        if not samples:
            return -float("inf")

        total = 0.0
        count = 0
        for sample in samples:
            # commit this hand hypothesis into a cloned state and step forward
            hypothesis = commit_sample(base_state, sample, unknown_slots, observer)
            try:
                next_state = hypothesis.step(action, observer)
            except ValueError:
                continue
            next_state = self._rollout_after(
                hypothesis, next_state, action, observer, rollout_kn_init,
            )
            total += evaluate_rollout_phase10(
                next_state, self.cfg, self._rollout_w,
            )
            count += 1
        if count == 0:
            return -float("inf")
        # average over all sampled worlds so we get an expected value
        return total / count

    def score_action_marginal(
        self,
        base_state: GameState,
        action,
        observer: int,
        knowledge,
        remaining: np.ndarray,
        rollout_kn_init: Optional[RolloutKnowledge] = None,
    ) -> float:
        if action.type in (ActionType.REVEAL_COLOR, ActionType.REVEAL_RANK):
            if base_state.info_tokens <= 0:
                return -float("inf")
            next_state = base_state.step(action, observer)
            next_state = self._rollout_after(
                base_state, next_state, action, observer, rollout_kn_init,
            )
            return evaluate_rollout_phase10(
                next_state, self.cfg, self._rollout_w,
            )

        slot = action.slot
        possibility = knowledge.per_player[0][slot].possibility
        # weight each card hypothesis by how many copies are still in the remaining pool
        weights_mat = remaining * possibility.astype(int)
        total = int(weights_mat.sum())
        if total <= 0:
            return -float("inf")

        expected = 0.0
        for c in range(self.cfg.colors):
            for r in range(self.cfg.ranks):
                w = int(weights_mat[c, r])
                if w <= 0:
                    continue
                prob = w / total
                hypothesis = base_state.clone()
                hypothesis.hands[observer][slot] = Card(color=c, rank=r)
                next_state = hypothesis.step(action, observer)
                next_state = self._rollout_after(
                    hypothesis, next_state, action, observer, rollout_kn_init,
                )
                expected += prob * evaluate_rollout_phase10(
                    next_state, self.cfg, self._rollout_w,
                )
        return expected

    def _p_playable_slot(
        self,
        base_state: GameState,
        knowledge,
        observer: int,
        slot: int,
    ) -> float:
        # estimates the probability that this own-hand slot is currently playable
        own = knowledge.per_player[0]
        if slot < 0 or slot >= len(own):
            return 0.0
        pmask = own[slot].possibility
        pool = pool_remaining(base_state, observer)
        w = pmask.astype(np.float64) * pool.astype(np.float64)
        tot = float(w.sum())
        if tot <= 0:
            return 0.0
        pl = _playable_matrix(base_state)
        return float((w * pl).sum() / tot)

    def select_action(self, observation_vec, action_mask, agent_name, history):
        decoded = self.obs_decoder.decode(observation_vec, agent_name)
        knowledge = self.kn_decoder.decode(observation_vec)
        observer = agent_name_to_index(agent_name)

        base_state = build_state_from_observation(decoded, self.cfg, observer)
        remaining = pool_remaining(base_state, observer)
        legal_ids = legal_actions_from_mask(action_mask)

        self.stats.decisions += 1

        # scale down sample count at high depths to keep per-decision time manageable
        effective_samples = _adaptive_sample_count(self.num_samples, self.lookahead_depth)

        beliefs = build_own_hand_beliefs(base_state, knowledge, observer)
        samples: List[SampledHand] = []
        if effective_samples > 0 and beliefs.feasible():
            sampler = BeliefSampler(
                beliefs,
                rng=self._rng,
                max_attempts=self.max_sample_attempts,
                stats=self.stats.belief,
            )
            samples = sampler.sample_many(effective_samples)

        # fall back to per-slot marginals if sampling produced nothing but we have unknown cards
        use_marginal_fallback = len(samples) == 0 and beliefs.num_unknown > 0

        if samples:
            self.stats.decisions_with_samples += 1
            self.stats.samples_drawn += len(samples)
            unique = len({s.key() for s in samples})
            self.stats.unique_samples_sum += unique
            for s in samples:
                if s.source == "greedy":
                    self.stats.greedy_samples += 1
                elif s.source == "rejection":
                    self.stats.rejection_samples += 1
        if use_marginal_fallback:
            self.stats.decisions_marginal_fallback += 1

        if self.debug:
            path = "marginal" if use_marginal_fallback else "sampled"
            uniq_str = (
                f" unique={len({s.key() for s in samples})}/{len(samples)}"
                if samples else ""
            )
            print(
                f"  [maximax debug] agent={agent_name} decision={self.stats.decisions} "
                f"path={path} num_samples={len(samples)}{uniq_str}"
            )

        # build rollout knowledge once per decision and clone it per action candidate
        rollout_kn_init: Optional[RolloutKnowledge] = None
        if self._num_rollout_turns > 0:
            rollout_kn_init = RolloutKnowledge.from_view(
                knowledge, observer, base_state, self.cfg,
            )

        structured_actions: List = []
        rollout_scores: List[float] = []
        for action_id in legal_ids:
            structured = self.codec.decode(int(action_id))
            if use_marginal_fallback:
                score = self.score_action_marginal(
                    base_state, structured, observer, knowledge, remaining,
                    rollout_kn_init,
                )
            else:
                score = self.score_action_sampled(
                    base_state, structured, observer, samples,
                    beliefs.unknown_slots, rollout_kn_init,
                )
            structured_actions.append(structured)
            rollout_scores.append(score)

        # ml scores default to zero if no model is loaded, so the final sum is unchanged
        ml_scores: List[float] = [0.0] * len(legal_ids)
        if self.ml_model is not None and legal_ids:
            ml_scores = self._ml_action_scores(
                legal_ids, structured_actions, decoded, action_mask, agent_name, history,
            )

        risk_adjusted: List[float] = []
        p_play_for_style: List[float] = []
        for i, rs in enumerate(rollout_scores):
            st = structured_actions[i]
            adj = rs
            if st.type is ActionType.PLAY:
                p = self._p_playable_slot(
                    base_state, knowledge, observer, st.slot,
                )
                p_play_for_style.append(p)
                # penalise low-confidence plays proportionally to how uncertain we are
                if p < self.play_confidence_threshold:
                    adj -= self.risk_penalty_scale * (
                        self.play_confidence_threshold - p
                    )
            else:
                p_play_for_style.append(0.0)
            risk_adjusted.append(adj)

        # keep track of what pure rollout would have picked for ml diagnostic logging
        best_rollout_idx = 0
        best_rollout_score = -float("inf")
        for i, s in enumerate(rollout_scores):
            if s > best_rollout_score:
                best_rollout_score = s
                best_rollout_idx = i

        best_idx = 0
        best_score = -float("inf")
        for i, _ in enumerate(risk_adjusted):
            st = structured_actions[i]
            ml_part = self.ml_weight * ml_scores[i] if self.ml_model is not None else 0.0
            st_align = playstyle_action_alignment(
                st,
                p_play_for_style[i],
                self.playstyle_probs,
                self.lookahead_depth,
            )
            # final score combines rollout, risk adjustment, ml prior, and playstyle nudge
            final_score = (
                risk_adjusted[i]
                + ml_part
                + self.playstyle_weight * st_align
            )
            if final_score > best_score:
                best_score = final_score
                best_idx = i

        # if every action scored -inf (extremely rare), pick randomly to avoid crashing
        if not legal_ids or best_score == -float("inf"):
            best_action = random.choice(legal_ids) if legal_ids else 0
        else:
            best_action = legal_ids[best_idx]

        if self.ml_model is not None and legal_ids:
            self.ml_stats.decisions += 1
            self.ml_stats.candidates_scored += len(legal_ids)
            if best_idx != best_rollout_idx:
                self.ml_stats.action_changes += 1
            self.ml_stats.sum_rollout_best += float(best_rollout_score)
            self.ml_stats.sum_ml_best += float(ml_scores[best_idx])
            self.ml_stats.sum_ml_mean += float(sum(ml_scores) / max(1, len(ml_scores)))

            if self.ml_debug:
                ml_best = ml_scores[best_idx]
                rb = rollout_scores[best_idx]
                ra = risk_adjusted[best_idx]
                changed = "*" if best_idx != best_rollout_idx else " "
                print(
                    f"  [phase8 debug{changed}] agent={agent_name} "
                    f"pick_id={int(best_action)}  rollout={rb:.3f}  risk_adj={ra:.3f}  "
                    f"ml_p={ml_best:.3f}  weight={self.ml_weight:.3g}  "
                    f"final={best_score:.3f}"
                )

        return int(best_action)

    def _ml_action_scores(
        self,
        legal_ids: List[int],
        structured_actions: List,
        decoded_obs: DecodedObservation,
        action_mask,
        agent_name: str,
        history: List,
    ) -> List[float]:
        if self.ml_model is None or not legal_ids:
            return [0.0] * len(legal_ids)

        # filter history to only actual trajectory steps in case the list has mixed content
        prefix: List[TrajectoryStep] = [
            s for s in (history or []) if isinstance(s, TrajectoryStep)
        ]

        mask_arr = np.asarray(action_mask).copy()
        sequences: List[List[TrajectoryStep]] = []
        for action_id, structured in zip(legal_ids, structured_actions):
            # build a hypothetical step appending this candidate action to the game so far
            hypo = TrajectoryStep(
                agent=agent_name,
                decoded_obs=decoded_obs,
                action_mask=mask_arr,
                action_id=int(action_id),
                decoded_action=structured,
                reward=0.0,
                controller_label=self.ml_target_class,
            )
            sequences.append(prefix + [hypo])

        try:
            probs = self.ml_model.predict_proba_batch(sequences)
        except Exception as exc:
            # ml inference is best-effort, we degrade to zero contribution on failure
            if self.ml_debug:
                print(f"  [phase8 debug] ML inference failed: {exc!r}")
            return [0.0] * len(legal_ids)

        label_map = getattr(self.ml_model, "label_map", LABEL_TO_INDEX)
        target_idx = label_map.get(self.ml_target_class, LABEL_TO_INDEX.get("maximax", 0))
        # clamp index so we never go out of bounds if the model has fewer classes
        target_idx = max(0, min(int(target_idx), probs.shape[1] - 1))
        return [float(p[target_idx]) for p in probs]


class HeuristicController(Controller):

    def __init__(
        self,
        name: str,
        cfg: HanabiConfig,
        policy: Optional[RolloutPolicy] = None,
    ) -> None:
        super().__init__(name, cfg)
        self.codec = ActionCodec(cfg)
        self.obs_decoder = ObservationDecoder(cfg)
        self.kn_decoder = KnowledgeDecoder(cfg)
        # use a wider hint search budget since each turn's choice here is final, not amortised
        self.policy = policy or RolloutPolicy(
            hint_eval_budget=64,
            early_exit_strong_hint=False,
        )

    def select_action(self, observation_vec, action_mask, agent_name, history):
        decoded = self.obs_decoder.decode(observation_vec, agent_name)
        knowledge = self.kn_decoder.decode(observation_vec)
        observer = agent_name_to_index(agent_name)
        base_state = build_state_from_observation(decoded, self.cfg, observer)

        legal_ids = legal_actions_from_mask(action_mask)
        legal_set = set(int(a) for a in legal_ids)

        rollout_kn = RolloutKnowledge.from_view(
            knowledge, observer, base_state, self.cfg,
        )
        action = self.policy.select(base_state, rollout_kn, observer)
        if action is not None:
            try:
                action_id = self.codec.encode(action)
            except ValueError:
                action_id = -1
            if action_id in legal_set:
                return action_id

        # defensive fallback: try discards then hints then the first legal action
        for slot in range(self.cfg.hand_size):
            aid = self.codec.encode(StructuredAction(ActionType.DISCARD, slot=slot))
            if aid in legal_set:
                return aid
        for off in range(1, self.cfg.players):
            for card in base_state.hands[(observer + off) % self.cfg.players]:
                if card is None:
                    continue
                aid = self.codec.encode(StructuredAction(
                    ActionType.REVEAL_RANK, target_offset=off, rank=int(card.rank),
                ))
                if aid in legal_set:
                    return aid
        return int(legal_ids[0])


class PlaystyleBiasedController(Controller):
    # wraps another controller and occasionally overrides its choice to inject style-specific behaviour
    def __init__(
        self,
        name: str,
        cfg: HanabiConfig,
        base: Controller,
        style: str,
        rng_seed: Optional[int] = None,
        override_prob: float = 0.65,   # probability of applying the style override each turn
    ) -> None:
        super().__init__(name, cfg)
        self.base = base
        self.style = str(style)
        self.codec = ActionCodec(cfg)
        self.rng = random.Random(rng_seed)
        self.override_prob = max(0.0, min(1.0, float(override_prob)))

    def select_action(self, observation_vec, action_mask, agent_name, history):
        legal_ids = legal_actions_from_mask(action_mask)
        if not legal_ids:
            return 0

        base_action = int(self.base.select_action(
            observation_vec, action_mask, agent_name, history,
        ))
        # skip the override most of the time to keep games realistic
        if self.rng.random() > self.override_prob:
            return base_action

        decoded = []
        for action_id in legal_ids:
            try:
                decoded.append((int(action_id), self.codec.decode(int(action_id))))
            except ValueError:
                continue
        if not decoded:
            return base_action

        def by_type(*types):
            wanted = set(types)
            return [aid for aid, action in decoded if action.type in wanted]

        if self.style == "chaotic":
            plays = by_type(ActionType.PLAY)
            discards = by_type(ActionType.DISCARD)
            roll = self.rng.random()
            if discards and roll < 0.65:
                return int(self.rng.choice(discards))
            if plays and roll < 0.88:
                return int(self.rng.choice(plays))
            return int(self.rng.choice(legal_ids))

        if self.style == "cooperative":
            rank_hints = by_type(ActionType.REVEAL_RANK)
            color_hints = by_type(ActionType.REVEAL_COLOR)
            # prefer rank hints since they tend to be more informative in this game
            hints = rank_hints or color_hints
            if hints:
                return int(self.rng.choice(hints))
            return base_action

        if self.style == "strategic":
            plays = by_type(ActionType.PLAY)
            # small chance to play even without certainty, keeping the style recognisable
            if plays and self.rng.random() < 0.25:
                return int(self.rng.choice(plays))
            return base_action

        return base_action
