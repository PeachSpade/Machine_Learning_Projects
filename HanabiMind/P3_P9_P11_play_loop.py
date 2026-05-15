from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from P0_P1_project_config import HanabiConfig
from P0_observation_decoding import (
    ActionCodec,
    KnowledgeDecoder,
    ObservationDecoder,
    StructuredAction,
)
from P0_P1_environment import PettingZooHanabiRunner
from P1_P5_P6_P8_P10_controllers import Controller, MaximaxController
from P2_P7_trajectory_schema import TrajectoryStep
from P8_P9_P10_adaptive_system import PlaystyleTracker


class HumanController(Controller):
    def __init__(self, name: str, cfg: HanabiConfig, ui) -> None:
        super().__init__(name, cfg)
        self.ui = ui
        self.obs_decoder = ObservationDecoder(cfg)
        self.kn_decoder = KnowledgeDecoder(cfg)

    def select_action(self, observation_vec, action_mask, agent_name, history) -> int:
        decoded = self.obs_decoder.decode(observation_vec, agent_name)
        knowledge = self.kn_decoder.decode(observation_vec)
        return int(self.ui.wait_for_human_action(
            decoded, knowledge, np.asarray(action_mask), agent_name,
        ))


def _human_pov(env, cfg: HanabiConfig, human_agent: str):
    obs_decoder = ObservationDecoder(cfg)
    kn_decoder = KnowledgeDecoder(cfg)
    snap = env.observe(human_agent)
    vec = snap["observation"]
    return obs_decoder.decode(vec, human_agent), kn_decoder.decode(vec)


def play_phase3_game(
    cfg: Optional[HanabiConfig] = None,
    seed: int = 0,
    human_seat: int = 0,
    ai_delay_ms: int = 400,
    ai_controller_cls: Optional[type] = None,
    ai_controller_factory: Optional[object] = None,
    playstyle_tracker: Optional["PlaystyleTracker"] = None,
    ui_verbose: bool = False,
    display_ml_wrapper: Optional[object] = None,
    display_model_label: str = "",
) -> Dict[str, object]:
    from P3_P9_P11_pygame_ui import (
        HanabiUI,
        UIClosed,
        Phase11ViewModel,
        Phase11Highlights,
        phase11_from_ai_step,
    )

    cfg = cfg or HanabiConfig()
    if not (0 <= human_seat < cfg.players):
        raise ValueError(f"human_seat {human_seat} out of range 0..{cfg.players - 1}")

    if ai_controller_factory is not None:
        make_ai = ai_controller_factory
    else:
        ai_cls = ai_controller_cls or MaximaxController
        make_ai = lambda i, cfg_local: ai_cls(f"ai_{i}", cfg_local)

    human_agent = f"player_{human_seat}"
    runner = PettingZooHanabiRunner(cfg)
    env = runner.make_env()
    env.reset(seed=seed)

    ui = HanabiUI(cfg, human_agent, ui_verbose=ui_verbose)
    controllers: Dict[str, Controller] = {}
    for i in range(cfg.players):
        name = f"player_{i}"
        if i == human_seat:
            controllers[name] = HumanController(f"you_{i}", cfg, ui)
        else:
            controllers[name] = make_ai(i, cfg)

    obs_decoder = ObservationDecoder(cfg)
    kn_decoder = KnowledgeDecoder(cfg)
    codec = ActionCodec(cfg)
    history: List[TrajectoryStep] = []

    last_obs_vec = None
    last_agent = None
    closed_by_user = False

    last_playstyle: Optional[Dict[str, float]] = None
    last_playstyle_history: List[Tuple[float, float, float]] = []
    last_ml_triplet: Optional[Tuple[float, float, float]] = None
    ai_p11_block: Dict[str, object] = {
        "name": "",
        "kind": "",
        "expl": "",
        "hi": Phase11Highlights(),
    }

    def _build_phase11() -> Phase11ViewModel:
        p = Phase11ViewModel(
            show_panel=bool(ui_verbose),
            model_name=str(display_model_label or ""),
        )
        r, h, m = (last_ml_triplet if last_ml_triplet is not None
                   else (0.0, 0.0, 0.0))
        p.p_random, p.p_heuristic, p.p_maximax = r, h, m
        if last_playstyle is not None:
            p.chaotic = float(last_playstyle.get("chaotic", 0.0))
            p.cooperative = float(last_playstyle.get("cooperative", 0.0))
            p.strategic = float(last_playstyle.get("strategic", 0.0))
            p.playstyle_history = list(last_playstyle_history)
            p.has_playstyle = True
        p.last_ai_name = str(ai_p11_block["name"])
        p.last_action_kind = str(ai_p11_block["kind"])
        p.last_explanation = str(ai_p11_block["expl"])
        p.highlight = ai_p11_block["hi"]
        return p

    def _update_ml_from_history() -> None:
        nonlocal last_ml_triplet
        if not ui_verbose or display_ml_wrapper is None:
            return
        try:
            seq = [s for s in history if isinstance(s, TrajectoryStep)]
            if not seq:
                return
            pr = display_ml_wrapper.predict_proba(seq)
            if pr is not None and len(pr) >= 3:
                last_ml_triplet = (float(pr[0]), float(pr[1]), float(pr[2]))
        except Exception:
            pass

    try:
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            last_obs_vec = obs["observation"]
            last_agent = agent

            if termination or truncation:
                env.step(None)
                continue

            try:
                decoded_h, knowledge_h = _human_pov(env, cfg, human_agent)
            except Exception:
                decoded_h = obs_decoder.decode(obs["observation"], agent)
                knowledge_h = kn_decoder.decode(obs["observation"])

            if agent == human_agent:
                if ui_verbose:
                    ui.set_phase11(_build_phase11())
                action = controllers[agent].select_action(
                    obs["observation"], obs["action_mask"], agent, history,
                )
            else:
                if ui_verbose:
                    ui.set_phase11(_build_phase11())
                    hold_ms = min(max(0, int(ai_delay_ms)), 220)
                    ui.render(
                        decoded_h, knowledge_h, agent,
                        [f"{agent} is thinking...",
                         "press Q to quit at any time"],
                    )
                    if hold_ms:
                        ui.wait_ms(hold_ms)
                else:
                    ui.render(
                        decoded_h, knowledge_h, agent,
                        [f"{agent} is thinking...",
                         "press Q to quit at any time"],
                    )
                    ui.wait_ms(ai_delay_ms)
                action = controllers[agent].select_action(
                    obs["observation"], obs["action_mask"], agent, history,
                )

            logged_step: Optional[TrajectoryStep] = None
            structured: Optional[StructuredAction] = None
            try:
                decoded_self = obs_decoder.decode(obs["observation"], agent)
                structured = codec.decode(int(action))
                logged_step = TrajectoryStep(
                    agent=agent,
                    decoded_obs=decoded_self,
                    action_mask=np.asarray(obs["action_mask"]).copy(),
                    action_id=int(action),
                    decoded_action=structured,
                    reward=float(reward),
                    controller_label=controllers[agent].__class__.__name__.lower(),
                )
                history.append(logged_step)
            except Exception:
                logged_step = None
                if structured is None:
                    try:
                        structured = codec.decode(int(action))
                    except Exception:
                        structured = None

            if logged_step is not None:
                _update_ml_from_history()

            if playstyle_tracker is not None and logged_step is not None:
                try:
                    ps = playstyle_tracker.observe(logged_step)
                    if ps is not None:
                        last_playstyle = dict(ps)
                        last_playstyle_history.append((
                            float(last_playstyle.get("chaotic", 0.0)),
                            float(last_playstyle.get("cooperative", 0.0)),
                            float(last_playstyle.get("strategic", 0.0)),
                        ))
                        last_playstyle_history = last_playstyle_history[-40:]
                except Exception as exc:
                    print(f"[Phase 9] tracker error: {exc}")

            if (
                ui_verbose
                and agent != human_agent
                and structured is not None
            ):
                kn_a = kn_decoder.decode(obs["observation"])
                dec_a = obs_decoder.decode(obs["observation"], agent)
                k, ex, hi = phase11_from_ai_step(
                    agent, structured, cfg, kn_a, dec_a,
                )
                ai_p11_block["name"] = str(agent)
                ai_p11_block["kind"] = k
                ai_p11_block["expl"] = ex
                ai_p11_block["hi"] = hi
                ui.set_phase11(_build_phase11())
                rest_ms = max(0, int(ai_delay_ms) - 220)
                st_lines = [
                    f"{agent}  ->  {k}",
                    ex,
                    "press Q to quit at any time",
                ]
                ui.render(decoded_h, knowledge_h, agent, st_lines)
                if rest_ms:
                    ui.wait_ms(rest_ms)

            env.step(action)
    except UIClosed:
        closed_by_user = True
        print("[Phase 3] window closed - aborting game")

    env.close()

    final_decoded = ObservationDecoder(cfg).decode(last_obs_vec, last_agent)
    final_knowledge = KnowledgeDecoder(cfg).decode(last_obs_vec)
    fireworks_score = int(sum(final_decoded.fireworks))
    if final_decoded.life_tokens == 0:
        fireworks_score = 0

    if not closed_by_user:
        headline = f"GAME OVER - final score: {fireworks_score} / {cfg.colors * cfg.ranks}"
        details = [
            f"lives remaining: {final_decoded.life_tokens}",
            f"info tokens: {final_decoded.information_tokens}",
            f"deck: {final_decoded.deck_size}",
        ]
        try:
            if ui_verbose:
                ui.set_phase11(_build_phase11())
            ui.wait_for_ack(
                final_decoded, final_knowledge, last_agent or human_agent,
                headline, details,
            )
        except UIClosed:
            pass

    ui.close()
    print(f"[Phase 3] final score: {fireworks_score} / {cfg.colors * cfg.ranks}")

    saved_path: Optional[str] = None
    if playstyle_tracker is not None:
        try:
            saved_path = playstyle_tracker.save()
        except Exception as exc:
            print(f"[Phase 9] failed to persist playstyle history: {exc}")

    result: Dict[str, object] = {"score": fireworks_score, "aborted": closed_by_user}
    if playstyle_tracker is not None:
        result["playstyle_history"] = list(playstyle_tracker.history)
        result["playstyle_history_path"] = saved_path
    return result
