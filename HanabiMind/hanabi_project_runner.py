from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from P0_P1_project_config import HanabiConfig
from P0_observation_decoding import *  # noqa: F401,F403
from P1_game_state_simulation import *  # noqa: F401,F403
from P5_belief_sampling import *  # noqa: F401,F403
from P6_rollout_policy import *  # noqa: F401,F403
from P6_rollout_policy import DEFAULT_HINT_SCORE_TUNE
from P2_P7_trajectory_schema import *  # noqa: F401,F403
from P1_P5_P6_P8_P10_controllers import *  # noqa: F401,F403
from P0_P1_environment import *  # noqa: F401,F403
from P1_P5_foundation_checks import *  # noqa: F401,F403
from P2_P7_ml_training import *  # noqa: F401,F403
from P3_P9_P11_play_loop import *  # noqa: F401,F403
from P4_P6_controller_evaluation import *  # noqa: F401,F403
from P8_P9_P10_adaptive_system import *  # noqa: F401,F403
from P8_P9_P10_adaptive_system import _phase8_resolve_path


PIPELINE_STEPS: Tuple[Dict[str, object], ...] = (
    {
        "order": 0,
        "legacy_flag": "--phase0",
        "clear_flag": "--validate-decoding",
        "entrypoint": "validate_observation_decoding",
        "purpose": "Decode actions, observations, and card-knowledge masks.",
        "owner_area": "Environment decoding and invariants",
    },
    {
        "order": 1,
        "legacy_flag": "--phase1",
        "clear_flag": "--compare-baselines",
        "entrypoint": "compare_baseline_controllers",
        "purpose": "Compare random play with the baseline Maximax controller.",
        "owner_area": "Game state simulation and baseline search",
    },
    {
        "order": 2,
        "legacy_flag": "--phase2",
        "clear_flag": "--train-gru-behavior-clone",
        "entrypoint": "train_gru_behavior_clone",
        "purpose": "Generate trajectories and train the first GRU classifier.",
        "owner_area": "Dataset generation and feature extraction",
    },
    {
        "order": 3,
        "legacy_flag": "--phase3-only",
        "clear_flag": "--play-ui / --gradio-ui",
        "entrypoint": "launch_interactive_game",
        "purpose": "Run the local pygame UI or the notebook Gradio UI.",
        "owner_area": "Interactive human-facing game loop",
    },
    {
        "order": 4,
        "legacy_flag": "--phase4-only",
        "clear_flag": "--compare-controllers",
        "entrypoint": "compare_controller_scores",
        "purpose": "Plot baseline controller scores.",
        "owner_area": "Evaluation harnesses and result plots",
    },
    {
        "order": 5,
        "legacy_flag": "--phase5",
        "clear_flag": "--validate-beliefs",
        "entrypoint": "validate_belief_sampling",
        "purpose": "Verify own-hand belief samples against global card counts.",
        "owner_area": "Belief modeling",
    },
    {
        "order": 6,
        "legacy_flag": "--phase6-only",
        "clear_flag": "--compare-rollouts",
        "entrypoint": "compare_rollout_lookahead",
        "purpose": "Compare one-step search with knowledge-aware rollout depth.",
        "owner_area": "Rollout policy and lookahead search",
    },
    {
        "order": 7,
        "legacy_flag": "--phase7-only",
        "clear_flag": "--train-sequence-models",
        "entrypoint": "train_sequence_classifiers",
        "purpose": "Train GRU, LSTM, and Transformer behavior classifiers.",
        "owner_area": "Sequence-model training and metrics",
    },
    {
        "order": 8,
        "legacy_flag": "--phase8-only",
        "clear_flag": "--compare-ml-guided-search",
        "entrypoint": "compare_ml_guided_search",
        "purpose": "Evaluate Maximax with ML action priors.",
        "owner_area": "Hybrid ML plus search",
    },
    {
        "order": 9,
        "legacy_flag": "--phase9",
        "clear_flag": "--track-playstyle",
        "entrypoint": "build_playstyle_tracker",
        "purpose": "Estimate human playstyle from recent actions.",
        "owner_area": "Playstyle modeling",
    },
    {
        "order": 10,
        "legacy_flag": "--phase10-only",
        "clear_flag": "--compare-integrated-controller",
        "entrypoint": "compare_integrated_controller",
        "purpose": "Evaluate rollout, ML guidance, playstyle, and risk together.",
        "owner_area": "Final integrated controller",
    },
)


validate_observation_decoding = run_phase0_check
compare_baseline_controllers = run_phase1_compare
train_gru_behavior_clone = run_phase2
launch_interactive_game = play_phase3_game
compare_controller_scores = run_phase4_eval
validate_belief_sampling = run_phase5_belief_check
compare_rollout_lookahead = run_phase6_compare
train_sequence_classifiers = run_phase7
compare_ml_guided_search = run_phase8_compare
compare_integrated_controller = run_phase10_compare
build_playstyle_tracker = build_phase9_tracker


def build_gradio_notebook_ui(default_seed: int = 0):
    from P3_P9_P11_gradio_ui import build_gradio_ui
    return build_gradio_ui(default_seed=default_seed)


def launch_gradio_notebook_ui(default_seed: int = 0, **launch_kwargs):
    from P3_P9_P11_gradio_ui import launch_gradio_notebook_ui as _launch
    return _launch(default_seed=default_seed, **launch_kwargs)


def _parse_phase10_ml_weights(s: Optional[str]) -> Optional[List[float]]:
    s = (s or "").strip()
    if not s:
        return None
    out: List[float] = []
    for part in s.split(","):
        p = part.strip()
        if p:
            out.append(float(p))
    return out or None


def _cli_hint_score_tune(args) -> Optional[Dict[str, float]]:
    d: Dict[str, float] = {}
    b = getattr(args, "hint_rank_immediate_bonus", None)
    c = getattr(args, "hint_color_breadth_penalty", None)
    if b is not None:
        d["w_rank_immediate"] = float(b)
    if c is not None:
        d["w_color_breadth"] = float(c)
    return d if d else None


def _parse_cli_args(argv: Optional[List[str]] = None):
    import argparse
    parser = argparse.ArgumentParser(description="Hanabi project runner")
    parser.add_argument("--validate-decoding", "--phase0", dest="phase0",
                        action="store_true")
    parser.add_argument("--compare-baselines", "--phase1", dest="phase1",
                        action="store_true")
    parser.add_argument("--train-gru-behavior-clone", "--phase2", dest="phase2",
                        action="store_true")
    parser.add_argument("--phase2-only", action="store_true")
    parser.add_argument("--phase2-games", type=int, default=40)
    parser.add_argument("--phase2-depth", type=int, default=1)
    parser.add_argument("--phase3", action="store_true")
    parser.add_argument("--phase3-only", "--play-ui", dest="phase3_only",
                        action="store_true")
    parser.add_argument("--gradio-ui", "--notebook-ui", dest="gradio_ui",
                        action="store_true")
    parser.add_argument("--phase4", action="store_true")
    parser.add_argument("--phase4-only", "--compare-controllers",
                        dest="phase4_only", action="store_true")
    parser.add_argument("--phase4-games", type=int, default=30)
    parser.add_argument("--phase4-debug-belief", action="store_true")
    parser.add_argument("--phase4-samples", type=int, default=80)
    parser.add_argument("--validate-beliefs", "--phase5", dest="phase5",
                        action="store_true")
    parser.add_argument("--phase5-checks", type=int, default=5)
    parser.add_argument("--phase5-samples", type=int, default=25)
    parser.add_argument("--phase6", action="store_true")
    parser.add_argument("--phase6-only", "--compare-rollouts",
                        dest="phase6_only", action="store_true")
    parser.add_argument("--phase6-games", type=int, default=20)
    parser.add_argument("--phase6-samples", type=int, default=80)
    parser.add_argument("--phase6-depth", type=int, default=2,
                        help="lookahead depth for the deep Maximax in Phase 6 (default: 2, supports 6 or 7 for deeper search)")
    parser.add_argument("--phase7", action="store_true")
    parser.add_argument("--phase7-only", "--train-sequence-models",
                        dest="phase7_only", action="store_true")
    parser.add_argument("--phase7-games", type=int, default=40)
    parser.add_argument("--phase7-epochs", type=int, default=15)
    parser.add_argument("--phase7-batch", type=int, default=8)
    parser.add_argument("--phase7-hidden", type=int, default=64)
    parser.add_argument("--phase7-lr", type=float, default=1e-3)
    parser.add_argument("--phase7-windowed-playstyle", action="store_true")
    parser.add_argument("--phase7-style-window", type=int, default=10)
    parser.add_argument("--phase7-style-biased-data", action="store_true")
    parser.add_argument("--phase8", action="store_true")
    parser.add_argument("--phase8-only", "--compare-ml-guided-search",
                        dest="phase8_only", action="store_true")
    parser.add_argument("--phase8-games", type=int, default=30)
    parser.add_argument("--phase8-samples", type=int, default=80)
    parser.add_argument("--phase8-depth", type=int, default=1,
                        help="Maximax lookahead depth for Phase 8/10 (supports 1-7; deeper is slower; default: 1)")
    parser.add_argument("--phase8-weight", type=float, default=0.1)
    parser.add_argument(
        "--rollout-config",
        type=str,
        default="A",
        choices=tuple(sorted(ROLLOUT_CONFIGS.keys())),
    )
    parser.add_argument("--phase10-ml-weights", type=str, default="")
    parser.add_argument("--hint-rank-immediate-bonus", type=float, default=None)
    parser.add_argument("--hint-color-breadth-penalty", type=float, default=None)
    parser.add_argument("--no-phase10-5-table", action="store_true")
    parser.add_argument("--phase8-debug-ml", action="store_true")
    parser.add_argument("--phase8-train", action="store_true")
    parser.add_argument("--phase10", action="store_true")
    parser.add_argument("--phase10-only", "--compare-integrated-controller",
                        action="store_true", dest="phase10_only")
    parser.add_argument("--skip-phase2", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--human-seat", type=int, default=0)
    parser.add_argument("--ai-delay-ms", type=int, default=400)
    parser.add_argument(
        "--ai",
        choices=[
            "maximax", "heuristic", "random",
            "maximax_gru", "maximax_lstm", "maximax_transformer",
        ],
        default="maximax",
    )
    parser.add_argument("--ai-ml-weight", type=float, default=0.1)
    parser.add_argument(
        "--phase9-model",
        choices=["gru", "lstm", "transformer"],
        default="transformer",
    )
    parser.add_argument("--track-playstyle", "--phase9", dest="phase9",
                        action="store_true")
    parser.add_argument("--playstyle-panel", action="store_true")
    parser.add_argument("--phase9-window", type=int, default=10)
    parser.add_argument("--phase9-history-path", type=str,
                        default=PHASE9_DEFAULT_HISTORY_PATH)
    parser.add_argument("--no-phase9", action="store_true")
    parser.add_argument("--phase9-debug", dest="phase9_debug",
                        action="store_true", default=False)
    parser.add_argument("--no-phase9-debug", dest="phase9_debug",
                        action="store_false")
    parser.add_argument("--ui-verbose", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    import sys

    args = _parse_cli_args()

    if args.playstyle_panel:
        args.phase9 = True
        args.ui_verbose = True
        args.phase9_debug = False

    if args.phase0 or args.phase1 or args.phase2 or args.phase5:
        if args.phase0:
            run_phase0_check(seed=args.seed)
            print()
        if args.phase1:
            run_phase1_compare(seed=args.seed)
            print()
        if args.phase2:
            run_phase2(num_games=args.phase2_games, seed=args.seed, lookahead_depth=args.phase2_depth)
            print()
        if args.phase5:
            run_phase5_belief_check(
                num_checks=args.phase5_checks,
                num_samples=args.phase5_samples,
                seed=args.seed,
            )
            print()
        sys.exit(0)

    if args.gradio_ui:
        launch_gradio_notebook_ui(default_seed=args.seed, inline=False)
        sys.exit(0)

    if args.phase2_only:
        cfg = HanabiConfig()
        generate_dataset(
            num_games=args.phase2_games,
            seed=args.seed,
            cfg=cfg,
            verbose=True,
        )
        sys.exit(0)

    skip_early_phases = (
        args.phase3_only or args.phase4_only or args.phase6_only
        or args.phase7_only or args.phase8_only or args.phase10_only
        or args.phase9 or args.gradio_ui
    )

    if not skip_early_phases:
        run_phase0_check()
        print()
        run_phase1_compare()
        print()
        if not args.skip_phase2:
            run_phase2()
            print()

    if args.phase4 or args.phase4_only:
        run_phase4_eval(
            num_games=args.phase4_games,
            seed=args.seed,
            debug_belief=args.phase4_debug_belief,
            maximax_samples=args.phase4_samples,
        )
        print()

    if args.phase6 or args.phase6_only:
        run_phase6_compare(
            num_games=args.phase6_games,
            seed=args.seed,
            maximax_samples=args.phase6_samples,
            lookahead_depth=args.phase6_depth,
        )
        print()

    if args.phase7 or args.phase7_only:
        run_phase7(
            num_games=args.phase7_games,
            epochs=args.phase7_epochs,
            batch_size=args.phase7_batch,
            hidden_dim=args.phase7_hidden,
            lr=args.phase7_lr,
            seed=args.seed,
            windowed_playstyle=args.phase7_windowed_playstyle,
            style_window=args.phase7_style_window,
            style_biased_data=args.phase7_style_biased_data,
        )
        print()

    if args.phase8 or args.phase8_only:
        if torch is None:
            print("[Phase 8] skipped: PyTorch is not installed")
        else:
            any_model_present = any(
                _phase8_resolve_path(k, None) is not None
                for k in DEFAULT_ML_MODEL_PATHS.keys()
            )
            if args.phase8_train or not any_model_present:
                if not any_model_present:
                    print(
                        "[Phase 8] no trained models/phase7_*.pt model weights found; "
                        "running Phase 7 first to produce them"
                    )
                run_phase7(
                    num_games=args.phase7_games,
                    epochs=args.phase7_epochs,
                    batch_size=args.phase7_batch,
                    hidden_dim=args.phase7_hidden,
                    lr=args.phase7_lr,
                    seed=args.seed,
                    windowed_playstyle=args.phase7_windowed_playstyle,
                    style_window=args.phase7_style_window,
                    style_biased_data=args.phase7_style_biased_data,
                )
                print()
            _rw = resolve_rollout_weights(args.rollout_config)
            _ht = _cli_hint_score_tune(args)
            run_phase8_compare(
                num_games=args.phase8_games,
                seed=args.seed,
                maximax_samples=args.phase8_samples,
                lookahead_depth=args.phase8_depth,
                ml_weight=args.phase8_weight,
                ml_debug=args.phase8_debug_ml,
                rollout_weights=_rw,
                hint_score_tune=_ht,
                rollout_config_label=str(args.rollout_config).upper(),
            )
            print()

    if args.phase10 or args.phase10_only:
        if torch is None:
            print("[Phase 10] skipped: PyTorch is not installed")
        else:
            any_model_present = any(
                _phase8_resolve_path(k, None) is not None
                for k in DEFAULT_ML_MODEL_PATHS.keys()
            )
            if args.phase8_train or not any_model_present:
                if not any_model_present:
                    print(
                        "[Phase 10] no trained models/phase7_*.pt model weights; "
                        "running Phase 7 first to produce them"
                    )
                run_phase7(
                    num_games=args.phase7_games,
                    epochs=args.phase7_epochs,
                    batch_size=args.phase7_batch,
                    hidden_dim=args.phase7_hidden,
                    lr=args.phase7_lr,
                    seed=args.seed,
                    windowed_playstyle=args.phase7_windowed_playstyle,
                    style_window=args.phase7_style_window,
                    style_biased_data=args.phase7_style_biased_data,
                )
                print()
            _rw10 = resolve_rollout_weights(args.rollout_config)
            _ht10 = _cli_hint_score_tune(args)
            _wlist = _parse_phase10_ml_weights(args.phase10_ml_weights)
            if _wlist is None:
                _wlist = [float(args.phase8_weight)]
            _rc = str(args.rollout_config).upper()
            _all_rows: List[Dict[str, object]] = []
            for _mw in _wlist:
                if len(_wlist) > 1:
                    _tag = str(_mw).replace(".", "p")
                    _plot = f"png/phase10_controller_scores_w{_tag}.png"
                else:
                    _plot = "png/phase10_controller_scores.png"
                _res = run_phase10_compare(
                    num_games=args.phase8_games,
                    seed=args.seed,
                    plot_path=_plot,
                    maximax_samples=args.phase8_samples,
                    lookahead_depth=args.phase8_depth,
                    ml_weight=_mw,
                    ml_debug=args.phase8_debug_ml,
                    rollout_weights=_rw10,
                    hint_score_tune=_ht10,
                    rollout_config_label=_rc,
                )
                _all_rows.extend(
                    build_phase10_5_table_rows(
                        _res,
                        rollout=_rc,
                        ml_weight=_mw,
                        depth=args.phase8_depth,
                        seed=args.seed,
                        num_games=args.phase8_games,
                    )
                )
            if not args.no_phase10_5_table:
                print_phase10_5_table(_all_rows)
            print()

    if args.phase3 or args.phase3_only or args.phase9:
        ai_map = {
            "random":    RandomController,
            "heuristic": HeuristicController,
            "maximax":   MaximaxController,
        }
        ai_factory = None
        ai_cls = None
        if args.ai in ai_map:
            ai_cls = ai_map[args.ai]
        elif args.ai.startswith("maximax_"):
            if torch is None:
                print(
                    f"[Phase 3] --ai {args.ai} needs PyTorch; "
                    "falling back to maximax"
                )
                ai_cls = MaximaxController
            else:
                arch = args.ai.split("_", 1)[1].upper()
                try:
                    wrapper = load_phase8_wrapper(arch, cfg=HanabiConfig())
                    print(f"[Phase 3] loaded {arch} ML guide for AI seats")
                except FileNotFoundError as exc:
                    print(f"[Phase 3] {exc}")
                    print("[Phase 3] falling back to plain maximax")
                    ai_cls = MaximaxController
                else:
                    def _factory(seat_idx: int, cfg_local: HanabiConfig,
                                 _wrapper=wrapper, _arch=arch,
                                 _weight=args.ai_ml_weight):
                        return MaximaxController(
                            f"ml_{_arch.lower()}_{seat_idx}", cfg_local,
                            ml_model=_wrapper,
                            ml_weight=_weight,
                        )
                    ai_factory = _factory
        else:
            ai_cls = MaximaxController

        playstyle_tracker = None
        if not args.no_phase9:
            playstyle_tracker = build_phase9_tracker(
                model_type=args.phase9_model,
                human_agent=f"player_{args.human_seat}",
                cfg=HanabiConfig(),
                style_window=args.phase9_window,
                save_path=args.phase9_history_path,
                verbose=not args.ui_verbose,
                debug=args.phase9_debug,
            )

        display_ml = None
        display_label = ""
        if args.ui_verbose and torch is not None:
            if playstyle_tracker is not None:
                display_ml = playstyle_tracker.wrapper
                display_label = str(getattr(
                    playstyle_tracker, "model_tag", "transformer",
                )).upper()
            else:
                try:
                    display_ml = load_phase8_wrapper(
                        args.phase9_model, cfg=HanabiConfig(),
                    )
                    display_label = str(args.phase9_model).upper()
                except Exception as exc:
                    print(
                        f"[Phase 11] could not load classifier for side panel: {exc}"
                    )

        play_phase3_game(
            seed=args.seed,
            human_seat=args.human_seat,
            ai_delay_ms=args.ai_delay_ms,
            ai_controller_cls=ai_cls,
            ai_controller_factory=ai_factory,
            playstyle_tracker=playstyle_tracker,
            ui_verbose=args.ui_verbose,
            display_ml_wrapper=display_ml,
            display_model_label=display_label,
        )
