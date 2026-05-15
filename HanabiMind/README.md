# Hanabi AI Project

This is a course project and is a full AI pipeline built around the cooperative card game Hanabi. The goal is to build agents that can play the game well and adapt to how human players behave over time. It covers everything from basic game mechanics and search-based controllers all the way through ML training, playstyle detection, and an interactive UI where you can play against the AI yourself.

## What is Hanabi

Hanabi is a cooperative card game where players work together to play cards in order, but with a twist: you can see everyone else's hand but not your own. Communication is limited to giving hints about colors or ranks. A perfect game scores 25 points. A lot of what makes it interesting from an AI perspective is the incomplete information and the need to reason about what your teammates know.

## Project structure

The project is split across numbered phases, each building on the last. Here is what each file handles:

**P0_P1_project_config.py** - Shared configuration dataclass with game settings like number of colors, ranks, players, and tokens.

**P0_observation_decoding.py** - Parses the raw observation vectors coming out of the PettingZoo environment into structured Python objects. Also handles action encoding and decoding, plus a consistency checker that validates observation invariants during self-play.

**P0_P1_environment.py** - Wraps the PettingZoo hanabi_v5 environment and provides a clean episode-running interface that controllers and training code can use.

**P1_game_state_simulation.py** - A deterministic game state class that can simulate moves forward. Used by the Maximax controller to score potential actions. Also includes utility functions for computing which cards are still alive and a heuristic evaluation function.

**P2_P7_trajectory_schema.py** - Shared data types for recording game trajectories, including the per-step schema and controller label constants.

**P1_P5_foundation_checks.py** - Quick sanity checks: runs random vs Maximax comparisons for Phase 1, and verifies the belief sampler in Phase 5.

**P5_belief_sampling.py** - Samples plausible own-hand assignments that are consistent with both the hint history and the global card counts. Used by the Maximax controller to average action scores over multiple hypothetical worlds rather than treating each slot independently.

**P6_rollout_policy.py** - A knowledge-aware rollout policy that simulates future turns for the deeper lookahead search. Players only play cards they are guaranteed to know are playable, and hints are chosen based on whether they unlock a new certain play for a partner on the next turn.

**P1_P5_P6_P8_P10_controllers.py** - All controller implementations. Includes RandomController, HeuristicController, MaximaxController (the main search agent), and PlaystyleBiasedController (used to generate training data with visible playstyle patterns). Also contains the rollout evaluation function and playstyle alignment scoring.

**P2_P7_ml_training.py** - Dataset generation, feature extraction, and training for three sequence classifiers (GRU, LSTM, Transformer) that learn to identify which controller style produced a given game trajectory.

**P4_P6_controller_evaluation.py** - Evaluation harnesses that run batches of games and plot score distributions for different controller configurations.

**P8_P9_P10_adaptive_system.py** - Loads trained models and feeds them back into Maximax as an action prior. Also implements the live playstyle tracker that watches the human player during a game and estimates whether they are playing chaotically, cooperatively, or strategically.

**P3_P9_P11_play_loop.py** - The main game loop for the interactive human vs AI mode. Connects the pygame UI, the AI controllers, trajectory logging, and the live playstyle tracker.

**P3_P9_P11_pygame_ui.py** - The pygame rendering code.

**hanabi_project_runner.py** - Command-line entry point. Run this with various flags to execute any phase of the pipeline.

## Setup

Install dependencies first:

```
pip install -r requirements.txt
```

If your system does not have Python installed yet, on Ubuntu/Debian/WSL run:

```
sudo apt update && sudo apt install python3 python3-pip python3-venv -y
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the project in order

Run these one at a time from the project folder:

```
python hanabi_project_runner.py --validate-decoding
python hanabi_project_runner.py --compare-baselines
python hanabi_project_runner.py --train-gru-behavior-clone --phase2-games 80 --phase2-depth 7
python hanabi_project_runner.py --compare-controllers --phase4-games 50
python hanabi_project_runner.py --validate-beliefs
python hanabi_project_runner.py --compare-rollouts --phase6-depth 7 --phase6-games 20
python hanabi_project_runner.py --train-sequence-models --phase7-games 80 --phase7-epochs 20 --phase7-windowed-playstyle --phase7-style-biased-data
python hanabi_project_runner.py --compare-ml-guided-search --phase8-train --phase8-games 30 --phase8-depth 7 --phase8-weight 0.15 --phase9-model transformer --rollout-config B
python hanabi_project_runner.py --compare-integrated-controller --phase8-games 30 --phase8-depth 7 --phase8-weight 0.15 --phase9-model transformer --rollout-config B
python hanabi_project_runner.py --play-ui --playstyle-panel --ai maximax_transformer --phase9-model transformer --ai-ml-weight 0.15
```

## Lookahead depth

The Maximax controller supports arbitrary lookahead depth via the `--phase6-depth` and `--phase8-depth` flags. Depth 1 is one-step lookahead (the default). Depth 2 adds one full round of rollout turns after the root action. Depth 6 and depth 7 are fully supported and will run the rollout policy for 5 or 6 additional turns.

At higher depths the controller automatically scales down the number of belief samples per decision to keep things tractable. Specifically, for depth 4 and above, the sample count is halved for each additional depth step beyond 3.

The `--phase2-depth` flag controls what depth the training dataset is generated at. For best results, this should match `--phase8-depth` so the models learn from trajectories that look like the controller they will guide.

```
python hanabi_project_runner.py --train-gru-behavior-clone --phase2-games 80 --phase2-depth 7
```

## Playing against the AI

```
python hanabi_project_runner.py --play-ui --playstyle-panel --ai maximax_transformer --phase9-model transformer
```

You sit in seat 0 by default. Use `--human-seat 2` to change your position. The `--ai` flag accepts `maximax`, `heuristic`, `random`, `maximax_gru`, `maximax_lstm`, and `maximax_transformer`. Add `--playstyle-panel` to get the live playstyle panel on the right side of the screen.

## Training the ML models

```
python hanabi_project_runner.py --train-sequence-models --phase7-games 80 --phase7-epochs 20 --phase7-windowed-playstyle
```

All three model architectures (GRU, LSTM, Transformer) are trained in one run. Weights are saved to the `trained models/` directory.

## Notes

The project requires PettingZoo with the classic games extras plus OpenSpiel through Shimmy. PyTorch is only needed for Phases 2 and later. All phases from 0 through 1 will run without it.

Results and plots get written to the `png/` directory. Training history and dataset caches are saved as pickle files in the project root.
