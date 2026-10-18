# Tetris AI

A Tetris project with a playable Pygame game and a reinforcement-learning lab. The interactive game keeps a heuristic baseline; the RL agent is trained separately from scratch with PPO.

## Play Tetris

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install pygame numpy
python main.py                   # Watch the heuristic baseline
python main.py --mode human      # Play yourself
```

### Controls

| Key | Action |
| --- | --- |
| Left / Right | Move piece |
| Down | Soft drop |
| Up or X | Rotate clockwise |
| Z | Rotate counter-clockwise |
| Space | Hard drop in human mode |
| P | Pause/resume |
| Tab | Switch modes |
| R | Restart |
| + / - | Change speed |
| Esc | Quit |

The game includes a ghost landing guide, next-piece preview, score/line/level stats, seven-bag randomizer, and animated heuristic autoplay.

## RL lab: train from scratch

The RL system is separate from the heuristic AI. Training uses a PyTorch actor-critic policy optimized with **Proximal Policy Optimization (PPO)**, generalized advantage estimation (GAE), legal-action masking, and reward/optimization telemetry.

### Technical dashboard (recommended)

Install the full dependencies and start the local dashboard:

```bash
python -m pip install -r requirements.txt
streamlit run dashboard.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Open it in your browser.

The dashboard lets you:
- Start a fresh run with randomly initialized weights.
- Choose update count, parallel environments, rollout length, PPO epochs, minibatch size, learning rate, seed, and compute device.
- Explicitly opt into resuming a checkpoint. Resume is **off by default**.
- Replay board snapshots of real sampled training actions, including the chosen piece, rotation, column, reward, legal placements, and resulting board.
- Monitor reward/return, lines cleared, pieces survived, environment steps per second, policy/value losses, entropy, approximate KL divergence, and PPO clip fraction.
- Read a plain-English glossary explaining PPO, GAE, reward shaping, and optimization metrics.
- Inspect recent metric rows and trainer logs, and download the latest 5,000 metric records.
- Stop the training process from the sidebar.

Runs are stored in `runs/ppo-<timestamp>/`, with `metrics.jsonl`, `moves.jsonl`, `moves.idx`, `trainer.log`, and `checkpoint.pt`. The move replay records **every placement from training environment 1**, including the before/after board, chosen piece, rotation, column, landing row, reward, lines, legal-action count, episode ID, and game-over state. Use Previous/Next, the move slider, or autoplay to inspect the sequence. The other parallel environments still train the policy but are not duplicated into the visual log, keeping storage manageable. The fixed-width `moves.idx` file lets the dashboard load a selected move without loading every board into memory. Checkpoints contain model and optimizer state, update count, and environment-step count.

### Train from Terminal instead

A modest starting configuration for an M3 Mac is:

```bash
python train.py --updates 1000 --envs 8 --rollout 128 --device auto
```

For a small smoke test before a long run:

```bash
python train.py --updates 2 --envs 2 --rollout 16 --epochs 1 --minibatch-size 32
```

The default device selection prefers CUDA when available, then Apple Silicon MPS, then CPU. You can override it with `--device mps`, `--device cpu`, or `--device cuda`.

Each terminal run creates a new timestamped run directory under `runs/` unless `--run-dir` is supplied. To resume intentionally:

```bash
python train.py --updates 500 --resume runs/ppo-YYYYMMDD-HHMMSS/checkpoint.pt
```

Replace that example path with the actual checkpoint path. If you do not pass `--resume`, a fresh randomly initialized model is created.

## Project structure

- `main.py` - Pygame game UI and heuristic autoplay
- `board.py`, `pieces.py` - board simulation and tetromino definitions
- `ai.py` - heuristic baseline
- `tetris_env.py` - placement-based RL environment and legal-action masks
- `model.py` - actor-critic policy/value network
- `train.py` - fresh-start PPO trainer, GAE, metrics, and checkpointing
- `dashboard.py` - local Streamlit training dashboard
- `play_model.py` - simple notebook-oriented model viewer

## Tests

```bash
python -m unittest discover -s tests -v
python -m py_compile train.py dashboard.py model.py tetris_env.py
```

A trained checkpoint is not included by default. Training is compute-intensive and improvement is not guaranteed; compare multiple seeds and evaluate policies on held-out games before treating a result as better than the heuristic baseline.


## Local data and security notes

- Training logs, move replays, and checkpoints stay in the local `runs/` directory unless you manually share them.
- The dashboard launches the trainer with a subprocess argument list rather than shell command text. It does not need API keys or network access to train.
- Only load model checkpoints you trust. The trainer and viewer use PyTorch's restricted `weights_only=True` checkpoint loader.
- Local virtual environments, run outputs, checkpoints, and environment files are excluded by `.gitignore` so they are less likely to be committed accidentally.
- The source review found no obvious hardcoded API keys, access tokens, private-key blocks, or outbound network calls in the current tracked source. This is a source-tree review, not a guarantee about the entire Git history or GitHub account; rotate any credential that may have been committed in the past.
