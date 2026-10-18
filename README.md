# Tetris AI

A Tetris project with a playable Pygame game, a fast heuristic agent, and an optional
PyTorch reinforcement-learning agent.

## Play the game

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install pygame numpy
python main.py                   # Watch the heuristic AI play
python main.py --mode human      # Play yourself
```

### Controls

| Key | Action |
| --- | --- |
| Left / Right | Move piece |
| Down | Soft drop |
| Up or X | Rotate clockwise |
| Z | Rotate counter-clockwise |
| Space | Hard drop in human mode; pause/resume in AI mode |
| P | Pause/resume |
| Tab | Switch between human and AI modes |
| R | Restart |
| + / - | Change game speed |
| Esc | Quit |

The game includes a ghost landing guide in human mode, a next-piece preview, score/line/level
stats, a seven-bag piece randomizer, and animated AI placements. The heuristic agent uses a
one-piece lookahead to consider the next tetromino before committing to a move.

## Project structure

- `main.py` - Pygame game UI, human controls, and AI autoplay
- `board.py` - Board simulation and line clearing
- `pieces.py` - Tetromino shapes and rotations
- `ai.py` - Heuristic evaluation and placement search
- `tetris_env.py` - Placement-based reinforcement-learning environment
- `model.py` - PyTorch actor-critic neural network
- `train.py` - Batched actor-critic training loop
- `play_model.py` - Notebook-friendly viewer for a trained RL model
- `colab_demo.py` - Notebook-friendly heuristic AI visualization

## Optional reinforcement-learning agent

Install the full dependencies:

```bash
python -m pip install -r requirements.txt
```

Check PyTorch device support:

```python
import torch
print("CUDA:", torch.cuda.is_available())
print("MPS:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
```

Train the actor-critic model:

```bash
python train.py --updates 1000 --envs 128 --rollout 128
```

Training automatically chooses CUDA, Apple Silicon MPS when available, or CPU. Checkpoints
are saved to `checkpoints/tetris_rl.pt`.

The trainer runs many independent environments in parallel at the Python level, batches the
network work, masks illegal placements, and stops return bootstrapping at episode boundaries.
The heuristic agent is ready to run immediately; the reinforcement-learning model must be
trained before it can be used.

For a trained model in a notebook environment:

```bash
python play_model.py --checkpoint checkpoints/tetris_rl.pt
```

## Tests

Run the dependency-light tests with:

```bash
python -m unittest discover -s tests -v
```

The CI workflow also checks Python syntax and runs the test suite.
