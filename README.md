# Tetris AI

A Tetris project that now contains two AIs:

1. A fast hand-written heuristic player.
2. A GPU-aware reinforcement-learning agent trained with PyTorch.

## Project structure

- `main.py` - Pygame heuristic AI visualization
- `colab_demo.py` - Notebook-friendly heuristic AI visualization
- `board.py` - Board simulation and line clearing
- `pieces.py` - Tetromino shapes and rotations
- `ai.py` - Original heuristic evaluation and move selection
- `tetris_env.py` - Placement-based reinforcement-learning environment
- `model.py` - PyTorch actor-critic neural network
- `train.py` - Batched GPU-aware RL training loop
- `play_model.py` - Render a trained RL agent inside Colab

## Colab quick start

Clone the repository:

```python
!git clone https://github.com/unknownplayr7/tetris-ai.git
%cd tetris-ai
!pip install -r requirements.txt
```

Check whether a GPU is available:

```python
import torch
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

Train with a reasonably large batch of simultaneous games:

```python
!python train.py --updates 1000 --envs 128 --rollout 128
```

The trainer automatically uses CUDA when available and otherwise falls back to CPU. Checkpoints are saved to `checkpoints/tetris_rl.pt`.

Watch the trained agent:

```python
!python play_model.py --checkpoint checkpoints/tetris_rl.pt
```

## How the RL agent works

The network receives a flattened 20x10 board plus a one-hot encoding of the current piece. It chooses a complete placement action: one of four rotation slots and ten horizontal positions. Illegal actions are masked out before sampling.

Each training update runs many independent games, collects a rollout from each environment, then performs an actor-critic update. The GPU accelerates neural-network inference and learning across the batch; the small Python game environments remain CPU-side.

## Reward

The agent receives reward for surviving and clearing lines, with penalties for holes, excessive stack height, and invalid placements. This is deliberately shaped to make early learning less random and less dependent on waiting for the agent to accidentally discover that a clean board is useful.

## Tuning for Colab

If the GPU is underutilized, increase `--envs` first:

```python
!python train.py --updates 1000 --envs 256 --rollout 128
```

If Colab runs out of RAM, reduce `--envs`. Larger models and a fully vectorized environment would be the next performance step.
