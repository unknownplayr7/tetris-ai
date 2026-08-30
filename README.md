# Tetris AI

A simple Tetris-playing AI written in Python.

The AI evaluates every possible rotation and horizontal placement for each tetromino, then chooses the move with the best heuristic score.

## What it evaluates

- Lines cleared
- Aggregate column height
- Holes
- Surface bumpiness
- Maximum height

## Installation

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Press `R` to restart after a game over.

## Project structure

- `main.py` - Pygame visualization and game loop
- `board.py` - Board simulation and line clearing
- `pieces.py` - Tetromino shapes and rotations
- `ai.py` - Heuristic evaluation and move selection

## Next upgrades

- Add a preview of the current and next piece
- Add two-piece look-ahead
- Use the modern 7-bag piece generator
- Automatically tune heuristic weights
- Train a reinforcement-learning agent
