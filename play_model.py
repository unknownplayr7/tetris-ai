"""Render a trained reinforcement-learning Tetris model in Colab."""

import argparse
import time

import torch
from IPython.display import clear_output

from model import TetrisActorCritic
from tetris_env import TetrisEnv, WIDTH


def render(board, piece_name, steps, lines):
    clear_output(wait=True)
    print("TETRIS RL AGENT")
    print("+" + "--" * WIDTH + "+")
    for row in board:
        print("|" + "".join("██" if cell else "  " for cell in row) + "|")
    print("+" + "--" * WIDTH + "+")
    print(f"Piece: {piece_name} | Moves: {steps} | Lines: {lines}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/tetris_rl.pt")
    parser.add_argument("--delay", type=float, default=0.08)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = TetrisActorCritic().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    env = TetrisEnv()
    obs = env.reset()
    lines_total = 0

    while True:
        mask = env.valid_action_mask()
        obs_t = torch.tensor(obs[None], dtype=torch.float32, device=device)
        mask_t = torch.tensor(mask[None], dtype=torch.bool, device=device)
        with torch.no_grad():
            logits, _ = model.masked_logits(obs_t, mask_t)
            action = int(logits.argmax(dim=1).item())

        obs, _, done, info = env.step(action)
        lines_total += info["lines"]
        render(env.board, env.piece_name, env.steps, lines_total)
        time.sleep(args.delay)

        if done:
            print("GAME OVER")
            break


if __name__ == "__main__":
    main()
