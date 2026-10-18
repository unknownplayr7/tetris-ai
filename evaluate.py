"""Reproducible benchmark for random, heuristic, and trained Tetris agents.

All policies receive the same seeded seven-bag piece sequences. Evaluation never
updates model weights. JSON output includes per-game results and approximate
95% confidence intervals so a single lucky game is not mistaken for progress.
"""
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ai import find_best_move
from board import WIDTH
from model import TetrisActorCritic
from tetris_env import TetrisEnv


POLICIES = ("random", "heuristic", "heuristic-lookahead", "ppo")


def choose_device(requested):
    if requested != "auto":
        if requested == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was requested but is not available.")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_policy(checkpoint_path, device):
    """Load a trusted checkpoint using PyTorch's restricted weights-only loader."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("Checkpoint must be a training checkpoint with 'model_state'.")
    model = TetrisActorCritic().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def confidence_interval(values):
    """Return mean, sample standard deviation, and a normal-approximation 95% CI."""
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot summarize an empty sample.")
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    half_width = 1.96 * std / float(np.sqrt(values.size))
    return {
        "mean": mean,
        "std": std,
        "ci95": [mean - half_width, mean + half_width],
    }


def select_action(policy_name, env, rng, model=None, device=None):
    mask = env.valid_action_mask()
    legal_actions = np.flatnonzero(mask)
    if legal_actions.size == 0:
        return None

    if policy_name == "random":
        return int(rng.choice(legal_actions.tolist()))

    if policy_name in ("heuristic", "heuristic-lookahead"):
        next_shape = env.peek_next_shape() if policy_name == "heuristic-lookahead" else None
        move = find_best_move(env.board, env.shape, next_shape=next_shape)
        if move is None:
            return None
        rotation_index, _rotated, x_position, _result, _lines = move
        return int(rotation_index * WIDTH + x_position)

    if policy_name == "ppo":
        if model is None:
            raise ValueError("The PPO policy requires --checkpoint.")
        observation = torch.as_tensor(
            env.observation()[None], dtype=torch.float32, device=device
        )
        action_mask = torch.as_tensor(mask[None], dtype=torch.bool, device=device)
        with torch.no_grad():
            logits, _value = model.masked_logits(observation, action_mask)
            return int(logits.argmax(dim=1).item())

    raise ValueError(f"Unknown policy: {policy_name}")


def evaluate_policy(policy_name, seeds, model=None, device=None, max_pieces=5000):
    """Play one episode for each seed and return individual, reproducible results."""
    results = []
    for seed in seeds:
        env = TetrisEnv(seed=int(seed))
        env.reset()
        rng = random.Random(int(seed) ^ 0x5EED5EED)
        total_reward = 0.0
        total_lines = 0
        pieces = 0
        game_over = False

        while pieces < max_pieces:
            action = select_action(policy_name, env, rng, model=model, device=device)
            if action is None:
                game_over = True
                break
            _observation, reward, done, info = env.step(action)
            total_reward += float(reward)
            total_lines += int(info.get("lines", 0))
            pieces += 1
            if done:
                game_over = True
                break

        results.append({
            "seed": int(seed),
            "pieces": pieces,
            "lines": total_lines,
            "return": total_reward,
            "game_over": game_over,
            "truncated": not game_over and pieces >= max_pieces,
        })
    return results


def summarize_policy(policy_name, episodes):
    return {
        "policy": policy_name,
        "episodes": len(episodes),
        "pieces": confidence_interval(row["pieces"] for row in episodes),
        "lines": confidence_interval(row["lines"] for row in episodes),
        "return": confidence_interval(row["return"] for row in episodes),
        "completed_games": sum(row["game_over"] for row in episodes),
        "truncated_games": sum(row["truncated"] for row in episodes),
        "per_episode": episodes,
    }


def run_evaluation(episodes=20, seed=1000, checkpoint=None, policies=None,
                   device_name="auto", max_pieces=5000):
    if episodes < 1:
        raise ValueError("episodes must be at least 1.")
    if max_pieces < 1:
        raise ValueError("max_pieces must be at least 1.")

    if policies is None:
        policies = ["random", "heuristic", "heuristic-lookahead"] + (["ppo"] if checkpoint else [])
    policies = list(dict.fromkeys(policies))
    if not policies:
        raise ValueError("Choose at least one policy.")
    if any(policy not in POLICIES for policy in policies):
        raise ValueError("Unknown policy requested.")
    if "ppo" in policies and not checkpoint:
        raise ValueError("The PPO policy requires --checkpoint.")

    device = choose_device(device_name)
    model = load_policy(checkpoint, device) if "ppo" in policies else None
    seeds = [int(seed) + index for index in range(episodes)]
    results = [
        summarize_policy(
            policy,
            evaluate_policy(policy, seeds, model=model, device=device, max_pieces=max_pieces),
        )
        for policy in policies
    ]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "episodes": episodes,
            "base_seed": seed,
            "seeds": seeds,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "device": str(device),
            "max_pieces_per_episode": max_pieces,
            "piece_randomizer": "seven-bag",
        },
        "results": results,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000,
                        help="Base seed; every policy gets the same episode seeds.")
    parser.add_argument("--checkpoint", help="Optional PPO checkpoint to compare.")
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--max-pieces", type=int, default=5000,
                        help="Safety cap per game; capped games are marked truncated.")
    parser.add_argument("--output", help="Optional path to save the JSON report.")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        report = run_evaluation(
            episodes=args.episodes,
            seed=args.seed,
            checkpoint=args.checkpoint,
            policies=args.policies,
            device_name=args.device,
            max_pieces=args.max_pieces,
        )
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        raise SystemExit(f"Evaluation failed: {exc}") from exc

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved evaluation report to {output_path}")
    print(rendered)


if __name__ == "__main__":
    main()
