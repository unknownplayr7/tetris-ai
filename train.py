"""Fresh-start PPO trainer for the placement-based Tetris environment.

Each run writes append-only JSONL metrics and resumable checkpoints. A new run
starts with randomly initialized weights unless --resume is explicitly supplied.
"""
import argparse
import json
import os
import random
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from board import WIDTH, drop_y
from pieces import rotations

from model import TetrisActorCritic
from tetris_env import TetrisEnv


def choose_device(requested):
    if requested != "auto":
        if requested == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was requested but is not available on this Mac.")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--run-dir", default=None, help="Output directory; created if missing.")
    parser.add_argument("--resume", default=None, help="Resume model/optimizer from a checkpoint.")
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()
    if min(args.updates, args.envs, args.rollout, args.epochs, args.minibatch_size) < 1:
        parser.error("updates, envs, rollout, epochs, and minibatch-size must be positive")
    if args.save_every < 1:
        parser.error("save-every must be positive")
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = choose_device(args.device)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.run_dir or Path("runs") / f"ppo-{stamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    checkpoint_path = run_dir / "checkpoint.pt"
    print(f"Device: {device} | run: {run_dir}", flush=True)
    print("Mode: resume" if args.resume else "Mode: fresh random initialization", flush=True)

    envs = [TetrisEnv(seed=args.seed + i) for i in range(args.envs)]
    obs = np.stack([env.reset() for env in envs]).astype(np.float32)
    model = TetrisActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    start_update = 1
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model_state"])
        if "optimizer_state" in state:
            optimizer.load_state_dict(state["optimizer_state"])
        start_update = int(state.get("update", 0)) + 1
        print(f"Resumed checkpoint at update {start_update - 1}", flush=True)

    episode_returns = np.zeros(args.envs, dtype=np.float64)
    episode_lines = np.zeros(args.envs, dtype=np.int64)
    episode_pieces = np.zeros(args.envs, dtype=np.int64)
    recent_returns = deque(maxlen=100)
    recent_lines = deque(maxlen=100)
    recent_pieces = deque(maxlen=100)
    env_steps = int(state.get("env_steps", 0)) if args.resume else 0
    run_started = time.time()
    metrics_file = metrics_path.open("a" if args.resume else "w", encoding="utf-8")

    try:
        for update in range(start_update, start_update + args.updates):
            update_started = time.time()
            obs_buf, mask_buf, action_buf = [], [], []
            logprob_buf, reward_buf, done_buf, value_buf = [], [], [], []
            completed_returns, completed_lines, completed_pieces = [], [], []
            entropy_samples = []
            viz_snapshot = {}

            for rollout_step in range(args.rollout):
                masks = np.stack([env.valid_action_mask() for env in envs])
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                mask_t = torch.as_tensor(masks, dtype=torch.bool, device=device)
                with torch.no_grad():
                    logits, values = model.masked_logits(obs_t, mask_t)
                    dist = Categorical(logits=logits)
                    actions = dist.sample()
                    logprobs = dist.log_prob(actions)
                    entropy_samples.append(float(dist.entropy().mean().item()))

                action_np = actions.cpu().numpy()
                next_obs, rewards, dones = [], [], []
                for i, (env, action_value) in enumerate(zip(envs, action_np)):
                    action = int(action_value)
                    capture_viz = rollout_step == args.rollout - 1 and i == 0
                    if capture_viz:
                        piece_name = env.piece_name
                        board_before = [row[:] for row in env.board]
                        rotation_index = action // WIDTH
                        x_position = action % WIDTH
                        rotated_shape = rotations(env.shape)[rotation_index]
                        landing_y = drop_y(env.board, rotated_shape, x_position)
                        valid_action_count = int(masks[i].sum())

                    new_obs, reward, done, info = env.step(action)
                    if capture_viz:
                        viz_snapshot = {
                            "viz_board_before": board_before,
                            "viz_board": [row[:] for row in env.board],
                            "viz_piece": piece_name,
                            "viz_shape": rotated_shape,
                            "viz_rotation": rotation_index,
                            "viz_x": x_position,
                            "viz_y": landing_y,
                            "viz_action": action,
                            "viz_valid_actions": valid_action_count,
                            "viz_reward": float(reward),
                            "viz_lines": int(info.get("lines", 0)),
                            "viz_done": bool(done),
                        }
                    episode_returns[i] += reward
                    episode_lines[i] += int(info.get("lines", 0))
                    episode_pieces[i] += 1
                    if done:
                        completed_returns.append(float(episode_returns[i]))
                        completed_lines.append(int(episode_lines[i]))
                        completed_pieces.append(int(episode_pieces[i]))
                        recent_returns.append(float(episode_returns[i]))
                        recent_lines.append(int(episode_lines[i]))
                        recent_pieces.append(int(episode_pieces[i]))
                        episode_returns[i] = 0.0
                        episode_lines[i] = 0
                        episode_pieces[i] = 0
                        new_obs = env.reset()
                    next_obs.append(new_obs)
                    rewards.append(float(reward))
                    dones.append(float(done))

                obs_buf.append(obs_t)
                mask_buf.append(mask_t)
                action_buf.append(actions)
                logprob_buf.append(logprobs)
                value_buf.append(values)
                reward_buf.append(torch.tensor(rewards, dtype=torch.float32, device=device))
                done_buf.append(torch.tensor(dones, dtype=torch.float32, device=device))
                obs = np.stack(next_obs).astype(np.float32)
                env_steps += args.envs

            with torch.no_grad():
                last_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
                last_masks = torch.as_tensor(
                    np.stack([env.valid_action_mask() for env in envs]),
                    dtype=torch.bool, device=device,
                )
                _, next_value = model.masked_logits(last_obs, last_masks)

            # Generalized Advantage Estimation; done masks prevent cross-game leakage.
            advantages = torch.zeros_like(torch.stack(reward_buf))
            last_gae = torch.zeros(args.envs, dtype=torch.float32, device=device)
            rewards_t = torch.stack(reward_buf)
            dones_t = torch.stack(done_buf)
            values_t = torch.stack(value_buf)
            for t in reversed(range(args.rollout)):
                next_nonterminal = 1.0 - dones_t[t]
                next_values = next_value if t == args.rollout - 1 else values_t[t + 1]
                delta = rewards_t[t] + args.gamma * next_values * next_nonterminal - values_t[t]
                last_gae = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
                advantages[t] = last_gae
            returns_t = advantages + values_t

            b_obs = torch.stack(obs_buf).reshape(-1, obs_buf[0].shape[-1])
            b_masks = torch.stack(mask_buf).reshape(-1, mask_buf[0].shape[-1])
            b_actions = torch.stack(action_buf).reshape(-1)
            b_logprobs = torch.stack(logprob_buf).reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns_t.reshape(-1)
            batch_size = b_actions.shape[0]
            minibatch_size = min(args.minibatch_size, batch_size)
            policy_losses, value_losses, entropies, approx_kls, clipfracs = [], [], [], [], []

            for _epoch in range(args.epochs):
                indices = torch.randperm(batch_size, device=device)
                for start in range(0, batch_size, minibatch_size):
                    mb = indices[start:start + minibatch_size]
                    logits, new_values = model.masked_logits(b_obs[mb], b_masks[mb])
                    dist = Categorical(logits=logits)
                    new_logprobs = dist.log_prob(b_actions[mb])
                    logratio = new_logprobs - b_logprobs[mb]
                    ratio = logratio.exp()
                    mb_adv = b_advantages[mb]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std(unbiased=False) + 1e-8)
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(
                        ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef
                    )
                    policy_loss = torch.maximum(pg_loss1, pg_loss2).mean()
                    value_loss = 0.5 * (new_values - b_returns[mb]).pow(2).mean()
                    entropy = dist.entropy().mean()
                    loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - logratio).mean()
                        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()
                    policy_losses.append(float(policy_loss.item()))
                    value_losses.append(float(value_loss.item()))
                    entropies.append(float(entropy.item()))
                    approx_kls.append(float(approx_kl.item()))
                    clipfracs.append(float(clipfrac.item()))

            elapsed = time.time() - run_started
            recent_window = list(recent_returns)
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "update": update,
                "env_steps": env_steps,
                "sps": int(env_steps / max(elapsed, 1e-6)),
                "update_seconds": round(time.time() - update_started, 4),
                "wall_seconds": round(elapsed, 2),
                "device": str(device),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "mean_reward": float(rewards_t.mean().item()),
                "mean_episode_return_100": float(np.mean(recent_window)) if recent_window else None,
                "best_episode_return_100": float(max(recent_window)) if recent_window else None,
                "mean_lines_100": float(np.mean(recent_lines)) if recent_lines else None,
                "mean_pieces_100": float(np.mean(recent_pieces)) if recent_pieces else None,
                "completed_episodes": len(completed_returns),
                "total_episodes": len(recent_returns) if update == start_update else None,
                "episodes_this_update": len(completed_returns),
                "mean_completed_return": float(np.mean(completed_returns)) if completed_returns else None,
                "mean_completed_lines": float(np.mean(completed_lines)) if completed_lines else None,
                "mean_completed_pieces": float(np.mean(completed_pieces)) if completed_pieces else None,
                "policy_loss": float(np.mean(policy_losses)),
                "value_loss": float(np.mean(value_losses)),
                "entropy": float(np.mean(entropies)),
                "sample_entropy": float(np.mean(entropy_samples)),
                "approx_kl": float(np.mean(approx_kls)),
                "clip_fraction": float(np.mean(clipfracs)),
                "advantage_mean": float(b_advantages.mean().item()),
                "return_mean": float(b_returns.mean().item()),
                **viz_snapshot,
            }
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()

            if update == start_update or update % 10 == 0:
                avg_return = row["mean_episode_return_100"]
                print(
                    f"update={update} steps={env_steps} sps={row['sps']} "
                    f"reward={row['mean_reward']:.3f} "
                    f"episode_return_100={avg_return if avg_return is not None else 'n/a'} "
                    f"policy_loss={row['policy_loss']:.4f} value_loss={row['value_loss']:.4f}",
                    flush=True,
                )

            if update % args.save_every == 0 or update == start_update + args.updates - 1:
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "update": update,
                    "env_steps": env_steps,
                    "args": vars(args),
                    "seed": args.seed,
                }, checkpoint_path)
    finally:
        metrics_file.close()
    print(f"Training finished. Metrics: {metrics_path} | checkpoint: {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
