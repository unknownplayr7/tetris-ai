"""Train a Tetris agent with batched actor-critic reinforcement learning."""

import argparse
import os
import time

import numpy as np
import torch
from torch.distributions import Categorical

from model import TetrisActorCritic
from tetris_env import TetrisEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--envs", type=int, default=128)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--checkpoint", default="checkpoints/tetris_rl.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    envs = [TetrisEnv(seed=i) for i in range(args.envs)]
    obs = np.stack([env.reset() for env in envs])
    model = TetrisActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)

    started = time.time()
    for update in range(1, args.updates + 1):
        log_probs = []
        values = []
        rewards = []
        entropies = []
        episode_lines = 0
        finished = 0

        for _ in range(args.rollout):
            masks = np.stack([env.valid_action_mask() for env in envs])
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            mask_t = torch.tensor(masks, dtype=torch.bool, device=device)
            logits, value = model.masked_logits(obs_t, mask_t)
            dist = Categorical(logits=logits)
            actions = dist.sample()

            next_obs = []
            reward_row = []
            done_row = []
            for env, action in zip(envs, actions.cpu().numpy()):
                new_obs, reward, done, info = env.step(int(action))
                episode_lines += info["lines"]
                if done:
                    finished += 1
                    new_obs = env.reset()
                next_obs.append(new_obs)
                reward_row.append(reward)
                done_row.append(done)

            log_probs.append(dist.log_prob(actions))
            values.append(value)
            rewards.append(torch.tensor(reward_row, dtype=torch.float32, device=device))
            entropies.append(dist.entropy())
            obs = np.stack(next_obs)

        with torch.no_grad():
            masks = np.stack([env.valid_action_mask() for env in envs])
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            mask_t = torch.tensor(masks, dtype=torch.bool, device=device)
            _, bootstrap_value = model.masked_logits(obs_t, mask_t)

        returns = []
        running = bootstrap_value
        for reward in reversed(rewards):
            running = reward + args.gamma * running
            returns.append(running)
        returns.reverse()

        returns_t = torch.stack(returns)
        values_t = torch.stack(values)
        log_probs_t = torch.stack(log_probs)
        advantages = returns_t - values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(log_probs_t * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        entropy_bonus = torch.stack(entropies).mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if update % 10 == 0 or update == 1:
            elapsed = time.time() - started
            print(
                f"update={update:5d} loss={loss.item():8.4f} "
                f"reward={returns_t[0].mean().item():7.3f} "
                f"lines={episode_lines:5d} finished={finished:4d} "
                f"elapsed={elapsed:6.1f}s"
            )

        if update % 100 == 0 or update == args.updates:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "update": update,
                    "args": vars(args),
                },
                args.checkpoint,
            )
            print(f"Saved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
