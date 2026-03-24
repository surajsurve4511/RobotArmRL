#!/usr/bin/env python3
"""
Evaluate a trained MCAD arm RL policy.

Usage:
    python scripts/evaluate.py --model checkpoints/sac_*/final_model.zip
    python scripts/evaluate.py --model checkpoints/sac_*/best/best_model.zip --episodes 50
"""
import os
import sys
import argparse
import numpy as np

from stable_baselines3 import SAC, PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcad_arm_rl.env import ReachEnv


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained MCAD arm agent")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to saved model (.zip file)",
    )
    parser.add_argument(
        "--algo", type=str, default="sac", choices=["sac", "ppo"],
        help="Algorithm used for training",
    )
    parser.add_argument(
        "--episodes", type=int, default=20,
        help="Number of evaluation episodes (default: 20)",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render in PyBullet GUI",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed",
    )
    args = parser.parse_args()

    # ─── Load Model ──────────────────────────────────────
    print(f"Loading model from: {args.model}")
    ModelClass = SAC if args.algo == "sac" else PPO
    model = ModelClass.load(args.model)

    # ─── Environment ─────────────────────────────────────
    env = ReachEnv(
        render_mode="human" if args.render else None,
    )

    # ─── Evaluation Loop ─────────────────────────────────
    print(f"\nEvaluating for {args.episodes} episodes...\n")

    rewards_list = []
    distances_list = []
    successes = 0
    steps_list = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total_reward = 0
        done = False
        ep_steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            ep_steps += 1

        final_dist = info.get("distance", float("inf"))
        success = info.get("is_success", final_dist < 0.02)
        successes += int(success)

        rewards_list.append(total_reward)
        distances_list.append(final_dist)
        steps_list.append(ep_steps)

        status = "✅ SUCCESS" if success else "❌ FAIL"
        print(f"  Episode {ep+1:3d}: {status}  reward={total_reward:7.2f}  dist={final_dist:.4f}m  steps={ep_steps}")

    env.close()

    # ─── Summary ─────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  EVALUATION SUMMARY")
    print("=" * 50)
    print(f"  Episodes:       {args.episodes}")
    print(f"  Success Rate:   {successes}/{args.episodes} ({100*successes/args.episodes:.1f}%)")
    print(f"  Avg Reward:     {np.mean(rewards_list):.2f} ± {np.std(rewards_list):.2f}")
    print(f"  Avg Distance:   {np.mean(distances_list):.4f} ± {np.std(distances_list):.4f}m")
    print(f"  Avg Steps:      {np.mean(steps_list):.1f}")
    print(f"  Min Distance:   {np.min(distances_list):.4f}m")
    print("=" * 50)


if __name__ == "__main__":
    main()
