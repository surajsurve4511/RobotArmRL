#!/usr/bin/env python3
"""
Visualize a trained policy in PyBullet GUI.

Usage:
    python scripts/visualize.py --model checkpoints/sac_*/final_model.zip
    python scripts/visualize.py --model checkpoints/sac_*/best/best_model.zip --episodes 5
"""
import os
import sys
import time
import argparse

from stable_baselines3 import SAC, PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcad_arm_rl.env import ReachEnv


def main():
    parser = argparse.ArgumentParser(description="Visualize MCAD arm RL policy")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to saved model (.zip)",
    )
    parser.add_argument(
        "--algo", type=str, default="sac", choices=["sac", "ppo"],
        help="Algorithm used for training",
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of episodes to visualize",
    )
    parser.add_argument(
        "--slow", action="store_true",
        help="Slow down for easier viewing (0.05s per step)",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    ModelClass = SAC if args.algo == "sac" else PPO
    model = ModelClass.load(args.model)

    env = ReachEnv(render_mode="human")

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step = 0

        print(f"\n--- Episode {ep+1} ---")
        print(f"  Target: ({info['target_pos'][0]:.3f}, {info['target_pos'][1]:.3f}, {info['target_pos'][2]:.3f})")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1

            if args.slow:
                time.sleep(0.05)

        print(f"  Result:  {'✅ SUCCESS' if info.get('is_success', False) else '❌ FAIL'}")
        print(f"  Distance: {info['distance']:.4f}m | Reward: {total_reward:.2f} | Steps: {step}")

        time.sleep(1.0)  # Pause between episodes

    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
