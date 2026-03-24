#!/usr/bin/env python3
"""
Training script for MCAD 6DOF arm RL.
Supports SAC and PPO algorithms via Stable-Baselines3.

Usage:
    # Reach task (default)
    python scripts/train.py --algo sac --timesteps 500000

    # Pick and place task
    python scripts/train.py --algo sac --task pick --timesteps 1000000

    # With domain randomization + GPU
    python scripts/train.py --algo sac --task pick --timesteps 500000 --domain-rand --device cuda
"""
import os
import sys
import argparse
from datetime import datetime

import torch
import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcad_arm_rl.env import ReachEnv, PickPlaceEnv
from mcad_arm_rl.config import TrainConfig, EnvConfig, LOG_DIR, CHECKPOINT_DIR


def make_env(rank: int, seed: int, task: str = "reach", domain_rand: bool = False):
    """Create a wrapped environment factory for SubprocVecEnv."""
    def _init():
        EnvClass = PickPlaceEnv if task == "pick" else ReachEnv
        env = EnvClass(
            render_mode=None,  # No rendering during training
            domain_randomization=domain_rand,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def detect_device(requested: str) -> str:
    """Auto-detect best available device."""
    if requested == "auto":
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"  🚀 GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
            return "cuda"
        else:
            print("  💻 No GPU detected, using CPU")
            return "cpu"
    return requested


def main():
    parser = argparse.ArgumentParser(description="Train MCAD Arm RL Agent")
    parser.add_argument(
        "--algo", type=str, default="sac", choices=["sac", "ppo"],
        help="RL algorithm to use (default: sac)",
    )
    parser.add_argument(
        "--task", type=str, default="reach", choices=["reach", "pick"],
        help="Task: 'reach' (move to target) or 'pick' (pick and place)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=500_000,
        help="Total training timesteps (default: 500000)",
    )
    parser.add_argument(
        "--n-envs", type=int, default=1,
        help="Number of parallel environments (default: 1, use >1 for PPO)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--domain-rand", action="store_true",
        help="Enable domain randomization",
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
        help="Training device (default: auto-detect)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a saved model to resume training from",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=10_000,
        help="Evaluate every N steps (default: 10000)",
    )
    args = parser.parse_args()

    config = TrainConfig()
    device = detect_device(args.device)

    # ─── Timestamps for unique run names ─────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.algo}_{args.task}_{timestamp}_seed{args.seed}"
    tb_log_path = os.path.join(LOG_DIR, "tensorboard")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, run_name)
    os.makedirs(tb_log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    print("=" * 60)
    print(f"  MCAD Arm RL Training")
    print(f"  Task:         {args.task.upper()}")
    print(f"  Algorithm:    {args.algo.upper()}")
    print(f"  Device:       {device.upper()}")
    print(f"  Timesteps:    {args.timesteps:,}")
    print(f"  Parallel envs:{args.n_envs}")
    print(f"  Seed:         {args.seed}")
    print(f"  Domain Rand:  {args.domain_rand}")
    print(f"  Run name:     {run_name}")
    print(f"  Checkpoints:  {checkpoint_path}")
    print(f"  TensorBoard:  {tb_log_path}")
    print("=" * 60)

    # ─── Environment Setup ───────────────────────────────
    print("\n[1/4] Creating environment(s)...")

    if args.n_envs == 1:
        # Single env, verify it first
        EnvClass = PickPlaceEnv if args.task == "pick" else ReachEnv
        test_env = EnvClass(domain_randomization=args.domain_rand)
        print("  Running environment check...")
        check_env(test_env, warn=True)
        print("  ✅ Environment check passed!")
        test_env.close()

        env = DummyVecEnv([make_env(0, args.seed, args.task, args.domain_rand)])
    else:
        env = SubprocVecEnv([
            make_env(i, args.seed, args.task, args.domain_rand)
            for i in range(args.n_envs)
        ])

    # Eval env (always single, no domain rand for fair eval)
    eval_env = DummyVecEnv([make_env(0, args.seed + 1000, args.task, False)])

    # ─── Model Setup ─────────────────────────────────────
    print(f"\n[2/4] Initializing {args.algo.upper()} model on {device}...")

    # Pick-place uses bigger network
    policy_kwargs = dict(config.policy_kwargs)
    if args.task == "pick":
        policy_kwargs["net_arch"] = [512, 512, 256]

    if args.resume:
        print(f"  Resuming from: {args.resume}")
        ModelClass = SAC if args.algo == "sac" else PPO
        model = ModelClass.load(args.resume, env=env, device=device)
    elif args.algo == "sac":
        # SAC with larger buffer for pick-place
        buffer_size = 1_000_000 if args.task == "pick" else config.sac_buffer_size
        model = SAC(
            policy=config.policy,
            env=env,
            learning_rate=config.sac_learning_rate,
            buffer_size=buffer_size,
            learning_starts=config.sac_learning_starts,
            batch_size=config.sac_batch_size,
            tau=config.sac_tau,
            gamma=config.sac_gamma,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            device=device,
            tensorboard_log=tb_log_path,
        )
    else:  # PPO
        model = PPO(
            policy=config.policy,
            env=env,
            learning_rate=config.ppo_learning_rate,
            n_steps=config.ppo_n_steps,
            batch_size=config.ppo_batch_size,
            n_epochs=config.ppo_n_epochs,
            gamma=config.ppo_gamma,
            clip_range=config.ppo_clip_range,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            device=device,
            tensorboard_log=tb_log_path,
        )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Total parameters: {total_params:,}")

    # ─── Callbacks ───────────────────────────────────────
    print("\n[3/4] Setting up callbacks...")

    checkpoint_cb = CheckpointCallback(
        save_freq=max(config.save_interval // args.n_envs, 1),
        save_path=checkpoint_path,
        name_prefix="mcad_arm",
        save_replay_buffer=(args.algo == "sac"),
        save_vecnormalize=True,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(checkpoint_path, "best"),
        log_path=os.path.join(LOG_DIR, "eval", run_name),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=config.eval_episodes,
        deterministic=True,
    )

    callbacks = CallbackList([checkpoint_cb, eval_cb])

    # ─── Training ────────────────────────────────────────
    print(f"\n[4/4] Starting training for {args.timesteps:,} timesteps...")
    print("  (Monitor progress with: tensorboard --logdir logs/tensorboard)\n")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name=run_name,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted. Saving current model...")

    # ─── Save Final Model ────────────────────────────────
    final_path = os.path.join(checkpoint_path, "final_model")
    model.save(final_path)
    print(f"\n✅ Final model saved to: {final_path}.zip")
    print(f"📊 View training curves: tensorboard --logdir {tb_log_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
