"""Test the Gymnasium environment passes SB3 check_env."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.env_checker import check_env
import numpy as np
from mcad_arm_rl.env import ReachEnv


def test_env_check():
    """Environment should pass SB3's strict validation."""
    env = ReachEnv()
    print("Running check_env...")
    check_env(env, warn=True)
    print("✅ check_env passed!")
    env.close()


def test_obs_shape():
    """Observation should be 18-dimensional."""
    env = ReachEnv()
    obs, info = env.reset(seed=42)

    expected_dim = 6 + 6 + 3 + 3  # joints + velocities + ee_pos + target
    assert obs.shape == (expected_dim,), f"Expected shape ({expected_dim},), got {obs.shape}"
    print(f"✅ Observation shape: {obs.shape}")

    env.close()


def test_action_shape():
    """Action space should be 6-dimensional."""
    env = ReachEnv()
    assert env.action_space.shape == (6,), f"Expected (6,), got {env.action_space.shape}"
    print(f"✅ Action space: {env.action_space}")
    env.close()


def test_step_runs():
    """One step should execute without errors."""
    env = ReachEnv()
    obs, info = env.reset(seed=42)

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (18,)
    assert np.isscalar(reward), f"Reward should be scalar, got {type(reward)}"
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "distance" in info

    print(f"✅ Step works: reward={reward:.4f}, distance={info['distance']:.4f}")
    env.close()


if __name__ == "__main__":
    test_obs_shape()
    test_action_shape()
    test_step_runs()
    test_env_check()
    print("\n✅ All environment tests passed!")
