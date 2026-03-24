"""Test the PickPlaceEnv Gymnasium environment."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.env_checker import check_env
import numpy as np
from mcad_arm_rl.env import PickPlaceEnv


def test_pick_obs_shape():
    """Observation should be 25-dimensional."""
    env = PickPlaceEnv()
    obs, info = env.reset(seed=42)

    expected_dim = 25  # 7pos + 7vel + 3ee + 3block + 3target + 1grip + 1grasp
    assert obs.shape == (expected_dim,), f"Expected shape ({expected_dim},), got {obs.shape}"
    print(f"✅ Observation shape: {obs.shape}")
    env.close()


def test_pick_action_shape():
    """Action space should be 7-dimensional."""
    env = PickPlaceEnv()
    assert env.action_space.shape == (7,), f"Expected (7,), got {env.action_space.shape}"
    print(f"✅ Action space: {env.action_space}")
    env.close()


def test_pick_step_runs():
    """One step should execute without errors."""
    env = PickPlaceEnv()
    obs, info = env.reset(seed=42)

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (25,)
    assert np.isscalar(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "reach_dist" in info
    assert "place_dist" in info
    assert "is_grasped" in info

    print(f"✅ Step works: reward={reward:.4f}, reach_dist={info['reach_dist']:.4f}, place_dist={info['place_dist']:.4f}")
    env.close()


def test_pick_check_env():
    """Environment should pass SB3's strict validation."""
    env = PickPlaceEnv()
    print("Running check_env for PickPlaceEnv...")
    check_env(env, warn=True)
    print("✅ check_env passed!")
    env.close()


if __name__ == "__main__":
    test_pick_obs_shape()
    test_pick_action_shape()
    test_pick_step_runs()
    test_pick_check_env()
    print("\n✅ All PickPlaceEnv tests passed!")
