"""
Reward functions for MCAD arm RL training.
Separated from the environment for easy tuning and swapping.
"""
import numpy as np
from mcad_arm_rl.config import EnvConfig


def reach_reward(
    ee_pos: np.ndarray,
    target_pos: np.ndarray,
    action: np.ndarray,
    joint_velocities: np.ndarray,
    config: EnvConfig,
) -> tuple:
    """
    Shaped reward for reaching a target position.

    Returns:
        (reward, is_success, info_dict)
    """
    distance = np.linalg.norm(ee_pos - target_pos)
    is_success = distance < config.goal_threshold

    # Dense distance-based reward (negative, closer = better)
    dist_reward = -config.reward_distance_weight * distance

    # Goal bonus
    goal_bonus = config.reward_goal_bonus if is_success else 0.0

    # Action smoothness penalty (prefer small actions)
    action_penalty = -config.reward_action_penalty * np.sum(action ** 2)

    # Velocity penalty (prefer slow, controlled movements)
    vel_penalty = -config.reward_velocity_penalty * np.sum(joint_velocities ** 2)

    total_reward = dist_reward + goal_bonus + action_penalty + vel_penalty

    info = {
        "distance": distance,
        "dist_reward": dist_reward,
        "goal_bonus": goal_bonus,
        "action_penalty": action_penalty,
        "vel_penalty": vel_penalty,
        "is_success": is_success,
    }

    return total_reward, is_success, info


def pick_reward(
    ee_pos: np.ndarray,
    block_pos: np.ndarray,
    target_place_pos: np.ndarray,
    gripper_pos: float,
    is_grasped: bool,
    action: np.ndarray,
    config: EnvConfig,
) -> tuple:
    """
    Multi-stage reward for pick and place.
    Stage 1: Reach the block
    Stage 2: Grasp the block  
    Stage 3: Lift the block
    Stage 4: Place at target

    Returns:
        (reward, stage, info_dict)
    """
    reach_dist = np.linalg.norm(ee_pos - block_pos)
    place_dist = np.linalg.norm(block_pos - target_place_pos)

    if not is_grasped:
        # Stage 1: Reach for the block
        reward = -reach_dist
        stage = "reach"

        # Bonus for closing gripper when near block
        if reach_dist < 0.03 and gripper_pos > 0.0:
            reward += 2.0
            stage = "grasp_attempt"
    else:
        # Stage 2+: Block is grasped
        block_height = block_pos[2]

        if block_height < 0.08:
            # Stage 3: Lift the block
            reward = 2.0 + block_height * 10.0
            stage = "lift"
        else:
            # Stage 4: Move to place location
            reward = 5.0 - place_dist * 5.0
            stage = "place"

            if place_dist < config.goal_threshold:
                reward += config.reward_goal_bonus
                stage = "success"

    # Action penalty always applies
    action_penalty = -config.reward_action_penalty * np.sum(action ** 2)
    reward += action_penalty

    info = {
        "reach_dist": reach_dist,
        "place_dist": place_dist,
        "stage": stage,
        "is_grasped": is_grasped,
    }

    return reward, stage, info


def curriculum_target_range(
    episode_num: int,
    warmup_episodes: int = 5000,
    initial_range: float = 0.05,
    final_range: float = 0.15,
) -> float:
    """
    Gradually increase target spawn range over training.
    Start with easy (close) targets, expand to full workspace.
    """
    progress = min(episode_num / warmup_episodes, 1.0)
    return initial_range + progress * (final_range - initial_range)
