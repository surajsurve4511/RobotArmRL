"""
Centralized configuration for MCAD Arm RL training.
All hyperparameters, joint limits, and paths are defined here.
"""
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

# ─── Paths ───────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
ROBOT_URDF = os.path.join(ASSETS_DIR, "mcad_robot.urdf")
TABLE_URDF = os.path.join(ASSETS_DIR, "table.urdf")
BLOCK_URDF = os.path.join(ASSETS_DIR, "block.urdf")
TARGET_URDF = os.path.join(ASSETS_DIR, "target_sphere.urdf")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")


@dataclass
class RobotConfig:
    """Robot physical parameters (from URDF xacro files)."""

    # Joint names in kinematic order
    arm_joint_names: List[str] = field(default_factory=lambda: [
        "r2_joint1", "r2_joint2", "r2_joint3",
        "r2_joint4", "r2_joint5", "r2_joint6",
    ])
    gripper_joint_name: str = "r2_joint7"

    # Joint limits [lower, upper] in radians — exact values from xacro
    joint_limits: List[Tuple[float, float]] = field(default_factory=lambda: [
        (-1.57, 1.57),      # joint1: base rotation (Z)
        (-0.5236, 1.2217),  # joint2: shoulder (Y)
        (-0.8378, 0.5236),  # joint3: elbow (Y)
        (-1.58, 1.58),      # joint4: forearm roll (X)
        (-0.85, 0.85),      # joint5: wrist pitch (Y)
        (-1.58, 1.58),      # joint6: wrist roll (X)
    ])
    gripper_limits: Tuple[float, float] = (-0.65, 0.65)

    # Max joint torques (effort)
    max_torques: List[float] = field(default_factory=lambda: [3.0] * 6)
    gripper_max_torque: float = 1.5

    # Max joint velocities (rad/s)
    max_velocities: List[float] = field(default_factory=lambda: [2.0] * 6)
    gripper_max_velocity: float = 2.0

    # Home position (all zeros = straight up)
    home_position: List[float] = field(default_factory=lambda: [0.0] * 6)

    # Tool0 (TCP) frame name for FK lookups
    tcp_link_name: str = "r2_tool0"

    @property
    def n_arm_joints(self) -> int:
        return len(self.arm_joint_names)

    @property
    def joint_lower(self) -> np.ndarray:
        return np.array([l[0] for l in self.joint_limits], dtype=np.float32)

    @property
    def joint_upper(self) -> np.ndarray:
        return np.array([l[1] for l in self.joint_limits], dtype=np.float32)


@dataclass
class EnvConfig:
    """Environment and simulation configuration."""

    # Simulation
    sim_timestep: float = 1.0 / 240.0       # PyBullet default (240 Hz physics)
    n_substeps: int = 12                      # Steps per action → 20 Hz control
    gravity: float = -9.81

    # Episode
    max_episode_steps: int = 100              # 100 steps × 0.05s = 5 second episodes
    goal_threshold: float = 0.02             # 2cm — success if EE within this of target

    # Target spawning (workspace bounds in meters, relative to base)
    target_x_range: Tuple[float, float] = (0.10, 0.30)
    target_y_range: Tuple[float, float] = (-0.15, 0.15)
    target_z_range: Tuple[float, float] = (0.05, 0.25)

    # Reward weights
    reward_distance_weight: float = 1.0       # Weight for -distance term
    reward_goal_bonus: float = 10.0           # Bonus for reaching within threshold
    reward_action_penalty: float = 0.01      # Penalty for large actions (smoothness)
    reward_velocity_penalty: float = 0.001   # Penalty for high joint velocities

    # Action scaling: actions ∈ [-1,1] → joint velocity × action_scale
    action_scale: float = 0.05               # radians per step at max action

    # Rendering
    render_width: int = 480
    render_height: int = 360


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Algorithm (sac or ppo)
    algorithm: str = "sac"

    # Total timesteps
    total_timesteps: int = 500_000

    # SAC hyperparams
    sac_learning_rate: float = 3e-4
    sac_batch_size: int = 256
    sac_buffer_size: int = 300_000
    sac_learning_starts: int = 1000
    sac_tau: float = 0.005
    sac_gamma: float = 0.99

    # PPO hyperparams
    ppo_learning_rate: float = 3e-4
    ppo_batch_size: int = 64
    ppo_n_steps: int = 2048
    ppo_n_epochs: int = 10
    ppo_gamma: float = 0.99
    ppo_clip_range: float = 0.2

    # Shared
    policy: str = "MlpPolicy"
    policy_kwargs: dict = field(default_factory=lambda: {
        "net_arch": [256, 256],  # Two hidden layers of 256 units
    })
    seed: int = 42

    # Parallel envs
    n_envs: int = 4

    # Logging
    log_interval: int = 10
    save_interval: int = 50_000       # Save checkpoint every N steps
    eval_episodes: int = 20           # Episodes for evaluation


@dataclass
class PickPlaceConfig:
    """Pick and place task specific configuration."""

    # Block spawn range (in front of robot on table surface)
    block_x_range: Tuple[float, float] = (0.15, 0.28)
    block_y_range: Tuple[float, float] = (-0.10, 0.10)
    block_z_start: float = 0.015  # Half block height (3cm block)

    # Place target range (different area from block spawn)
    place_x_range: Tuple[float, float] = (0.12, 0.25)
    place_y_range: Tuple[float, float] = (-0.12, 0.12)
    place_z: float = 0.015

    # Grasp detection
    grasp_threshold: float = 0.03     # EE must be within 3cm of block to grasp
    min_grasp_force: float = 0.1      # Minimum contact force for valid grasp
    lift_height: float = 0.08         # Block must rise above this to count as lifted

    # Gripper
    gripper_open: float = -0.6        # Open position
    gripper_close: float = 0.5        # Closed position
    gripper_action_scale: float = 0.1 # How fast gripper opens/closes per step

    # Reward weights (multi-stage)
    reach_reward_weight: float = 1.0
    grasp_bonus: float = 3.0
    lift_bonus: float = 5.0
    place_bonus: float = 15.0
    drop_penalty: float = -2.0       # Penalty for dropping the block

    # Episode
    max_episode_steps: int = 200      # Longer episodes for pick-place (10s)

