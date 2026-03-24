"""
Gymnasium environments for MCAD 6DOF arm RL training.
ReachEnv: move end-effector to a random 3D target.
"""
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

from mcad_arm_rl.config import (
    RobotConfig, EnvConfig, PickPlaceConfig,
    ROBOT_URDF, TARGET_URDF, BLOCK_URDF, ASSETS_DIR,
)
from mcad_arm_rl.robot import MCadRobot
from mcad_arm_rl.rewards import reach_reward, pick_reward
from mcad_arm_rl.randomization import DomainRandomizer


class ReachEnv(gym.Env):
    """
    6DOF arm reaching environment.
    
    Goal: move the end-effector (tool0) to a randomly spawned target position.
    
    Observation (18D):
        - 6 joint positions (normalized to [-1, 1])
        - 6 joint velocities (clipped)
        - 3D end-effector position
        - 3D target position
    
    Action (6D):
        - Joint velocity deltas ∈ [-1, 1], scaled by action_scale
    
    Reward:
        - Shaped: -distance + goal_bonus - action_penalty - velocity_penalty
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode: str = None,
        robot_config: RobotConfig = None,
        env_config: EnvConfig = None,
        domain_randomization: bool = False,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.robot_config = robot_config or RobotConfig()
        self.env_config = env_config or EnvConfig()
        self.use_dr = domain_randomization

        # ─── Spaces ──────────────────────────────────────────
        n_joints = self.robot_config.n_arm_joints  # 6

        # Action: 6 joint velocity deltas in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(n_joints,),
            dtype=np.float32,
        )

        # Observation: joints(6) + velocities(6) + ee_pos(3) + target(3) = 18
        obs_dim = n_joints * 2 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # ─── PyBullet ───────────────────────────────────────
        self._physics_client = None
        self._robot: MCadRobot = None
        self._target_id = None
        self._plane_id = None
        self._target_pos = np.zeros(3, dtype=np.float32)
        self._step_count = 0
        self._episode_count = 0
        self._rng = np.random.default_rng()
        self._randomizer: DomainRandomizer = None

        # Deferred init (created on first reset)
        self._initialized = False

    def _init_sim(self) -> None:
        """Initialize PyBullet simulation."""
        if self.render_mode == "human":
            self._physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self._physics_client)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self._physics_client)

            # Nice camera view
            p.resetDebugVisualizerCamera(
                cameraDistance=0.5,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0.15, 0, 0.15],
                physicsClientId=self._physics_client,
            )
        else:
            self._physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.env_config.gravity, physicsClientId=self._physics_client)
        p.setTimeStep(self.env_config.sim_timestep, physicsClientId=self._physics_client)

        # Ground plane
        self._plane_id = p.loadURDF(
            "plane.urdf", physicsClientId=self._physics_client,
        )

        # Robot (URDF mesh paths are relative to assets/)
        self._robot = MCadRobot(
            physics_client=self._physics_client,
            config=self.robot_config,
            urdf_path=ROBOT_URDF,
        )

        # Target visual marker
        if os.path.exists(TARGET_URDF):
            self._target_id = p.loadURDF(
                TARGET_URDF,
                basePosition=[0.2, 0, 0.15],
                useFixedBase=True,
                physicsClientId=self._physics_client,
            )

        # Domain randomizer
        if self.use_dr:
            self._randomizer = DomainRandomizer(
                physics_client=self._physics_client,
                enabled=True,
            )

        self._initialized = True

    # ─── Gymnasium API ───────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if not self._initialized:
            self._init_sim()

        self._step_count = 0
        self._episode_count += 1

        # Reset the robot to home position
        self._robot.reset()

        # Randomize target position within (reachable) workspace
        self._target_pos = np.array([
            self._rng.uniform(*self.env_config.target_x_range),
            self._rng.uniform(*self.env_config.target_y_range),
            self._rng.uniform(*self.env_config.target_z_range),
        ], dtype=np.float32)

        # Move the visual target marker
        if self._target_id is not None:
            p.resetBasePositionAndOrientation(
                self._target_id,
                self._target_pos.tolist(),
                [0, 0, 0, 1],
                physicsClientId=self._physics_client,
            )

        # Apply domain randomization
        if self._randomizer is not None:
            self._randomizer.randomize(self._robot.robot_id, rng=self._rng)

        # Step physics a few times to settle
        for _ in range(10):
            p.stepSimulation(physicsClientId=self._physics_client)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Apply action and step physics
        self._robot.apply_action(action, action_scale=self.env_config.action_scale)

        for _ in range(self.env_config.n_substeps):
            p.stepSimulation(physicsClientId=self._physics_client)

        self._step_count += 1

        # Get observation
        obs = self._get_obs()

        # Compute reward
        ee_pos = self._robot.get_end_effector_pos()
        joint_vel = self._robot.get_joint_velocities()
        reward, is_success, reward_info = reach_reward(
            ee_pos=ee_pos,
            target_pos=self._target_pos,
            action=action,
            joint_velocities=joint_vel,
            config=self.env_config,
        )

        # Done conditions (cast to native Python types for Gymnasium compliance)
        terminated = bool(is_success)
        truncated = bool(self._step_count >= self.env_config.max_episode_steps)

        info = self._get_info()
        info.update(reward_info)

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Build flat observation vector (18D)."""
        joint_pos = self._robot.get_joint_positions()
        joint_vel = self._robot.get_joint_velocities()
        ee_pos = self._robot.get_end_effector_pos()

        # Normalize joint positions to [-1, 1]
        joint_range = self.robot_config.joint_upper - self.robot_config.joint_lower
        joint_mid = (self.robot_config.joint_upper + self.robot_config.joint_lower) / 2
        norm_pos = (joint_pos - joint_mid) / (joint_range / 2 + 1e-8)

        # Clip velocities
        clipped_vel = np.clip(joint_vel, -5.0, 5.0) / 5.0

        obs = np.concatenate([
            norm_pos,            # 6D - normalized joint positions
            clipped_vel,         # 6D - normalized joint velocities
            ee_pos,              # 3D - end-effector position
            self._target_pos,    # 3D - target position
        ]).astype(np.float32)

        return obs

    def _get_info(self) -> dict:
        ee_pos = self._robot.get_end_effector_pos()
        return {
            "distance": float(np.linalg.norm(ee_pos - self._target_pos)),
            "ee_pos": ee_pos.tolist(),
            "target_pos": self._target_pos.tolist(),
            "step": self._step_count,
            "episode_num": self._episode_count,
        }

    def render(self):
        """Render for 'rgb_array' mode."""
        if self.render_mode == "rgb_array":
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0.15, 0, 0.15],
                distance=0.5,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2,
                physicsClientId=self._physics_client,
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=float(self.env_config.render_width) / self.env_config.render_height,
                nearVal=0.01,
                farVal=10.0,
                physicsClientId=self._physics_client,
            )
            _, _, img, _, _ = p.getCameraImage(
                width=self.env_config.render_width,
                height=self.env_config.render_height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                physicsClientId=self._physics_client,
            )
            return np.array(img, dtype=np.uint8).reshape(
                self.env_config.render_height,
                self.env_config.render_width,
                4,
            )[:, :, :3]  # Drop alpha
        return None

    def close(self):
        if self._physics_client is not None:
            p.disconnect(self._physics_client)
            self._physics_client = None
            self._initialized = False


class PickPlaceEnv(gym.Env):
    """
    6DOF arm + gripper pick-and-place environment.

    Goal: pick up a block and place it at a target location.

    Observation (25D):
        - 7 joint positions (6 arm + 1 gripper, normalized)
        - 7 joint velocities (6 arm + 1 gripper, clipped)
        - 3D end-effector position
        - 3D block position
        - 3D place target position
        - 1 gripper state (normalized)
        - 1 grasp flag (0 or 1)

    Action (7D):
        - 6 arm joint velocity deltas ∈ [-1, 1]
        - 1 gripper command ∈ [-1, 1] (negative=open, positive=close)

    Reward (multi-stage):
        Stage 1 (Reach):  -distance(EE, block)
        Stage 2 (Grasp):  bonus for grasping
        Stage 3 (Lift):   bonus for lifting block above threshold
        Stage 4 (Place):  -distance(block, target) + bonus for placing
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode: str = None,
        robot_config: RobotConfig = None,
        env_config: EnvConfig = None,
        pick_config: PickPlaceConfig = None,
        domain_randomization: bool = False,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.robot_config = robot_config or RobotConfig()
        self.env_config = env_config or EnvConfig()
        self.pick_config = pick_config or PickPlaceConfig()
        self.use_dr = domain_randomization

        # ─── Spaces ──────────────────────────────────────────
        # Action: 7D (6 arm + 1 gripper)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32,
        )

        # Observation: 25D
        # 7 pos + 7 vel + 3 ee + 3 block + 3 target + 1 grip + 1 grasp = 25
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32,
        )

        # ─── State ───────────────────────────────────────────
        self._physics_client = None
        self._robot: MCadRobot = None
        self._block_id = None
        self._place_marker_id = None
        self._plane_id = None
        self._block_pos = np.zeros(3, dtype=np.float32)
        self._place_target = np.zeros(3, dtype=np.float32)
        self._is_grasped = False
        self._was_grasped = False  # Track if block was ever grasped (drop detection)
        self._step_count = 0
        self._episode_count = 0
        self._rng = np.random.default_rng()
        self._randomizer: DomainRandomizer = None
        self._initialized = False

    def _init_sim(self) -> None:
        """Initialize PyBullet simulation."""
        if self.render_mode == "human":
            self._physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self._physics_client)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self._physics_client)
            p.resetDebugVisualizerCamera(
                cameraDistance=0.6, cameraYaw=45, cameraPitch=-30,
                cameraTargetPosition=[0.15, 0, 0.1],
                physicsClientId=self._physics_client,
            )
        else:
            self._physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.env_config.gravity, physicsClientId=self._physics_client)
        p.setTimeStep(self.env_config.sim_timestep, physicsClientId=self._physics_client)

        # Ground plane
        self._plane_id = p.loadURDF("plane.urdf", physicsClientId=self._physics_client)

        # Robot
        self._robot = MCadRobot(
            physics_client=self._physics_client,
            config=self.robot_config,
            urdf_path=ROBOT_URDF,
        )

        # Place target marker (blue sphere)
        if os.path.exists(TARGET_URDF):
            self._place_marker_id = p.loadURDF(
                TARGET_URDF,
                basePosition=[0.2, 0, 0.015],
                useFixedBase=True,
                physicsClientId=self._physics_client,
            )
            # Make it blue to distinguish from green reach target
            p.changeVisualShape(
                self._place_marker_id, -1,
                rgbaColor=[0.2, 0.4, 0.9, 0.7],
                physicsClientId=self._physics_client,
            )

        if self.use_dr:
            self._randomizer = DomainRandomizer(
                physics_client=self._physics_client, enabled=True,
            )

        self._initialized = True

    # ─── Gymnasium API ───────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if not self._initialized:
            self._init_sim()

        self._step_count = 0
        self._episode_count += 1
        self._is_grasped = False
        self._was_grasped = False

        # Reset robot to home with open gripper
        self._robot.reset()

        # Remove old block
        if self._block_id is not None:
            p.removeBody(self._block_id, physicsClientId=self._physics_client)

        # Spawn block at random position
        pc = self.pick_config
        block_x = self._rng.uniform(*pc.block_x_range)
        block_y = self._rng.uniform(*pc.block_y_range)
        self._block_pos = np.array([block_x, block_y, pc.block_z_start], dtype=np.float32)

        self._block_id = p.loadURDF(
            BLOCK_URDF,
            basePosition=self._block_pos.tolist(),
            useFixedBase=False,
            physicsClientId=self._physics_client,
        )

        # Set block friction
        p.changeDynamics(
            self._block_id, -1,
            lateralFriction=1.0,
            spinningFriction=0.01,
            physicsClientId=self._physics_client,
        )

        # Place target (ensure it's different from block spawn)
        for _ in range(20):
            place_x = self._rng.uniform(*pc.place_x_range)
            place_y = self._rng.uniform(*pc.place_y_range)
            self._place_target = np.array([place_x, place_y, pc.place_z], dtype=np.float32)
            if np.linalg.norm(self._place_target[:2] - self._block_pos[:2]) > 0.05:
                break

        # Move place marker
        if self._place_marker_id is not None:
            p.resetBasePositionAndOrientation(
                self._place_marker_id,
                self._place_target.tolist(),
                [0, 0, 0, 1],
                physicsClientId=self._physics_client,
            )

        # Domain randomization
        if self._randomizer is not None:
            self._randomizer.randomize(self._robot.robot_id, rng=self._rng)

        # Settle physics
        for _ in range(20):
            p.stepSimulation(physicsClientId=self._physics_client)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Apply 7D action (arm + gripper)
        self._robot.apply_action_with_gripper(
            action,
            action_scale=self.env_config.action_scale,
            gripper_scale=self.pick_config.gripper_action_scale,
        )

        for _ in range(self.env_config.n_substeps):
            p.stepSimulation(physicsClientId=self._physics_client)

        self._step_count += 1

        # Update block position
        block_state = p.getBasePositionAndOrientation(
            self._block_id, physicsClientId=self._physics_client,
        )
        self._block_pos = np.array(block_state[0], dtype=np.float32)

        # Check grasp
        self._is_grasped = self._robot.check_grasp(self._block_id)
        if self._is_grasped:
            self._was_grasped = True

        # Observation
        obs = self._get_obs()

        # Compute reward
        ee_pos = self._robot.get_end_effector_pos()
        gripper_pos = self._robot.get_gripper_position()

        reward, stage, reward_info = pick_reward(
            ee_pos=ee_pos,
            block_pos=self._block_pos,
            target_place_pos=self._place_target,
            gripper_pos=gripper_pos,
            is_grasped=self._is_grasped,
            action=action,
            config=self.env_config,
        )

        # Drop penalty: was grasped but no longer
        if self._was_grasped and not self._is_grasped:
            reward += self.pick_config.drop_penalty

        # Success: block placed at target
        place_dist = np.linalg.norm(self._block_pos - self._place_target)
        is_success = place_dist < self.env_config.goal_threshold and not self._is_grasped
        terminated = bool(is_success)
        truncated = bool(self._step_count >= self.pick_config.max_episode_steps)

        # Block fell off table
        if self._block_pos[2] < -0.05:
            reward += self.pick_config.drop_penalty
            truncated = True

        info = self._get_info()
        info.update(reward_info)
        info["is_success"] = is_success

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Build 25D observation vector."""
        # Joint positions (7D: arm + gripper, normalized)
        arm_pos = self._robot.get_joint_positions()
        grip_pos = self._robot.get_gripper_position()
        joint_range = self.robot_config.joint_upper - self.robot_config.joint_lower
        joint_mid = (self.robot_config.joint_upper + self.robot_config.joint_lower) / 2
        norm_arm = (arm_pos - joint_mid) / (joint_range / 2 + 1e-8)
        norm_grip = np.array([grip_pos / 0.65], dtype=np.float32)  # Normalize to [-1, 1]

        # Joint velocities (7D, clipped)
        arm_vel = self._robot.get_joint_velocities()
        clipped_arm_vel = np.clip(arm_vel, -5.0, 5.0) / 5.0
        grip_vel = self._robot.get_full_joint_velocities()[-1:]
        clipped_grip_vel = np.clip(grip_vel, -5.0, 5.0) / 5.0

        # Positions
        ee_pos = self._robot.get_end_effector_pos()

        # Grasp flag
        grasp_flag = np.array([1.0 if self._is_grasped else 0.0], dtype=np.float32)

        obs = np.concatenate([
            norm_arm,           # 6D
            norm_grip,          # 1D
            clipped_arm_vel,    # 6D
            clipped_grip_vel,   # 1D
            ee_pos,             # 3D
            self._block_pos,    # 3D
            self._place_target, # 3D
            norm_grip,          # 1D (gripper state)
            grasp_flag,         # 1D
        ]).astype(np.float32)

        return obs

    def _get_info(self) -> dict:
        ee_pos = self._robot.get_end_effector_pos()
        reach_dist = float(np.linalg.norm(ee_pos - self._block_pos))
        place_dist = float(np.linalg.norm(self._block_pos - self._place_target))
        return {
            "distance": reach_dist,
            "reach_dist": reach_dist,
            "place_dist": place_dist,
            "block_pos": self._block_pos.tolist(),
            "place_target": self._place_target.tolist(),
            "ee_pos": ee_pos.tolist(),
            "is_grasped": self._is_grasped,
            "step": self._step_count,
            "episode_num": self._episode_count,
        }

    def render(self):
        if self.render_mode == "rgb_array":
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0.15, 0, 0.1],
                distance=0.6, yaw=45, pitch=-30, roll=0, upAxisIndex=2,
                physicsClientId=self._physics_client,
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=float(self.env_config.render_width) / self.env_config.render_height,
                nearVal=0.01, farVal=10.0,
                physicsClientId=self._physics_client,
            )
            _, _, img, _, _ = p.getCameraImage(
                width=self.env_config.render_width,
                height=self.env_config.render_height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                physicsClientId=self._physics_client,
            )
            return np.array(img, dtype=np.uint8).reshape(
                self.env_config.render_height,
                self.env_config.render_width, 4,
            )[:, :, :3]
        return None

    def close(self):
        if self._physics_client is not None:
            p.disconnect(self._physics_client)
            self._physics_client = None
            self._initialized = False

