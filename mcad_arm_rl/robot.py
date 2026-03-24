"""
PyBullet robot wrapper for the MCAD 6DOF arm.
Handles URDF loading, joint control, and forward kinematics.
"""
import pybullet as p
import numpy as np
from typing import Optional, List, Tuple

from mcad_arm_rl.config import RobotConfig, ROBOT_URDF


class MCadRobot:
    """Wrapper around PyBullet for the MCAD 6DOF robotic arm."""

    def __init__(
        self,
        physics_client: int,
        config: Optional[RobotConfig] = None,
        urdf_path: Optional[str] = None,
        base_position: Tuple[float, float, float] = (0, 0, 0),
        base_orientation: Tuple[float, float, float, float] = (0, 0, 0, 1),
        use_fixed_base: bool = True,
    ):
        self.client = physics_client
        self.config = config or RobotConfig()
        self.urdf_path = urdf_path or ROBOT_URDF

        # Load the robot URDF
        self.robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=base_position,
            baseOrientation=base_orientation,
            useFixedBase=use_fixed_base,
            flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
            physicsClientId=self.client,
        )

        # Discover joint indices from the URDF
        self._arm_joint_indices: List[int] = []
        self._gripper_joint_index: Optional[int] = None
        self._tcp_link_index: Optional[int] = None
        self._joint_name_to_index = {}

        n_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        for i in range(n_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            joint_name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")
            self._joint_name_to_index[joint_name] = i

            if joint_name in self.config.arm_joint_names:
                self._arm_joint_indices.append(i)
            elif joint_name == self.config.gripper_joint_name:
                self._gripper_joint_index = i

            if link_name == self.config.tcp_link_name:
                self._tcp_link_index = i

        assert len(self._arm_joint_indices) == self.config.n_arm_joints, (
            f"Expected {self.config.n_arm_joints} arm joints, "
            f"found {len(self._arm_joint_indices)}: {self._arm_joint_indices}"
        )

        # Disable default velocity motors (we use position/velocity control manually)
        for idx in self._arm_joint_indices:
            p.setJointMotorControl2(
                self.robot_id, idx,
                p.VELOCITY_CONTROL,
                force=0,
                physicsClientId=self.client,
            )
        if self._gripper_joint_index is not None:
            p.setJointMotorControl2(
                self.robot_id, self._gripper_joint_index,
                p.VELOCITY_CONTROL,
                force=0,
                physicsClientId=self.client,
            )

    # ─── Properties ──────────────────────────────────────────

    @property
    def arm_joint_indices(self) -> List[int]:
        return self._arm_joint_indices

    @property
    def n_arm_joints(self) -> int:
        return len(self._arm_joint_indices)

    # ─── Joint State ─────────────────────────────────────────

    def get_joint_positions(self) -> np.ndarray:
        """Get current arm joint positions (6D)."""
        states = p.getJointStates(
            self.robot_id, self._arm_joint_indices,
            physicsClientId=self.client,
        )
        return np.array([s[0] for s in states], dtype=np.float32)

    def get_joint_velocities(self) -> np.ndarray:
        """Get current arm joint velocities (6D)."""
        states = p.getJointStates(
            self.robot_id, self._arm_joint_indices,
            physicsClientId=self.client,
        )
        return np.array([s[1] for s in states], dtype=np.float32)

    def get_gripper_position(self) -> float:
        """Get current gripper joint position."""
        if self._gripper_joint_index is None:
            return 0.0
        state = p.getJointState(
            self.robot_id, self._gripper_joint_index,
            physicsClientId=self.client,
        )
        return float(state[0])

    # ─── Forward Kinematics ──────────────────────────────────

    def get_end_effector_pos(self) -> np.ndarray:
        """Get tool0 (TCP) position in world frame via FK (3D)."""
        if self._tcp_link_index is None:
            raise RuntimeError(
                f"TCP link '{self.config.tcp_link_name}' not found in URDF"
            )
        state = p.getLinkState(
            self.robot_id, self._tcp_link_index,
            computeForwardKinematics=True,
            physicsClientId=self.client,
        )
        return np.array(state[4], dtype=np.float32)  # worldLinkFramePosition

    def get_end_effector_orn(self) -> np.ndarray:
        """Get tool0 (TCP) orientation as quaternion [x,y,z,w]."""
        if self._tcp_link_index is None:
            raise RuntimeError(
                f"TCP link '{self.config.tcp_link_name}' not found in URDF"
            )
        state = p.getLinkState(
            self.robot_id, self._tcp_link_index,
            computeForwardKinematics=True,
            physicsClientId=self.client,
        )
        return np.array(state[5], dtype=np.float32)  # worldLinkFrameOrientation

    # ─── Joint Control ───────────────────────────────────────

    def set_joint_positions(self, positions: np.ndarray) -> None:
        """Directly set arm joint positions (no physics, used for reset)."""
        positions = np.clip(
            positions,
            self.config.joint_lower,
            self.config.joint_upper,
        )
        for i, idx in enumerate(self._arm_joint_indices):
            p.resetJointState(
                self.robot_id, idx,
                targetValue=float(positions[i]),
                targetVelocity=0.0,
                physicsClientId=self.client,
            )

    def apply_action(self, action: np.ndarray, action_scale: float = 0.05) -> None:
        """
        Apply velocity-based action to arm joints.

        Args:
            action: 6D array in [-1, 1], scaled by action_scale
            action_scale: max position change per step in radians
        """
        current_pos = self.get_joint_positions()
        delta = action * action_scale

        target_pos = current_pos + delta
        target_pos = np.clip(
            target_pos,
            self.config.joint_lower,
            self.config.joint_upper,
        )

        # Use position control with moderate force
        for i, idx in enumerate(self._arm_joint_indices):
            p.setJointMotorControl2(
                self.robot_id, idx,
                p.POSITION_CONTROL,
                targetPosition=float(target_pos[i]),
                force=float(self.config.max_torques[i]),
                maxVelocity=float(self.config.max_velocities[i]),
                physicsClientId=self.client,
            )

    def set_gripper(self, position: float) -> None:
        """Set gripper to a specific position (open=-0.65, close=0.65)."""
        if self._gripper_joint_index is None:
            return
        position = np.clip(
            position,
            self.config.gripper_limits[0],
            self.config.gripper_limits[1],
        )
        p.setJointMotorControl2(
            self.robot_id, self._gripper_joint_index,
            p.POSITION_CONTROL,
            targetPosition=float(position),
            force=self.config.gripper_max_torque,
            physicsClientId=self.client,
        )

    # ─── Reset ───────────────────────────────────────────────

    def reset(self, positions: Optional[np.ndarray] = None) -> None:
        """Reset arm to home position (or given positions)."""
        if positions is None:
            positions = np.array(self.config.home_position, dtype=np.float32)
        self.set_joint_positions(positions)
        if self._gripper_joint_index is not None:
            p.resetJointState(
                self.robot_id, self._gripper_joint_index,
                targetValue=-0.6,  # Open gripper
                targetVelocity=0.0,
                physicsClientId=self.client,
            )

    # ─── Utility ─────────────────────────────────────────────

    def get_observation(self) -> dict:
        """Get full robot observation as a dict."""
        return {
            "joint_positions": self.get_joint_positions(),
            "joint_velocities": self.get_joint_velocities(),
            "ee_pos": self.get_end_effector_pos(),
            "ee_orn": self.get_end_effector_orn(),
            "gripper_pos": self.get_gripper_position(),
        }

    # ─── Pick & Place Methods ────────────────────────────────

    def get_full_joint_positions(self) -> np.ndarray:
        """Get all 7 joint positions (6 arm + 1 gripper)."""
        arm_pos = self.get_joint_positions()
        grip_pos = np.array([self.get_gripper_position()], dtype=np.float32)
        return np.concatenate([arm_pos, grip_pos])

    def get_full_joint_velocities(self) -> np.ndarray:
        """Get all 7 joint velocities (6 arm + 1 gripper)."""
        arm_vel = self.get_joint_velocities()
        if self._gripper_joint_index is not None:
            state = p.getJointState(
                self.robot_id, self._gripper_joint_index,
                physicsClientId=self.client,
            )
            grip_vel = np.array([state[1]], dtype=np.float32)
        else:
            grip_vel = np.array([0.0], dtype=np.float32)
        return np.concatenate([arm_vel, grip_vel])

    def apply_action_with_gripper(
        self,
        action: np.ndarray,
        action_scale: float = 0.05,
        gripper_scale: float = 0.1,
    ) -> None:
        """
        Apply 7D action: 6 arm velocity deltas + 1 gripper command.

        Args:
            action: 7D array in [-1, 1]. action[:6] = arm, action[6] = gripper
            action_scale: arm position change per step
            gripper_scale: gripper position change per step
        """
        # Arm joints (first 6)
        self.apply_action(action[:6], action_scale=action_scale)

        # Gripper (7th action): positive = close, negative = open
        if self._gripper_joint_index is not None:
            current_grip = self.get_gripper_position()
            target_grip = current_grip + float(action[6]) * gripper_scale
            target_grip = np.clip(
                target_grip,
                self.config.gripper_limits[0],
                self.config.gripper_limits[1],
            )
            p.setJointMotorControl2(
                self.robot_id, self._gripper_joint_index,
                p.POSITION_CONTROL,
                targetPosition=float(target_grip),
                force=self.config.gripper_max_torque,
                physicsClientId=self.client,
            )

    def check_grasp(self, block_id: int) -> bool:
        """
        Check if the gripper is grasping a block using PyBullet contact points.

        Returns True if there are contact points between the robot's gripper
        link and the block with sufficient normal force.
        """
        if self._gripper_joint_index is None:
            return False

        contacts = p.getContactPoints(
            bodyA=self.robot_id,
            bodyB=block_id,
            physicsClientId=self.client,
        )

        if len(contacts) == 0:
            return False

        # Check if any contact has enough force
        total_force = sum(c[9] for c in contacts)  # Normal force
        return total_force > 0.1

    def __repr__(self) -> str:
        return (
            f"MCadRobot(id={self.robot_id}, "
            f"arm_joints={self._arm_joint_indices}, "
            f"gripper={self._gripper_joint_index}, "
            f"tcp_link={self._tcp_link_index})"
        )

