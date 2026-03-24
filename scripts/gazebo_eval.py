#!/usr/bin/env python3
"""
Evaluate trained policy LIVE in Gazebo.

This loads a trained SB3 model, connects to a running Gazebo simulation
via ROS2, and drives the robot arm using the learned policy. You can
watch the robot perform reach/pick-and-place tasks in real-time 3D.

Prerequisites:
    1. Gazebo simulation running with the MCAD arm
    2. ROS2 workspace sourced
    3. Controllers active (arm + gripper)

Usage:
    # Terminal 1: Launch Gazebo
    ros2 launch mcad_bringup mcad_master_bringup.launch.py

    # Terminal 2: Run this script (with venv AND ROS2 sourced)
    python scripts/gazebo_eval.py --model checkpoints/<run>/best/best_model.zip --task reach
    python scripts/gazebo_eval.py --model checkpoints/<run>/best/best_model.zip --task pick
"""
import os
import sys
import argparse
import numpy as np
import time

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from geometry_msgs.msg import PoseStamped
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

from stable_baselines3 import SAC, PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcad_arm_rl.config import RobotConfig, EnvConfig


class GazeboEvalNode(Node):
    """
    ROS2 node that runs a trained RL policy and drives the robot in Gazebo.
    Provides a live 3D visualization of the trained policy's behavior.
    """

    def __init__(self, model, task: str, config: RobotConfig, env_config: EnvConfig):
        super().__init__("gazebo_rl_evaluator")
        self.model = model
        self.task = task
        self.config = config
        self.env_config = env_config

        self.joint_positions = np.zeros(6, dtype=np.float32)
        self.joint_velocities = np.zeros(6, dtype=np.float32)
        self.gripper_position = 0.0
        self.got_state = False
        self.step_count = 0
        self.episode_count = 0

        # Target position (can be updated via topic or parameter)
        self.target_pos = np.array([0.2, 0.0, 0.15], dtype=np.float32)
        self.block_pos = np.array([0.2, 0.0, 0.015], dtype=np.float32)
        self.place_target = np.array([0.15, 0.1, 0.015], dtype=np.float32)

        # Joint names
        self.arm_joint_names = [
            "r2_joint1", "r2_joint2", "r2_joint3",
            "r2_joint4", "r2_joint5", "r2_joint6",
        ]
        self.gripper_joint_name = "r2_joint7"

        # Subscribe to joint states
        self.create_subscription(
            JointState, "/joint_states", self.joint_state_cb, 10,
        )

        # Subscribe to target position (optional)
        self.create_subscription(
            PoseStamped, "/rl_target", self.target_cb, 10,
        )

        # Publishers
        self.arm_pub = self.create_publisher(
            JointTrajectory,
            "/arm_r2_controller/joint_trajectory",
            10,
        )
        self.gripper_pub = self.create_publisher(
            JointTrajectory,
            "/gripper_r2_controller/joint_trajectory",
            10,
        )

        # Control loop at 20Hz (matches training freq)
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info(f"✅ Gazebo RL Evaluator ready (task={task})")
        self.get_logger().info(f"   Target: {self.target_pos}")
        self.get_logger().info(f"   Publishing to /arm_r2_controller + /gripper_r2_controller")

    def joint_state_cb(self, msg):
        """Read current joint positions from Gazebo."""
        for i, name in enumerate(self.arm_joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                self.joint_positions[i] = msg.position[idx]
                if idx < len(msg.velocity):
                    self.joint_velocities[i] = msg.velocity[idx]

        if self.gripper_joint_name in msg.name:
            idx = msg.name.index(self.gripper_joint_name)
            self.gripper_position = msg.position[idx]

        self.got_state = True

    def target_cb(self, msg):
        """Update target from external topic (e.g., from GUI or planning)."""
        self.target_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=np.float32)
        self.get_logger().info(f"Target updated: {self.target_pos}")

    def _build_obs_reach(self) -> np.ndarray:
        """Build 18D observation for ReachEnv."""
        joint_range = self.config.joint_upper - self.config.joint_lower
        joint_mid = (self.config.joint_upper + self.config.joint_lower) / 2
        norm_pos = (self.joint_positions - joint_mid) / (joint_range / 2 + 1e-8)
        clipped_vel = np.clip(self.joint_velocities, -5.0, 5.0) / 5.0

        # Approximate EE position (in real deployment, get from TF)
        ee_pos = np.zeros(3, dtype=np.float32)

        return np.concatenate([
            norm_pos, clipped_vel, ee_pos, self.target_pos,
        ]).astype(np.float32)

    def _build_obs_pick(self) -> np.ndarray:
        """Build 25D observation for PickPlaceEnv."""
        joint_range = self.config.joint_upper - self.config.joint_lower
        joint_mid = (self.config.joint_upper + self.config.joint_lower) / 2
        norm_pos = (self.joint_positions - joint_mid) / (joint_range / 2 + 1e-8)
        norm_grip = np.array([self.gripper_position / 0.65], dtype=np.float32)
        clipped_vel = np.clip(self.joint_velocities, -5.0, 5.0) / 5.0
        grip_vel = np.array([0.0], dtype=np.float32)

        ee_pos = np.zeros(3, dtype=np.float32)
        grasp_flag = np.array([0.0], dtype=np.float32)

        return np.concatenate([
            norm_pos, norm_grip,
            clipped_vel, grip_vel,
            ee_pos, self.block_pos, self.place_target,
            norm_grip, grasp_flag,
        ]).astype(np.float32)

    def control_loop(self):
        """Run one policy inference step and publish commands."""
        if not self.got_state:
            return

        # Build observation based on task
        if self.task == "pick":
            obs = self._build_obs_pick()
        else:
            obs = self._build_obs_reach()

        # Get action from trained policy
        action, _ = self.model.predict(obs, deterministic=True)

        # Compute target joint positions
        delta = action[:6] * self.env_config.action_scale
        target_pos = self.joint_positions + delta
        target_pos = np.clip(target_pos, self.config.joint_lower, self.config.joint_upper)

        # Publish arm trajectory
        arm_msg = JointTrajectory()
        arm_msg.joint_names = self.arm_joint_names
        point = JointTrajectoryPoint()
        point.positions = target_pos.tolist()
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(0.05 * 1e9)
        arm_msg.points.append(point)
        self.arm_pub.publish(arm_msg)

        # Publish gripper command (for pick task)
        if self.task == "pick" and len(action) > 6:
            grip_delta = float(action[6]) * 0.1
            target_grip = np.clip(
                self.gripper_position + grip_delta,
                self.config.gripper_limits[0],
                self.config.gripper_limits[1],
            )
            grip_msg = JointTrajectory()
            grip_msg.joint_names = [self.gripper_joint_name]
            grip_pt = JointTrajectoryPoint()
            grip_pt.positions = [float(target_grip)]
            grip_pt.time_from_start.sec = 0
            grip_pt.time_from_start.nanosec = int(0.05 * 1e9)
            grip_msg.points.append(grip_pt)
            self.gripper_pub.publish(grip_msg)

        self.step_count += 1
        if self.step_count % 100 == 0:
            self.get_logger().info(
                f"Step {self.step_count}: "
                f"joints={np.round(target_pos, 3).tolist()}, "
                f"target={self.target_pos.tolist()}"
            )


def main():
    if not HAS_ROS2:
        print("=" * 60)
        print("  ERROR: ROS2 not found!")
        print("=" * 60)
        print("\nTo use Gazebo evaluation, source your ROS2 workspace first:")
        print("  source /opt/ros/jazzy/setup.bash")
        print("  source ~/your_ws/install/setup.bash")
        print("\nAlternatively, use PyBullet visualization:")
        print("  python scripts/visualize.py --model <model_path>")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Evaluate RL policy in Gazebo")
    parser.add_argument("--model", type=str, required=True, help="Path to model .zip")
    parser.add_argument("--algo", type=str, default="sac", choices=["sac", "ppo"])
    parser.add_argument("--task", type=str, default="reach", choices=["reach", "pick"])
    parser.add_argument("--target-x", type=float, default=0.2)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--target-z", type=float, default=0.15)
    args = parser.parse_args()

    print(f"Loading {args.algo.upper()} model from: {args.model}")
    ModelClass = SAC if args.algo == "sac" else PPO
    model = ModelClass.load(args.model)

    rclpy.init()
    config = RobotConfig()
    env_config = EnvConfig()
    node = GazeboEvalNode(model, args.task, config, env_config)
    node.target_pos = np.array([args.target_x, args.target_y, args.target_z], dtype=np.float32)

    print("\n" + "=" * 60)
    print(f"  🤖 Running {args.task.upper()} policy in Gazebo")
    print(f"  Target: ({args.target_x}, {args.target_y}, {args.target_z})")
    print(f"  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n✅ Evaluation stopped.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
