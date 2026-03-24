#!/usr/bin/env python3
"""
Deploy trained RL policy to ROS2 (Gazebo or real robot).

This script loads a trained SB3 model and publishes joint commands
to the arm controller via ROS2 topics.

Usage (after sourcing ROS2 workspace):
    python scripts/ros2_deploy.py --model checkpoints/sac_*/best/best_model.zip
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
except ImportError:
    print("ERROR: rclpy not found. Source your ROS2 workspace first:")
    print("  source /opt/ros/jazzy/setup.bash")
    print("  source ~/your_workspace/install/setup.bash")
    sys.exit(1)

from stable_baselines3 import SAC, PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcad_arm_rl.config import RobotConfig, EnvConfig


class PolicyDeployerNode(Node):
    """ROS2 node that runs a trained RL policy and publishes joint commands."""

    def __init__(self, model, algo: str, config: RobotConfig, env_config: EnvConfig):
        super().__init__("rl_policy_deployer")
        self.model = model
        self.config = config
        self.env_config = env_config

        self.joint_positions = np.zeros(6, dtype=np.float32)
        self.joint_velocities = np.zeros(6, dtype=np.float32)
        self.got_state = False

        # Target position (set this via a parameter or topic)
        self.target_pos = np.array([0.2, 0.0, 0.15], dtype=np.float32)

        # Map from ROS joint names → index
        self.joint_names = [
            "r2_joint1", "r2_joint2", "r2_joint3",
            "r2_joint4", "r2_joint5", "r2_joint6",
        ]

        # Subscribers
        self.create_subscription(
            JointState, "/joint_states", self.joint_state_cb, 10,
        )

        # Publisher
        self.cmd_pub = self.create_publisher(
            JointTrajectory,
            "/arm_r2_controller/joint_trajectory",
            10,
        )

        # Control loop at 20Hz (matches training)
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("✅ RL Policy Deployer ready. Publishing to /arm_r2_controller")

    def joint_state_cb(self, msg):
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                self.joint_positions[i] = msg.position[idx]
                if idx < len(msg.velocity):
                    self.joint_velocities[i] = msg.velocity[idx]
        self.got_state = True

    def control_loop(self):
        if not self.got_state:
            return

        # Build observation (same format as training env)
        joint_range = self.config.joint_upper - self.config.joint_lower
        joint_mid = (self.config.joint_upper + self.config.joint_lower) / 2
        norm_pos = (self.joint_positions - joint_mid) / (joint_range / 2 + 1e-8)
        clipped_vel = np.clip(self.joint_velocities, -5.0, 5.0) / 5.0

        # We don't have FK in ROS2 mode, so we use [0,0,0] as placeholder.
        # In practice, you'd compute FK or subscribe to a TF.
        ee_pos = np.zeros(3, dtype=np.float32)

        obs = np.concatenate([
            norm_pos, clipped_vel, ee_pos, self.target_pos,
        ]).astype(np.float32)

        # Get action from trained policy
        action, _ = self.model.predict(obs, deterministic=True)

        # Convert action to target positions
        delta = action * self.env_config.action_scale
        target_pos = self.joint_positions + delta
        target_pos = np.clip(target_pos, self.config.joint_lower, self.config.joint_upper)

        # Publish trajectory
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = target_pos.tolist()
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(0.05 * 1e9)
        msg.points.append(point)
        self.cmd_pub.publish(msg)


def main():
    parser = argparse.ArgumentParser(description="Deploy RL policy to ROS2")
    parser.add_argument("--model", type=str, required=True, help="Path to model .zip")
    parser.add_argument("--algo", type=str, default="sac", choices=["sac", "ppo"])
    parser.add_argument("--target-x", type=float, default=0.2)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--target-z", type=float, default=0.15)
    args = parser.parse_args()

    ModelClass = SAC if args.algo == "sac" else PPO
    model = ModelClass.load(args.model)

    rclpy.init()
    config = RobotConfig()
    env_config = EnvConfig()
    node = PolicyDeployerNode(model, args.algo, config, env_config)
    node.target_pos = np.array([args.target_x, args.target_y, args.target_z], dtype=np.float32)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
