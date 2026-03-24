"""Test that the robot URDF loads correctly in PyBullet."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pybullet as p
import pybullet_data

from mcad_arm_rl.config import ROBOT_URDF, RobotConfig
from mcad_arm_rl.robot import MCadRobot


def test_urdf_loads():
    """URDF should load without errors."""
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    robot = MCadRobot(physics_client=client)
    print(f"Robot loaded: {robot}")

    assert robot.n_arm_joints == 6, f"Expected 6 arm joints, got {robot.n_arm_joints}"
    print(f"✅ Found {robot.n_arm_joints} arm joints: {robot.arm_joint_indices}")

    p.disconnect(client)


def test_joint_count():
    """Should find all expected joints by name."""
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    config = RobotConfig()
    robot = MCadRobot(physics_client=client, config=config)

    n_total = p.getNumJoints(robot.robot_id, physicsClientId=client)
    print(f"Total joints in URDF: {n_total}")

    # Print all joint names for debugging
    for i in range(n_total):
        info = p.getJointInfo(robot.robot_id, i, physicsClientId=client)
        print(f"  Joint {i}: name={info[1].decode()}, type={info[2]}, link={info[12].decode()}")

    assert robot.n_arm_joints == 6
    p.disconnect(client)


if __name__ == "__main__":
    test_urdf_loads()
    test_joint_count()
    print("\n✅ All URDF tests passed!")
