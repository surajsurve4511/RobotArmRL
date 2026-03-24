"""Test forward kinematics returns reasonable positions."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pybullet as p
import pybullet_data

from mcad_arm_rl.robot import MCadRobot


def test_home_fk():
    """At home position (all zeros), tool0 should be roughly above the base."""
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # No gravity — we just want to test pure FK, not dynamics
    p.setGravity(0, 0, 0, physicsClientId=client)

    robot = MCadRobot(physics_client=client)
    robot.reset()

    # Read FK immediately after reset (resetJointState is instantaneous)
    ee_pos = robot.get_end_effector_pos()
    print(f"End-effector at home: ({ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f})")

    # At home (all zeros), the kinematic chain places tool0 forward and up
    # Just verify the FK returns a reasonable non-zero position
    dist_from_base = np.linalg.norm(ee_pos)
    assert dist_from_base > 0.05, f"tool0 too close to base: dist={dist_from_base:.4f}"
    print(f"✅ tool0 distance from base: {dist_from_base:.4f}m (> 0.05m)")

    p.disconnect(client)


def test_fk_changes_with_joints():
    """Moving joints should change end-effector position."""
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0, physicsClientId=client)

    robot = MCadRobot(physics_client=client)

    # Position 1: home
    robot.reset()
    ee_home = robot.get_end_effector_pos().copy()

    # Position 2: move shoulder forward
    robot.set_joint_positions(np.array([0.0, 0.5, 0.3, 0.0, 0.0, 0.0]))
    ee_moved = robot.get_end_effector_pos()

    dist = np.linalg.norm(ee_moved - ee_home)
    print(f"Home EE:  ({ee_home[0]:.4f}, {ee_home[1]:.4f}, {ee_home[2]:.4f})")
    print(f"Moved EE: ({ee_moved[0]:.4f}, {ee_moved[1]:.4f}, {ee_moved[2]:.4f})")
    print(f"Distance: {dist:.4f}m")

    assert dist > 0.01, f"FK should change with joint movement, distance={dist}"
    print(f"✅ FK changes correctly: Δ={dist:.4f}m")

    p.disconnect(client)


if __name__ == "__main__":
    test_home_fk()
    test_fk_changes_with_joints()
    print("\n✅ All FK tests passed!")
