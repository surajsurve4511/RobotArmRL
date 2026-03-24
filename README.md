# MCAD Arm RL — Reinforcement Learning for 6DOF Robotic Arm

Standalone PyBullet-based RL training for the MCAD 6DOF robotic arm. No ROS2/Gazebo needed during training — policies train at ~5000 steps/sec and can be deployed back to your ROS2 stack.

## Quick Start

### 1. Install Dependencies
```bash
cd /home/suraj/Suraj/mcad/RL/mcad_arm_rl
pip install -r requirements.txt
```

### 2. Verify Setup
```bash
# Test URDF loads correctly
python tests/test_urdf_load.py

# Test environment works
python tests/test_env.py

# Test forward kinematics
python tests/test_fk.py
```

### 3. Train
```bash
# SAC (recommended for robotics — more sample efficient)
python scripts/train.py --algo sac --timesteps 500000

# PPO (simpler, good baseline)
python scripts/train.py --algo ppo --timesteps 1000000 --n-envs 4

# With domain randomization (for sim-to-real transfer)
python scripts/train.py --algo sac --timesteps 500000 --domain-rand
```

### 4. Monitor Training
```bash
tensorboard --logdir logs/tensorboard
```

### 5. Evaluate
```bash
python scripts/evaluate.py --model checkpoints/<run_name>/best/best_model.zip
```

### 6. Visualize (PyBullet GUI)
```bash
python scripts/visualize.py --model checkpoints/<run_name>/best/best_model.zip --slow
```

### 7. Deploy to ROS2 (Gazebo / Real Robot)
```bash
source /opt/ros/jazzy/setup.bash
source ~/your_ws/install/setup.bash
python scripts/ros2_deploy.py --model checkpoints/<run_name>/best/best_model.zip
```

## Project Structure

```
mcad_arm_rl/
├── assets/                     # Robot description (copied from existing project)
│   ├── mcad_robot.urdf         # Standalone URDF for PyBullet
│   ├── urdf/                   # Original xacro files (reference)
│   ├── meshes/                 # STL/DAE mesh files
│   ├── table.urdf              # Table + block for pick tasks
│   └── target_sphere.urdf      # Green target marker
├── mcad_arm_rl/                # Core Python package
│   ├── config.py               # All hyperparameters & joint limits
│   ├── robot.py                # PyBullet robot wrapper with FK
│   ├── env.py                  # Gymnasium ReachEnv (18D obs, 6D action)
│   ├── rewards.py              # Shaped reward functions
│   └── randomization.py        # Domain randomization
├── scripts/                    # Entry points
│   ├── train.py                # SAC/PPO training with SB3
│   ├── evaluate.py             # Evaluate saved policy
│   ├── visualize.py            # PyBullet GUI visualization
│   └── ros2_deploy.py          # Deploy to ROS2 robot
├── tests/                      # Verification tests
├── logs/                       # TensorBoard logs
├── checkpoints/                # Saved models
├── requirements.txt
└── pyproject.toml
```

## Environment Details

| Property | Value |
|----------|-------|
| **Observation** | 18D: 6 normalized joint pos + 6 normalized joint vel + 3D EE pos + 3D target |
| **Action** | 6D: joint velocity deltas ∈ [-1, 1] |
| **Reward** | `-distance + goal_bonus(10) - action_penalty - velocity_penalty` |
| **Success** | End-effector within 2cm of target |
| **Max Steps** | 100 (5 seconds at 20Hz) |
| **Physics** | 240Hz PyBullet, 12 substeps per action = 20Hz control |
