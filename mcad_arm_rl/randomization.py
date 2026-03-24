"""
Domain randomization for sim-to-real transfer.
Randomizes physics parameters each episode to make policies more robust.
"""
import pybullet as p
import numpy as np
from typing import Optional


class DomainRandomizer:
    """Applies domain randomization to the PyBullet simulation."""

    def __init__(
        self,
        physics_client: int,
        enabled: bool = True,
        mass_range: float = 0.2,        # ±20%
        friction_range: float = 0.3,     # ±30%
        gravity_range: float = 0.05,     # ±5%
        damping_range: float = 0.3,      # ±30%
    ):
        self.client = physics_client
        self.enabled = enabled
        self.mass_range = mass_range
        self.friction_range = friction_range
        self.gravity_range = gravity_range
        self.damping_range = damping_range

        # Store original values on first call
        self._original_masses = {}
        self._original_frictions = {}
        self._initialized = False

    def _store_originals(self, robot_id: int) -> None:
        """Cache original dynamics values from the URDF."""
        n_joints = p.getNumJoints(robot_id, physicsClientId=self.client)
        for i in range(-1, n_joints):  # -1 = base link
            dynamics = p.getDynamicsInfo(robot_id, i, physicsClientId=self.client)
            self._original_masses[i] = dynamics[0]
            self._original_frictions[i] = dynamics[1]
        self._initialized = True

    def randomize(self, robot_id: int, rng: Optional[np.random.Generator] = None) -> None:
        """Apply domain randomization for one episode."""
        if not self.enabled:
            return

        if rng is None:
            rng = np.random.default_rng()

        if not self._initialized:
            self._store_originals(robot_id)

        n_joints = p.getNumJoints(robot_id, physicsClientId=self.client)

        # Randomize link masses and friction
        for i in range(-1, n_joints):
            orig_mass = self._original_masses[i]
            orig_friction = self._original_frictions[i]

            if orig_mass > 0:
                new_mass = orig_mass * (1.0 + rng.uniform(-self.mass_range, self.mass_range))
                p.changeDynamics(
                    robot_id, i,
                    mass=max(new_mass, 0.001),
                    physicsClientId=self.client,
                )

            new_friction = orig_friction * (1.0 + rng.uniform(-self.friction_range, self.friction_range))
            p.changeDynamics(
                robot_id, i,
                lateralFriction=max(new_friction, 0.01),
                physicsClientId=self.client,
            )

        # Randomize gravity
        base_gravity = -9.81
        gravity_noise = base_gravity * rng.uniform(-self.gravity_range, self.gravity_range)
        p.setGravity(0, 0, base_gravity + gravity_noise, physicsClientId=self.client)

    def reset_to_defaults(self, robot_id: int) -> None:
        """Restore original dynamics values."""
        if not self._initialized:
            return

        n_joints = p.getNumJoints(robot_id, physicsClientId=self.client)
        for i in range(-1, n_joints):
            p.changeDynamics(
                robot_id, i,
                mass=self._original_masses[i],
                lateralFriction=self._original_frictions[i],
                physicsClientId=self.client,
            )
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
