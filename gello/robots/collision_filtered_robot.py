from typing import Any, Dict

import numpy as np

from gello.robots.robot import Robot
from gello.safety.collision_checker import CollisionChecker


class CollisionFilteredRobot(Robot):
    def __init__(
        self,
        robot: Robot,
        collision_checker: CollisionChecker,
        debug: bool = False,
    ) -> None:
        self._robot = robot
        self._collision_checker = collision_checker
        self._debug = debug

    def __getattr__(self, name: str) -> Any:
        return getattr(self._robot, name)

    def num_dofs(self) -> int:
        return self._robot.num_dofs()

    def get_joint_state(self) -> np.ndarray:
        return self._robot.get_joint_state()

    def get_observations(self) -> Dict[str, np.ndarray]:
        return self._robot.get_observations()

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        joint_state = np.asarray(joint_state, dtype=float)
        expected_shape = (self._robot.num_dofs(),)
        if joint_state.shape != expected_shape:
            raise ValueError(
                f"Expected joint_state shape {expected_shape}, got {joint_state.shape}"
            )

        arm_dofs = self._collision_checker.arm_dofs
        desired_arm = joint_state[:arm_dofs]
        if hasattr(self._robot, "r_inter"):
            current_arm = np.asarray(self._robot.r_inter.getActualQ(), dtype=float)
        else:
            current_arm = np.asarray(self._robot.get_joint_state(), dtype=float)[:arm_dofs]
        safe_arm = self._collision_checker.project_to_safe(current_arm, desired_arm)

        safe_joint_state = joint_state.copy()
        safe_joint_state[:arm_dofs] = safe_arm

        if self._debug and not np.allclose(safe_arm, desired_arm):
            result = self._collision_checker.check(desired_arm)
            print(
                "Self-collision filter clipped UR3 command:",
                {
                    "current": np.round(current_arm, 4).tolist(),
                    "desired": np.round(desired_arm, 4).tolist(),
                    "safe": np.round(safe_arm, 4).tolist(),
                    "pairs": list(result.collision_pairs),
                    "minimum_distance": result.minimum_distance,
                },
            )

        self._robot.command_joint_state(safe_joint_state)
