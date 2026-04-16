from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class CollisionCheckResult:
    in_collision: bool
    collision_pairs: tuple[tuple[str, str], ...] = ()
    minimum_distance: float | None = None


class CollisionChecker(Protocol):
    arm_dofs: int

    def check(self, joints: np.ndarray) -> CollisionCheckResult: ...

    def is_state_safe(self, joints: np.ndarray) -> bool: ...

    def project_to_safe(
        self, current_joints: np.ndarray, desired_joints: np.ndarray
    ) -> np.ndarray: ...
