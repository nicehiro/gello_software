from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UR3_URDF_PATH = PROJECT_ROOT / "third_party" / "ur_description" / "urdf" / "ur3.urdf"
DEFAULT_UR3_PACKAGE_DIR = PROJECT_ROOT / "third_party"
UR3_ARM_DOFS = 6

DEFAULT_MAX_JOINT_STEP = 0.05
DEFAULT_COLLISION_MARGIN = 0.0
DEFAULT_LINE_SEARCH_STEPS = 10
DEFAULT_TABLE_HEIGHT = 0.0
DEFAULT_TABLE_WALL_HEIGHT = 0.05

DEFAULT_IGNORED_COLLISION_PAIRS: tuple[tuple[str, str], ...] = (
    ("base_link_inertia_0", "shoulder_link_0"),
    ("shoulder_link_0", "upper_arm_link_0"),
    ("upper_arm_link_0", "forearm_link_0"),
    ("forearm_link_0", "wrist_1_link_0"),
    ("wrist_1_link_0", "wrist_2_link_0"),
    ("wrist_2_link_0", "wrist_3_link_0"),
)

DEFAULT_TABLE_MONITORED_POINTS: tuple[tuple[str, str, tuple[float, float, float]], ...] = (
    ("wrist_2_clearance", "wrist_2_link", (0.0, 0.0, 0.0)),
    ("wrist_3_clearance", "wrist_3_link", (0.0, 0.0, 0.0)),
    ("pinch_proxy", "wrist_3_link", (0.0, 0.0, 0.1488)),
)
