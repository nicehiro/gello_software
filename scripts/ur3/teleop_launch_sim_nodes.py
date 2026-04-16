import sys
from dataclasses import dataclass
from pathlib import Path

import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.launch_nodes import Args as LaunchNodesArgs
from experiments.launch_nodes import main as launch_nodes_main


@dataclass
class Args:
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    collision_debug: bool = False
    collision_margin: float = 0.01
    collision_line_search_steps: int = 10
    collision_max_joint_step: float = 0.05
    collision_urdf_path: str | None = None
    table_height: float = 0.0
    table_wall_height: float = 0.01
    enable_cameras: bool = True
    eye_in_hand_camera_port: int = 5000
    agentview_camera_port: int = 5001
    sideview_camera_port: int = 5002
    camera_width: int = 128
    camera_height: int = 128
    show_camera_windows: bool = False


def main(args: Args) -> None:
    launch_nodes_main(
        LaunchNodesArgs(
            robot="sim_ur3",
            robot_port=args.robot_port,
            hostname=args.hostname,
            collision_debug=args.collision_debug,
            collision_margin=args.collision_margin,
            collision_line_search_steps=args.collision_line_search_steps,
            collision_max_joint_step=args.collision_max_joint_step,
            collision_urdf_path=args.collision_urdf_path,
            table_height=args.table_height,
            table_wall_height=args.table_wall_height,
            enable_cameras=args.enable_cameras,
            eye_in_hand_camera_port=args.eye_in_hand_camera_port,
            agentview_camera_port=args.agentview_camera_port,
            sideview_camera_port=args.sideview_camera_port,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            show_camera_windows=args.show_camera_windows,
        )
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
