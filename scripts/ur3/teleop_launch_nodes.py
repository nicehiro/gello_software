from dataclasses import dataclass
from pathlib import Path
import sys

import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.launch_nodes import Args as LaunchNodesArgs
from experiments.launch_nodes import main as launch_nodes_main

DEFAULT_ROBOT_IP = "158.132.172.214"


@dataclass
class Args:
    robot_ip: str = DEFAULT_ROBOT_IP
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    no_gripper: bool = False


def main(args: Args) -> None:
    launch_nodes_main(
        LaunchNodesArgs(
            robot="ur",
            robot_port=args.robot_port,
            hostname=args.hostname,
            robot_ip=args.robot_ip,
            no_gripper=args.no_gripper,
        )
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
