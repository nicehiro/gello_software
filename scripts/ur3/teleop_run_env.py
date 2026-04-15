from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Tuple

import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_env import Args as RunEnvArgs
from experiments.run_env import main as run_env_main

DEFAULT_GELLO_PORT = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTB8HJD7-if00-port0"
DEFAULT_START_JOINTS = (1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 1.5708, 0.0)


@dataclass
class Args:
    gello_port: str = DEFAULT_GELLO_PORT
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    hz: int = 100
    start_joints: Tuple[float, ...] = DEFAULT_START_JOINTS
    mock: bool = False
    verbose: bool = False
    use_save_interface: bool = False
    data_dir: str = "~/bc_data"


def main(args: Args) -> None:
    run_env_main(
        RunEnvArgs(
            agent="gello",
            robot_port=args.robot_port,
            hostname=args.hostname,
            hz=args.hz,
            start_joints=args.start_joints,
            gello_port=args.gello_port,
            mock=args.mock,
            use_save_interface=args.use_save_interface,
            data_dir=args.data_dir,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
