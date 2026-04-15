from dataclasses import dataclass
from typing import Optional

import numpy as np
import tyro

SAFE_JOINTS_RAD = np.array(
    [1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 1.5708], dtype=float
)
DEFAULT_ROBOT_IP = "158.132.172.214"


@dataclass
class Args:
    robot_ip: str = DEFAULT_ROBOT_IP
    velocity: float = 0.5
    acceleration: float = 0.2
    tolerance: float = 0.05
    dry_run: bool = False
    skip_arm: bool = False
    activate_gripper: bool = False
    gripper_position: Optional[float] = None
    gripper_speed: int = 255
    gripper_force: int = 10


def format_joints(joints: np.ndarray) -> str:
    return "[" + ", ".join(f"{joint:.4f}" for joint in joints) + "]"


def clamp_gripper_position(position: float) -> float:
    return max(0.0, min(1.0, position))


def main(args: Args) -> None:
    control = None

    try:
        if not args.skip_arm:
            import rtde_control
            import rtde_receive

            receive = rtde_receive.RTDEReceiveInterface(args.robot_ip)
            control = rtde_control.RTDEControlInterface(args.robot_ip)

            current_joints = np.asarray(receive.getActualQ(), dtype=float)
            delta = SAFE_JOINTS_RAD - current_joints

            print(f"Connected to UR3 at {args.robot_ip}")
            print(f"Current joints (rad): {format_joints(current_joints)}")
            print(f"Target joints  (rad): {format_joints(SAFE_JOINTS_RAD)}")
            print(f"Target joints  (deg): {format_joints(np.rad2deg(SAFE_JOINTS_RAD))}")
            print(
                "Max joint delta: "
                f"{np.abs(delta).max():.4f} rad / {np.rad2deg(np.abs(delta).max()):.2f} deg"
            )

            if args.dry_run:
                print("Dry run only, not sending moveJ.")
            else:
                print(
                    f"Sending moveJ with velocity={args.velocity:.3f}, acceleration={args.acceleration:.3f}"
                )
                result = control.moveJ(
                    SAFE_JOINTS_RAD.tolist(), args.velocity, args.acceleration
                )
                if result is False:
                    raise RuntimeError("RTDE moveJ returned False")

                final_joints = np.asarray(receive.getActualQ(), dtype=float)
                final_error = np.abs(final_joints - SAFE_JOINTS_RAD)

                print(f"Final joints   (rad): {format_joints(final_joints)}")
                print(
                    "Final max error: "
                    f"{final_error.max():.4f} rad / {np.rad2deg(final_error.max()):.2f} deg"
                )

                if final_error.max() > args.tolerance:
                    print(
                        "Warning: final joint error is larger than tolerance. "
                        "Check the robot state on the pendant."
                    )
                else:
                    print("UR3 reset to safe joint position.")

        if args.activate_gripper or args.gripper_position is not None:
            from gello.robots.robotiq_gripper import RobotiqGripper

            gripper = RobotiqGripper()
            gripper.connect(hostname=args.robot_ip, port=63352)
            print(f"Connected to gripper at {args.robot_ip}:63352")

            current_raw = gripper.get_current_position()
            print(
                f"Current gripper position: raw={current_raw}, normalized={current_raw / 255.0:.3f}"
            )

            if args.activate_gripper:
                if args.dry_run:
                    print("Dry run only, not activating gripper.")
                else:
                    print("Activating gripper...")
                    gripper.activate(auto_calibrate=False)
                    print("Gripper activated.")

            if args.gripper_position is not None:
                gripper_position = clamp_gripper_position(args.gripper_position)
                raw_target = int(round(gripper_position * 255))
                print(
                    "Target gripper position: "
                    f"normalized={gripper_position:.3f}, raw={raw_target}, "
                    f"speed={args.gripper_speed}, force={args.gripper_force}"
                )
                if args.dry_run:
                    print("Dry run only, not sending gripper command.")
                else:
                    final_raw, status = gripper.move_and_wait_for_pos(
                        raw_target, args.gripper_speed, args.gripper_force
                    )
                    print(
                        "Final gripper position: "
                        f"raw={final_raw}, normalized={final_raw / 255.0:.3f}, status={status.name}"
                    )

        if args.skip_arm and not args.activate_gripper and args.gripper_position is None:
            print("Nothing to do: arm reset skipped and no gripper action requested.")
    finally:
        if control is not None:
            control.stopScript()


if __name__ == "__main__":
    main(tyro.cli(Args))
