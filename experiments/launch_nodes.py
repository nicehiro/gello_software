from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro

from gello.robots.robot import BimanualRobot, PrintRobot
from gello.zmq_core.robot_node import ZMQServerRobot


@dataclass
class Args:
    robot: str = "xarm"
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    robot_ip: str = "192.168.1.10"
    no_gripper: bool = False
    enable_collision_filter: bool = True
    collision_debug: bool = False
    collision_margin: float = 0.0
    collision_line_search_steps: int = 10
    collision_max_joint_step: float = 0.05
    collision_urdf_path: Optional[str] = None
    table_height: float = 0.0
    table_wall_height: float = 0.05
    enable_cameras: bool = False
    eye_in_hand_camera_port: int = 5000
    agentview_camera_port: int = 5001
    sideview_camera_port: int = 5002
    camera_width: int = 128
    camera_height: int = 128
    show_camera_windows: bool = False


def launch_robot_server(args: Args):
    port = args.robot_port
    if args.robot == "sim_ur":
        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "universal_robots_ur5e" / "ur5e.xml"
        gripper_xml = MENAGERIE_ROOT / "robotiq_2f85" / "2f85.xml"
        from gello.robots.sim_robot import MujocoRobotServer

        server = MujocoRobotServer(
            xml_path=xml, gripper_xml_path=gripper_xml, port=port, host=args.hostname
        )
        server.serve()
    elif args.robot == "sim_ur3":
        from gello.robots.ur3_mujoco_server import UR3MujocoServer
        from gello.safety.ur3_self_collision import UR3SelfCollisionChecker

        if args.collision_urdf_path is None:
            urdf_path = (
                Path(__file__).parent.parent
                / "third_party"
                / "ur_description"
                / "urdf"
                / "ur3.urdf"
            )
            package_dir = Path(__file__).parent.parent / "third_party"
        else:
            urdf_path = Path(args.collision_urdf_path)
            package_dir = urdf_path.resolve().parents[2]
        collision_checker = UR3SelfCollisionChecker(
            urdf_path=urdf_path,
            package_dir=package_dir,
            collision_margin=args.collision_margin,
            line_search_steps=args.collision_line_search_steps,
            max_joint_step=args.collision_max_joint_step,
            table_height=args.table_height,
            table_wall_height=args.table_wall_height,
        )

        server = UR3MujocoServer(
            port=port,
            host=args.hostname,
            collision_checker=collision_checker,
            collision_debug=args.collision_debug,
            enable_cameras=args.enable_cameras,
            eye_in_hand_camera_port=args.eye_in_hand_camera_port,
            agentview_camera_port=args.agentview_camera_port,
            sideview_camera_port=args.sideview_camera_port,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            show_camera_windows=args.show_camera_windows,
            table_height=args.table_height,
            table_wall_height=args.table_wall_height,
        )
        server.serve()
    elif args.robot == "sim_yam":
        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "i2rt_yam" / "yam.xml"
        from gello.robots.sim_robot import MujocoRobotServer

        server = MujocoRobotServer(
            xml_path=xml, gripper_xml_path=None, port=port, host=args.hostname
        )
        server.serve()
    elif args.robot == "sim_panda":
        from gello.robots.sim_robot import MujocoRobotServer

        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "franka_emika_panda" / "panda.xml"
        gripper_xml = None
        server = MujocoRobotServer(
            xml_path=xml, gripper_xml_path=gripper_xml, port=port, host=args.hostname
        )
        server.serve()
    elif args.robot == "sim_xarm":
        from gello.robots.sim_robot import MujocoRobotServer

        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "ufactory_xarm7" / "xarm7.xml"
        gripper_xml = None
        server = MujocoRobotServer(
            xml_path=xml, gripper_xml_path=gripper_xml, port=port, host=args.hostname
        )
        server.serve()

    else:
        if args.robot == "xarm":
            from gello.robots.xarm_robot import XArmRobot

            robot = XArmRobot(ip=args.robot_ip)
        elif args.robot == "ur":
            from gello.robots.ur import URRobot

            robot = URRobot(robot_ip=args.robot_ip, no_gripper=args.no_gripper)
            if args.enable_collision_filter:
                from gello.robots.collision_filtered_robot import CollisionFilteredRobot
                from gello.safety.ur3_self_collision import UR3SelfCollisionChecker

                if args.collision_urdf_path is None:
                    urdf_path = (
                        Path(__file__).parent.parent
                        / "third_party"
                        / "ur_description"
                        / "urdf"
                        / "ur3.urdf"
                    )
                    package_dir = Path(__file__).parent.parent / "third_party"
                else:
                    urdf_path = Path(args.collision_urdf_path)
                    package_dir = urdf_path.resolve().parents[2]

                collision_checker = UR3SelfCollisionChecker(
                    urdf_path=urdf_path,
                    package_dir=package_dir,
                    collision_margin=args.collision_margin,
                    line_search_steps=args.collision_line_search_steps,
                    max_joint_step=args.collision_max_joint_step,
                    table_height=args.table_height,
                    table_wall_height=args.table_wall_height,
                )
                robot = CollisionFilteredRobot(
                    robot,
                    collision_checker=collision_checker,
                    debug=args.collision_debug,
                )
        elif args.robot == "panda":
            from gello.robots.panda import PandaRobot

            robot = PandaRobot(robot_ip=args.robot_ip)
        elif args.robot == "bimanual_ur":
            from gello.robots.ur import URRobot

            # IP for the bimanual robot setup is hardcoded
            _robot_l = URRobot(robot_ip="192.168.2.10")
            _robot_r = URRobot(robot_ip="192.168.1.10")
            robot = BimanualRobot(_robot_l, _robot_r)
        elif args.robot == "yam":
            from gello.robots.yam import YAMRobot

            robot = YAMRobot(channel="can0")
        elif args.robot == "none" or args.robot == "print":
            robot = PrintRobot(8)

        else:
            raise NotImplementedError(
                f"Robot {args.robot} not implemented, choose one of: sim_ur, xarm, ur, bimanual_ur, none"
            )
        server = ZMQServerRobot(robot, port=port, host=args.hostname)
        print(f"Starting robot server on port {port}")
        server.serve()


def main(args):
    launch_robot_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
