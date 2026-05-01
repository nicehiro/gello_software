from typing import Dict

import numpy as np

from gello.robots.robot import Robot


def _rotvec_to_quaternion_xyzw(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=float).reshape(3)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

    axis = rotvec / angle
    half_angle = 0.5 * angle
    sin_half = float(np.sin(half_angle))
    return np.array(
        [
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
            float(np.cos(half_angle)),
        ],
        dtype=float,
    )


class URRobot(Robot):
    """A class representing a UR robot."""

    def __init__(self, robot_ip: str = "192.168.1.10", no_gripper: bool = False):
        import rtde_control
        import rtde_receive

        [print("in ur robot") for _ in range(4)]
        try:
            self.robot = rtde_control.RTDEControlInterface(robot_ip)
        except Exception as e:
            print(e)
            print(robot_ip)
            raise

        self.r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)
        if not no_gripper:
            from gello.robots.robotiq_gripper import RobotiqGripper

            self.gripper = RobotiqGripper()
            self.gripper.connect(hostname=robot_ip, port=63352)
            print("gripper connected")

        [print("connect") for _ in range(4)]

        self._free_drive = False
        self.robot.endFreedriveMode()
        self._use_gripper = not no_gripper

    def num_dofs(self) -> int:
        if self._use_gripper:
            return 7
        return 6

    def _get_gripper_pos(self) -> float:
        import time

        time.sleep(0.01)
        gripper_pos = self.gripper.get_current_position()
        assert 0 <= gripper_pos <= 255, "Gripper position must be between 0 and 255"
        return gripper_pos / 255

    def get_joint_state(self) -> np.ndarray:
        robot_joints = np.asarray(self.r_inter.getActualQ(), dtype=float)
        if self._use_gripper:
            gripper_pos = self._get_gripper_pos()
            return np.concatenate([robot_joints, [gripper_pos]])
        return robot_joints

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        velocity = 0.5
        acceleration = 0.5
        dt = 1.0 / 500
        lookahead_time = 0.2
        gain = 100

        robot_joints = joint_state[:6]
        t_start = self.robot.initPeriod()
        self.robot.servoJ(
            robot_joints, velocity, acceleration, dt, lookahead_time, gain
        )
        if self._use_gripper:
            gripper_pos = joint_state[-1] * 255
            self.gripper.move(gripper_pos, 255, 10)
        self.robot.waitPeriod(t_start)

    def freedrive_enabled(self) -> bool:
        return self._free_drive

    def set_freedrive_mode(self, enable: bool) -> None:
        if enable and not self._free_drive:
            self._free_drive = True
            self.robot.freedriveMode()
        elif not enable and self._free_drive:
            self._free_drive = False
            self.robot.endFreedriveMode()

    def get_observations(self) -> Dict[str, np.ndarray]:
        joint_positions_arm = np.asarray(self.r_inter.getActualQ(), dtype=float)
        joint_velocities_arm = np.asarray(self.r_inter.getActualQd(), dtype=float)
        tcp_pose = np.asarray(self.r_inter.getActualTCPPose(), dtype=float)

        ee_pos = tcp_pose[:3]
        ee_quat = _rotvec_to_quaternion_xyzw(tcp_pose[3:6])

        if self._use_gripper:
            gripper_pos = float(self._get_gripper_pos())
            joint_positions = np.concatenate([joint_positions_arm, [gripper_pos]])
            joint_velocities = np.concatenate([joint_velocities_arm, [0.0]])
            gripper_position = np.array([gripper_pos], dtype=float)
        else:
            joint_positions = joint_positions_arm
            joint_velocities = joint_velocities_arm
            gripper_position = np.array([0.0], dtype=float)

        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "ee_pos_quat": np.concatenate([ee_pos, ee_quat]),
            "gripper_position": gripper_position,
        }


def main():
    robot_ip = "192.168.1.11"
    ur = URRobot(robot_ip, no_gripper=True)
    print(ur)
    ur.set_freedrive_mode(True)
    print(ur.get_observations())


if __name__ == "__main__":
    main()
