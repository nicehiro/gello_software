import json
import re
import select
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import h5py
import numpy as np
import pyrealsense2 as rs
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gello.agents.gello_agent import GelloAgent
from gello.env import RobotEnv
from gello.zmq_core.robot_node import ZMQClientRobot

DEFAULT_GELLO_PORT = (
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTB8HJD7-if00-port0"
)
DEFAULT_START_JOINTS = (1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 1.5708, 0.0)
CAMERA_NAMES = ("agentview", "eye_in_hand")
_RAW_SEGMENT_PATTERN = re.compile(r"raw_(\d{5})\.hdf5$")


@dataclass
class Args:
    output_dir: str
    gello_port: str = DEFAULT_GELLO_PORT
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    control_hz: int = 100
    record_hz: int = 10
    start_joints: Optional[Tuple[float, ...]] = DEFAULT_START_JOINTS
    ignore_gripper_during_startup: bool = True
    operator: str = ""
    env_name: str = "UR3RealWorld"
    schema_version: str = "ur3_raw_segment_v1"
    robot_name: str = "ur3"
    gripper_name: str = "robotiq_2f"
    teleop_device: str = "gello"
    compression: str = "gzip"
    filename_prefix: str = "raw"
    agentview_serial: Optional[str] = None
    eye_in_hand_serial: Optional[str] = None
    width: int = 640
    height: int = 480
    fps: int = 30
    warmup_frames: int = 30
    camera_timeout_ms: int = 2000
    agentview_flip: bool = False
    eye_in_hand_flip: bool = False


@dataclass
class SegmentBuffer:
    timestamps: list[float] = field(default_factory=list)
    agentview_rgb: list[np.ndarray] = field(default_factory=list)
    eye_in_hand_rgb: list[np.ndarray] = field(default_factory=list)
    joint_states: list[np.ndarray] = field(default_factory=list)
    joint_velocities: list[np.ndarray] = field(default_factory=list)
    gripper_states: list[np.ndarray] = field(default_factory=list)
    ee_pos: list[np.ndarray] = field(default_factory=list)
    ee_ori: list[np.ndarray] = field(default_factory=list)
    ee_states: list[np.ndarray] = field(default_factory=list)
    commanded_joint_positions: list[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.timestamps)

    def append(
        self,
        obs: dict[str, np.ndarray],
        command: np.ndarray,
        timestamp: float,
    ) -> None:
        self.timestamps.append(float(timestamp))
        self.agentview_rgb.append(
            np.array(obs["agentview_rgb"], dtype=np.uint8, copy=True)
        )
        self.eye_in_hand_rgb.append(
            np.array(obs["eye_in_hand_rgb"], dtype=np.uint8, copy=True)
        )

        joint_positions = np.array(
            obs["joint_positions"], dtype=np.float32, copy=True
        ).reshape(-1)
        joint_velocities = np.array(
            obs["joint_velocities"], dtype=np.float32, copy=True
        ).reshape(-1)
        self.joint_states.append(joint_positions)
        self.joint_velocities.append(joint_velocities)

        gripper_state = np.array(
            obs["gripper_position"], dtype=np.float32, copy=True
        ).reshape(-1)
        if gripper_state.size != 1:
            raise ValueError(
                f"Expected scalar gripper position, got shape {gripper_state.shape}."
            )
        self.gripper_states.append(gripper_state)

        ee_pos_quat = np.array(obs["ee_pos_quat"], dtype=np.float64, copy=True).reshape(
            -1
        )
        if ee_pos_quat.size != 7:
            raise ValueError(
                f"Expected ee_pos_quat with 7 values, got shape {ee_pos_quat.shape}."
            )
        ee_pos = ee_pos_quat[:3].astype(np.float32)
        ee_ori = ee_pos_quat[3:].astype(np.float32)
        self.ee_pos.append(ee_pos)
        self.ee_ori.append(ee_ori)
        self.ee_states.append(np.concatenate([ee_pos, ee_ori]).astype(np.float32))
        self.commanded_joint_positions.append(
            np.array(command, dtype=np.float32, copy=True)
        )


def normalize_quaternion_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


def quaternion_conjugate_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return np.array([-quat[0], -quat[1], -quat[2], quat[3]], dtype=np.float64)


def quaternion_multiply_xyzw(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = lhs
    x2, y2, z2, w2 = rhs
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def quaternion_to_euler_xyz_xyzw(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternion_xyzw(quat)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float32)


def quaternion_delta_to_euler_xyzw(curr: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    curr = normalize_quaternion_xyzw(curr)
    nxt = normalize_quaternion_xyzw(nxt)
    delta = quaternion_multiply_xyzw(nxt, quaternion_conjugate_xyzw(curr))
    if delta[3] < 0.0:
        delta = -delta
    return quaternion_to_euler_xyz_xyzw(delta)


def compute_derived_ee_action_fields(
    ee_pos: np.ndarray,
    ee_ori: np.ndarray,
    gripper_cmd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_steps = ee_pos.shape[0]
    action_delta_position = np.zeros((num_steps, 3), dtype=np.float32)
    action_delta_euler = np.zeros((num_steps, 3), dtype=np.float32)
    action_gripper = np.asarray(gripper_cmd, dtype=np.float32).reshape(num_steps, 1)

    if num_steps > 1:
        action_delta_position[:-1] = ee_pos[1:] - ee_pos[:-1]
        for idx in range(num_steps - 1):
            action_delta_euler[idx] = quaternion_delta_to_euler_xyzw(
                ee_ori[idx], ee_ori[idx + 1]
            )
        action_delta_position[-1] = action_delta_position[-2]
        action_delta_euler[-1] = action_delta_euler[-2]

    action_7d = np.concatenate(
        [action_delta_position, action_delta_euler, action_gripper], axis=-1
    ).astype(np.float32)
    return action_delta_position, action_delta_euler, action_gripper, action_7d


@dataclass
class CameraInfo:
    serial: str
    intrinsics: np.ndarray
    resolution: tuple[int, int]
    fps: int


class RealSenseStream:
    def __init__(
        self,
        serial: str,
        width: int,
        height: int,
        fps: int,
        flip: bool,
        timeout_ms: int,
    ) -> None:
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.flip = flip
        self.timeout_ms = timeout_ms
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(self.config)

        color_profile = self.profile.get_stream(
            rs.stream.color
        ).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self.info = CameraInfo(
            serial=serial,
            intrinsics=np.array(
                [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            resolution=(width, height),
            fps=fps,
        )

    def read(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames(timeout_ms=self.timeout_ms)
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError(f"Missing color frame for camera {self.serial}")

        color = np.ascontiguousarray(np.asanyarray(color_frame.get_data())[:, :, ::-1])
        if self.flip:
            color = cv2.rotate(color, cv2.ROTATE_180)
        return color

    def warmup(self, num_frames: int) -> None:
        for _ in range(max(num_frames, 0)):
            self.read()

    def close(self) -> None:
        self.pipeline.stop()


class DualRealSenseRig:
    def __init__(self, args: Args) -> None:
        serials = choose_camera_serials(args.agentview_serial, args.eye_in_hand_serial)
        self.streams = {
            "agentview": RealSenseStream(
                serial=serials[0],
                width=args.width,
                height=args.height,
                fps=args.fps,
                flip=args.agentview_flip,
                timeout_ms=args.camera_timeout_ms,
            ),
            "eye_in_hand": RealSenseStream(
                serial=serials[1],
                width=args.width,
                height=args.height,
                fps=args.fps,
                flip=args.eye_in_hand_flip,
                timeout_ms=args.camera_timeout_ms,
            ),
        }
        if args.warmup_frames > 0:
            print(f"Warming up RealSense streams for {args.warmup_frames} frames...")
            for stream in self.streams.values():
                stream.warmup(args.warmup_frames)

    def read(self) -> dict[str, np.ndarray]:
        obs = {}
        for name, stream in self.streams.items():
            obs[f"{name}_rgb"] = stream.read()
        return obs

    def camera_info(self) -> dict[str, CameraInfo]:
        return {name: stream.info for name, stream in self.streams.items()}

    def close(self) -> None:
        for stream in self.streams.values():
            close = getattr(stream, "close", None)
            if close is not None:
                close()


def list_realsense_devices() -> list[dict[str, str]]:
    devices = []
    ctx = rs.context()
    for dev in ctx.query_devices():
        devices.append(
            {
                "name": dev.get_info(rs.camera_info.name),
                "serial": dev.get_info(rs.camera_info.serial_number),
                "firmware": dev.get_info(rs.camera_info.firmware_version),
                "usb": dev.get_info(rs.camera_info.usb_type_descriptor),
            }
        )
    return devices


def choose_camera_serials(
    agentview_serial: Optional[str],
    eye_in_hand_serial: Optional[str],
) -> tuple[str, str]:
    devices = list_realsense_devices()
    if len(devices) < 2:
        raise RuntimeError("Need 2 RealSense devices for real-world collection.")

    print(f"Found {len(devices)} RealSense device(s):")
    for idx, dev in enumerate(devices):
        print(
            f"[{idx}] {dev['name']} | serial={dev['serial']} | "
            f"firmware={dev['firmware']} | usb={dev['usb']}"
        )

    if agentview_serial is None or eye_in_hand_serial is None:
        serials = [dev["serial"] for dev in devices]
        if agentview_serial is None:
            agentview_serial = serials[0]
        if eye_in_hand_serial is None:
            eye_in_hand_serial = serials[1]

    if agentview_serial == eye_in_hand_serial:
        raise ValueError("agentview_serial and eye_in_hand_serial must be different.")

    known_serials = {dev["serial"] for dev in devices}
    missing = [
        serial
        for serial in (agentview_serial, eye_in_hand_serial)
        if serial not in known_serials
    ]
    if missing:
        raise RuntimeError(f"Requested RealSense serial(s) not found: {missing}")

    print(
        f"Camera mapping: agentview={agentview_serial}, eye_in_hand={eye_in_hand_serial}"
    )
    return agentview_serial, eye_in_hand_serial


def freeze_gripper(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    frozen = np.array(target, copy=True)
    if len(frozen) == len(current) and len(frozen) > 0:
        frozen[-1] = current[-1]
    return frozen


def move_robot_to_reset_joints(env: RobotEnv, reset_joints: np.ndarray) -> None:
    current_joints = np.asarray(env.get_obs()["joint_positions"], dtype=np.float64)
    if reset_joints.shape != current_joints.shape:
        return

    max_delta = float(np.abs(current_joints - reset_joints).max())
    steps = max(1, min(int(max_delta / 0.01), 100))
    for joints in np.linspace(current_joints, reset_joints, steps):
        env.step(joints.astype(np.float64))
        time.sleep(0.001)


def align_with_gello(
    env: RobotEnv, agent: GelloAgent, ignore_gripper_during_startup: bool
) -> None:
    print("Aligning real UR3 with GELLO start pose...")
    start_pos = np.asarray(agent.act(env.get_obs()), dtype=np.float64)
    obs = env.get_obs()
    joints = np.asarray(obs["joint_positions"], dtype=np.float64)

    if ignore_gripper_during_startup:
        start_pos = freeze_gripper(start_pos, joints)

    max_joint_delta = 0.8
    abs_deltas = np.abs(start_pos - joints)
    if abs_deltas.max() > max_joint_delta:
        raise RuntimeError(
            "GELLO and robot are too far apart to align safely. "
            f"Max joint delta {abs_deltas.max():.3f} > {max_joint_delta:.3f}."
        )

    for _ in range(25):
        obs = env.get_obs()
        command_joints = np.asarray(agent.act(obs), dtype=np.float64)
        current_joints = np.asarray(obs["joint_positions"], dtype=np.float64)
        if ignore_gripper_during_startup:
            command_joints = freeze_gripper(command_joints, current_joints)
        delta = command_joints - current_joints
        max_delta = np.abs(delta).max()
        if max_delta > 0.05:
            delta = delta / max_delta * 0.05
        env.step(current_joints + delta)


def stdin_line_available() -> bool:
    return bool(select.select([sys.stdin], [], [], 0.0)[0])


def consume_stdin_line() -> str:
    return sys.stdin.readline().rstrip("\n")


def prompt_line(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def prompt_session_action() -> str:
    while True:
        response = prompt_line("[Enter] start new segment, [q] quit session: ")
        if response == "":
            return "start"
        if response.lower() == "q":
            return "quit"
        print("Invalid input. Press Enter to start or q to quit.")


def prompt_save_action() -> str:
    while True:
        response = prompt_line("[Enter] save, [d] discard, [q] quit session: ").lower()
        if response == "":
            return "save"
        if response in {"d", "q"}:
            return response
        print("Invalid input. Press Enter to save, d to discard, or q to quit.")


def summarize_segment(buffer: SegmentBuffer, elapsed_s: float) -> str:
    num_samples = len(buffer)
    avg_saved_hz = num_samples / elapsed_s if elapsed_s > 0 else 0.0
    return (
        f"Recorded {num_samples} samples over {elapsed_s:.2f}s "
        f"(avg saved rate {avg_saved_hz:.2f} Hz)."
    )


def get_full_obs(env: RobotEnv, cameras: DualRealSenseRig) -> dict[str, np.ndarray]:
    obs = env.get_obs()
    obs.update(cameras.read())
    return obs


def record_segment(
    env: RobotEnv,
    agent: GelloAgent,
    cameras: DualRealSenseRig,
    args: Args,
) -> tuple[SegmentBuffer, float]:
    buffer = SegmentBuffer()
    obs = get_full_obs(env, cameras)
    start_time = time.monotonic()
    record_period = 1.0 / float(args.record_hz)
    next_record_time = start_time
    print("Recording... press Enter to stop current segment.")

    while True:
        action = np.asarray(agent.act(obs), dtype=np.float64)

        now = time.monotonic()
        if now >= next_record_time:
            buffer.append(obs, action, now - start_time)
            while next_record_time <= now:
                next_record_time += record_period

        if stdin_line_available():
            consume_stdin_line()
            break

        env.step(action)
        obs = get_full_obs(env, cameras)

    elapsed_s = time.monotonic() - start_time
    return buffer, elapsed_s


def create_dataset(
    group: h5py.Group, name: str, data: np.ndarray, compression: str
) -> None:
    kwargs = {}
    if data.ndim >= 1:
        kwargs["chunks"] = (1, *data.shape[1:])
    if compression:
        kwargs["compression"] = compression
    group.create_dataset(name, data=data, **kwargs)


def write_camera_info(h5_file: h5py.File, camera_info: dict[str, CameraInfo]) -> None:
    root = h5_file.require_group("camera_info")
    for name, info in camera_info.items():
        group = root.require_group(name)
        group.attrs["serial"] = info.serial
        group.attrs["resolution"] = np.asarray(info.resolution, dtype=np.int32)
        group.attrs["fps"] = int(info.fps)
        if "intrinsics" in group:
            del group["intrinsics"]
        group.create_dataset("intrinsics", data=info.intrinsics.astype(np.float32))


def next_raw_segment_path(output_dir: Path, prefix: str) -> Path:
    max_index = -1
    pattern = re.compile(rf"{re.escape(prefix)}_(\d{{5}})\.hdf5$")
    for path in output_dir.glob(f"{prefix}_*.hdf5"):
        match = pattern.fullmatch(path.name)
        if match is not None:
            max_index = max(max_index, int(match.group(1)))
    return output_dir / f"{prefix}_{max_index + 1:05d}.hdf5"


def write_raw_segment(
    dataset_path: Path,
    buffer: SegmentBuffer,
    args: Args,
    camera_info: dict[str, CameraInfo],
) -> None:
    with h5py.File(dataset_path, "w") as h5_file:
        h5_file.attrs["schema_version"] = args.schema_version
        h5_file.attrs["env_name"] = args.env_name
        h5_file.attrs["robot"] = args.robot_name
        h5_file.attrs["gripper"] = args.gripper_name
        h5_file.attrs["teleop_device"] = args.teleop_device
        h5_file.attrs["control_freq"] = int(args.control_hz)
        h5_file.attrs["record_freq"] = int(args.record_hz)
        joint_states = np.stack(buffer.joint_states, axis=0).astype(np.float32)
        joint_velocities = np.stack(buffer.joint_velocities, axis=0).astype(np.float32)
        gripper_states = np.stack(buffer.gripper_states, axis=0).astype(np.float32)
        ee_pos = np.stack(buffer.ee_pos, axis=0).astype(np.float32)
        ee_ori = np.stack(buffer.ee_ori, axis=0).astype(np.float32)
        ee_states = np.stack(buffer.ee_states, axis=0).astype(np.float32)
        timestamps = np.asarray(buffer.timestamps, dtype=np.float64)
        actions_joint_position = np.stack(buffer.commanded_joint_positions, axis=0).astype(
            np.float32
        )
        (
            action_delta_position,
            action_delta_euler,
            action_gripper,
            action_7d,
        ) = compute_derived_ee_action_fields(
            ee_pos,
            ee_ori,
            actions_joint_position[:, -1:],
        )

        h5_file.attrs["action_space"] = "joint_position_raw"
        h5_file.attrs["camera_names"] = json.dumps(list(CAMERA_NAMES))
        h5_file.attrs["operator"] = args.operator
        h5_file.attrs["collection_date"] = datetime.now().isoformat(timespec="seconds")
        h5_file.attrs["num_samples"] = int(len(buffer))
        h5_file.attrs["ee_quaternion_convention"] = "xyzw"
        h5_file.attrs["action_7d_source"] = "derived_from_observations"

        obs_group = h5_file.create_group("obs")
        create_dataset(
            obs_group,
            "agentview_rgb",
            np.stack(buffer.agentview_rgb, axis=0).astype(np.uint8),
            args.compression,
        )
        create_dataset(
            obs_group,
            "eye_in_hand_rgb",
            np.stack(buffer.eye_in_hand_rgb, axis=0).astype(np.uint8),
            args.compression,
        )
        create_dataset(
            obs_group,
            "joint_states",
            joint_states,
            args.compression,
        )
        create_dataset(
            obs_group,
            "joint_velocities",
            joint_velocities,
            args.compression,
        )
        create_dataset(
            obs_group,
            "gripper_states",
            gripper_states,
            args.compression,
        )
        create_dataset(
            obs_group,
            "ee_pos",
            ee_pos,
            args.compression,
        )
        create_dataset(
            obs_group,
            "ee_ori",
            ee_ori,
            args.compression,
        )
        create_dataset(
            obs_group,
            "ee_states",
            ee_states,
            args.compression,
        )
        create_dataset(
            obs_group,
            "timestamps",
            timestamps,
            args.compression,
        )
        create_dataset(
            h5_file,
            "actions_joint_position",
            actions_joint_position,
            args.compression,
        )
        create_dataset(
            h5_file,
            "action_delta_position",
            action_delta_position,
            args.compression,
        )
        create_dataset(
            h5_file,
            "action_delta_euler",
            action_delta_euler,
            args.compression,
        )
        create_dataset(
            h5_file,
            "action_gripper",
            action_gripper,
            args.compression,
        )
        create_dataset(
            h5_file,
            "action_7d",
            action_7d,
            args.compression,
        )
        write_camera_info(h5_file, camera_info)
        h5_file.flush()


def main(args: Args) -> None:
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.record_hz <= 0 or args.control_hz <= 0:
        raise ValueError("control_hz and record_hz must be positive.")
    if args.record_hz > args.control_hz:
        raise ValueError("record_hz must be <= control_hz.")

    robot_client: Optional[ZMQClientRobot] = None
    cameras: Optional[DualRealSenseRig] = None
    agent: Optional[GelloAgent] = None

    try:
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
        env = RobotEnv(
            robot_client, control_rate_hz=float(args.control_hz), camera_dict={}
        )
        cameras = DualRealSenseRig(args)

        reset_joints = (
            np.array(args.start_joints, dtype=np.float64)
            if args.start_joints is not None
            else None
        )
        if reset_joints is not None:
            move_robot_to_reset_joints(env, reset_joints)

        gello_start_joints = (
            reset_joints
            if reset_joints is not None
            else np.asarray(env.get_obs()["joint_positions"])
        )
        agent = GelloAgent(port=args.gello_port, start_joints=gello_start_joints)
        align_with_gello(env, agent, args.ignore_gripper_during_startup)

        print(f"Saving raw segments to {output_dir}")
        print(f"control_hz={args.control_hz}, record_hz={args.record_hz}")
        print(
            "Expected setup: launch scripts/ur3/teleop_launch_nodes.py first for the robot server."
        )

        while True:
            action = prompt_session_action()
            if action == "quit":
                break

            buffer, elapsed_s = record_segment(env, agent, cameras, args)
            if len(buffer) == 0:
                print("Recorded 0 samples; discarding segment.")
                continue

            print(summarize_segment(buffer, elapsed_s))
            save_action = prompt_save_action()
            if save_action == "d":
                print("Discarded segment.")
                continue
            if save_action == "q":
                print("Discarded segment and quitting session.")
                break

            segment_path = next_raw_segment_path(output_dir, args.filename_prefix)
            write_raw_segment(segment_path, buffer, args, cameras.camera_info())
            print(f"Saved raw segment to {segment_path}")

    finally:
        if agent is not None:
            close = getattr(agent, "close", None)
            if close is not None:
                close()
        if robot_client is not None:
            close = getattr(robot_client, "close", None)
            if close is not None:
                close()
        if cameras is not None:
            cameras.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
