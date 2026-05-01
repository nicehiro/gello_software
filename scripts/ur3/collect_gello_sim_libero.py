import json
import re
import select
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gello.agents.gello_agent import GelloAgent
from gello.env import RobotEnv
from gello.zmq_core.camera_node import ZMQClientCamera
from gello.zmq_core.robot_node import ZMQClientRobot

DEFAULT_GELLO_PORT = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTB8HJD7-if00-port0"
DEFAULT_START_JOINTS = (1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 1.5708, 0.0)
CAMERA_NAMES = ("agentview", "eye_in_hand")


@dataclass
class Args:
    dataset_path: str
    gello_port: str = DEFAULT_GELLO_PORT
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    control_hz: int = 100
    record_hz: int = 20
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    start_joints: Optional[Tuple[float, ...]] = DEFAULT_START_JOINTS
    ignore_gripper_during_startup: bool = True
    task_name_default: str = "free_teleop"
    operator: str = ""
    env_name: str = "UR3SimGello"
    schema_version: str = "ur3_libero_sim_v1"
    robot_name: str = "ur3"
    gripper_name: str = "robotiq_2f"
    teleop_device: str = "gello"
    compression: str = "gzip"


@dataclass
class EpisodeBuffer:
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

    def append(self, obs: dict[str, np.ndarray], command: np.ndarray, timestamp: float) -> None:
        self.timestamps.append(float(timestamp))
        self.agentview_rgb.append(np.array(obs["agentview_rgb"], dtype=np.uint8, copy=True))
        self.eye_in_hand_rgb.append(np.array(obs["eye_in_hand_rgb"], dtype=np.uint8, copy=True))
        self.joint_states.append(np.array(obs["joint_positions"], dtype=np.float32, copy=True))
        self.joint_velocities.append(np.array(obs["joint_velocities"], dtype=np.float32, copy=True))

        gripper_state = np.array(obs["gripper_position"], dtype=np.float32, copy=True).reshape(-1)
        if gripper_state.size != 1:
            raise ValueError(f"Expected scalar gripper position, got shape {gripper_state.shape}.")
        self.gripper_states.append(gripper_state)

        ee_pos_quat = np.array(obs["ee_pos_quat"], dtype=np.float64, copy=True).reshape(-1)
        if ee_pos_quat.size != 7:
            raise ValueError(f"Expected ee_pos_quat with 7 values, got shape {ee_pos_quat.shape}.")
        ee_pos = ee_pos_quat[:3].astype(np.float32)
        ee_ori = ee_pos_quat[3:].astype(np.float32)
        self.ee_pos.append(ee_pos)
        self.ee_ori.append(ee_ori)
        self.ee_states.append(np.concatenate([ee_pos, ee_ori]).astype(np.float32))
        self.commanded_joint_positions.append(np.array(command, dtype=np.float32, copy=True))


@dataclass
class FinalizedEpisode:
    obs: dict[str, np.ndarray]
    actions: np.ndarray
    actions_joint_position: np.ndarray
    robot_states: np.ndarray
    states: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray

    @property
    def num_samples(self) -> int:
        return int(self.actions.shape[0])


def freeze_gripper(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    frozen = np.array(target, copy=True)
    if len(frozen) == len(current) and len(frozen) > 0:
        frozen[-1] = current[-1]
    return frozen


def validate_camera_clients(camera_clients: dict[str, ZMQClientCamera]) -> None:
    for name, camera in camera_clients.items():
        try:
            camera.read()
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to initialize {name} camera. Start the matching camera node first."
            ) from exc


def move_robot_to_reset_joints(env: RobotEnv, reset_joints: np.ndarray) -> None:
    current_joints = np.asarray(env.get_obs()["joint_positions"], dtype=np.float64)
    if reset_joints.shape != current_joints.shape:
        return

    max_delta = float(np.abs(current_joints - reset_joints).max())
    steps = max(1, min(int(max_delta / 0.01), 100))
    for joints in np.linspace(current_joints, reset_joints, steps):
        env.step(joints.astype(np.float64))
        time.sleep(0.001)


def align_with_gello(env: RobotEnv, agent: GelloAgent, ignore_gripper_during_startup: bool) -> None:
    print("Aligning sim UR3 with GELLO start pose...")
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
        response = prompt_line("[Enter] start new demo, [q] quit session: ")
        if response == "":
            return "start"
        if response.lower() == "q":
            return "quit"
        print("Invalid input. Press Enter to start or q to quit.")


def prompt_review_action() -> str:
    while True:
        response = prompt_line("[k] keep, [d] discard, [q] quit session: ").lower()
        if response in {"k", "d", "q"}:
            return response
        print("Invalid input. Choose k, d, or q.")


def prompt_episode_metadata(default_task_name: str, default_instruction: str = "TBD") -> dict[str, str]:
    instruction = prompt_line(f"Instruction [blank={default_instruction}]: ") or default_instruction
    task_name = prompt_line(f"Task name [blank={default_task_name}]: ") or default_task_name
    notes = prompt_line("Notes [blank=]: ")
    return {
        "language": instruction,
        "task_name": task_name,
        "notes": notes,
    }


def normalize_quaternion_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def quaternion_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def quaternion_multiply_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = lhs
    w2, x2, y2, z2 = rhs
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quaternion_delta_to_rotvec_wxyz(curr: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    curr = normalize_quaternion_wxyz(curr)
    nxt = normalize_quaternion_wxyz(nxt)
    delta = quaternion_multiply_wxyz(nxt, quaternion_conjugate_wxyz(curr))
    delta = normalize_quaternion_wxyz(delta)

    if delta[0] < 0.0:
        delta = -delta

    w = float(np.clip(delta[0], -1.0, 1.0))
    xyz = delta[1:]
    xyz_norm = float(np.linalg.norm(xyz))
    if xyz_norm < 1e-12:
        return np.zeros(3, dtype=np.float32)

    angle = 2.0 * np.arctan2(xyz_norm, w)
    if angle > np.pi:
        angle -= 2.0 * np.pi
    axis = xyz / xyz_norm
    return (axis * angle).astype(np.float32)


def compute_ee_delta_actions(ee_pos: np.ndarray, ee_ori: np.ndarray, gripper_cmd: np.ndarray) -> np.ndarray:
    num_steps = ee_pos.shape[0]
    deltas = np.zeros((num_steps, 7), dtype=np.float32)
    if num_steps == 0:
        return deltas

    if num_steps > 1:
        deltas[:-1, :3] = ee_pos[1:] - ee_pos[:-1]
        for idx in range(num_steps - 1):
            deltas[idx, 3:6] = quaternion_delta_to_rotvec_wxyz(ee_ori[idx], ee_ori[idx + 1])
        deltas[-1, :6] = deltas[-2, :6]

    deltas[:, 6] = gripper_cmd.reshape(-1)
    return deltas


def finalize_episode(buffer: EpisodeBuffer) -> FinalizedEpisode:
    agentview_rgb = np.stack(buffer.agentview_rgb, axis=0).astype(np.uint8)
    eye_in_hand_rgb = np.stack(buffer.eye_in_hand_rgb, axis=0).astype(np.uint8)
    joint_states = np.stack(buffer.joint_states, axis=0).astype(np.float32)
    joint_velocities = np.stack(buffer.joint_velocities, axis=0).astype(np.float32)
    gripper_states = np.stack(buffer.gripper_states, axis=0).astype(np.float32)
    ee_pos = np.stack(buffer.ee_pos, axis=0).astype(np.float32)
    ee_ori = np.stack(buffer.ee_ori, axis=0).astype(np.float32)
    ee_states = np.stack(buffer.ee_states, axis=0).astype(np.float32)
    timestamps = np.asarray(buffer.timestamps, dtype=np.float64)
    actions_joint_position = np.stack(buffer.commanded_joint_positions, axis=0).astype(np.float32)

    actions = compute_ee_delta_actions(ee_pos, ee_ori, actions_joint_position[:, -1:])
    robot_states = np.concatenate(
        [joint_states, joint_velocities, ee_pos, ee_ori, gripper_states],
        axis=-1,
    ).astype(np.float32)
    states = robot_states.copy()

    rewards = np.zeros((len(buffer),), dtype=np.uint8)
    dones = np.zeros((len(buffer),), dtype=np.uint8)
    rewards[-1] = 1
    dones[-1] = 1

    return FinalizedEpisode(
        obs={
            "agentview_rgb": agentview_rgb,
            "eye_in_hand_rgb": eye_in_hand_rgb,
            "joint_states": joint_states,
            "joint_velocities": joint_velocities,
            "gripper_states": gripper_states,
            "ee_pos": ee_pos,
            "ee_ori": ee_ori,
            "ee_states": ee_states,
            "timestamps": timestamps,
        },
        actions=actions,
        actions_joint_position=actions_joint_position,
        robot_states=robot_states,
        states=states,
        rewards=rewards,
        dones=dones,
    )


def summarize_episode(buffer: EpisodeBuffer, elapsed_s: float) -> str:
    num_samples = len(buffer)
    avg_saved_hz = num_samples / elapsed_s if elapsed_s > 0 else 0.0
    return (
        f"Recorded {num_samples} samples over {elapsed_s:.2f}s "
        f"(avg saved rate {avg_saved_hz:.2f} Hz)."
    )


def record_episode(env: RobotEnv, agent: GelloAgent, args: Args) -> tuple[EpisodeBuffer, float]:
    buffer = EpisodeBuffer()
    obs = env.get_obs()
    start_time = time.monotonic()
    last_record_time = start_time - 1.0 / float(args.record_hz)
    print("Recording... press Enter to stop current demo.")

    while True:
        action = np.asarray(agent.act(obs), dtype=np.float64)

        now = time.monotonic()
        if now - last_record_time >= 1.0 / float(args.record_hz):
            buffer.append(obs, action, now - start_time)
            last_record_time = now

        if stdin_line_available():
            consume_stdin_line()
            break

        obs = env.step(action)

    elapsed_s = time.monotonic() - start_time
    return buffer, elapsed_s


def create_or_update_root_attrs(data_group: h5py.Group, args: Args) -> None:
    data_group.attrs["schema_version"] = args.schema_version
    data_group.attrs["env_name"] = args.env_name
    data_group.attrs["robot"] = args.robot_name
    data_group.attrs["gripper"] = args.gripper_name
    data_group.attrs["teleop_device"] = args.teleop_device
    data_group.attrs["control_freq"] = int(args.control_hz)
    data_group.attrs["record_freq"] = int(args.record_hz)
    data_group.attrs["action_space"] = "ee_delta"
    data_group.attrs["camera_names"] = json.dumps(list(CAMERA_NAMES))


_DEMO_PATTERN = re.compile(r"demo_(\d+)$")


def next_demo_index(data_group: h5py.Group) -> int:
    max_index = -1
    for key in data_group.keys():
        match = _DEMO_PATTERN.fullmatch(key)
        if match is not None:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def create_dataset(group: h5py.Group, name: str, data: np.ndarray, compression: str) -> None:
    kwargs = {}
    if data.ndim >= 1:
        kwargs["chunks"] = (1, *data.shape[1:])
    if compression:
        kwargs["compression"] = compression
    group.create_dataset(name, data=data, **kwargs)


def update_summary_attrs(data_group: h5py.Group) -> None:
    demo_keys = [key for key in data_group.keys() if _DEMO_PATTERN.fullmatch(key)]
    total = 0
    for key in demo_keys:
        total += int(data_group[key].attrs["num_samples"])
    data_group.attrs["num_demos"] = len(demo_keys)
    data_group.attrs["total"] = total


def write_episode(
    h5_file: h5py.File,
    finalized: FinalizedEpisode,
    args: Args,
    metadata: dict[str, str],
) -> str:
    data_group = h5_file.require_group("data")
    create_or_update_root_attrs(data_group, args)
    demo_index = next_demo_index(data_group)
    demo_name = f"demo_{demo_index}"
    demo_group = data_group.create_group(demo_name)
    obs_group = demo_group.create_group("obs")

    for key, value in finalized.obs.items():
        create_dataset(obs_group, key, value, compression=args.compression)

    create_dataset(demo_group, "actions", finalized.actions, compression=args.compression)
    create_dataset(
        demo_group,
        "actions_joint_position",
        finalized.actions_joint_position,
        compression=args.compression,
    )
    create_dataset(demo_group, "robot_states", finalized.robot_states, compression=args.compression)
    create_dataset(demo_group, "states", finalized.states, compression=args.compression)
    create_dataset(demo_group, "rewards", finalized.rewards, compression=args.compression)
    create_dataset(demo_group, "dones", finalized.dones, compression=args.compression)

    demo_group.attrs["num_samples"] = finalized.num_samples
    demo_group.attrs["task_name"] = metadata["task_name"]
    demo_group.attrs["language"] = metadata["language"]
    demo_group.attrs["success"] = np.uint8(1)
    demo_group.attrs["operator"] = args.operator
    demo_group.attrs["episode_id"] = demo_name
    demo_group.attrs["collection_date"] = datetime.now().isoformat(timespec="seconds")
    demo_group.attrs["action_space"] = "ee_delta"
    demo_group.attrs["joint_action_space"] = "joint_position"
    demo_group.attrs["state_encoding"] = "robot_only"
    demo_group.attrs["notes"] = metadata["notes"]

    update_summary_attrs(data_group)
    h5_file.flush()
    return demo_name


def main(args: Args) -> None:
    dataset_path = Path(args.dataset_path).expanduser()
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    if args.record_hz <= 0 or args.control_hz <= 0:
        raise ValueError("control_hz and record_hz must be positive.")
    if args.record_hz > args.control_hz:
        raise ValueError("record_hz must be <= control_hz.")

    robot_client: Optional[ZMQClientRobot] = None
    camera_clients: dict[str, ZMQClientCamera] = {}
    agent: Optional[GelloAgent] = None

    try:
        camera_clients = {
            "eye_in_hand": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
            "agentview": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
        }
        validate_camera_clients(camera_clients)

        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
        env = RobotEnv(robot_client, control_rate_hz=float(args.control_hz), camera_dict=camera_clients)

        reset_joints = np.array(args.start_joints, dtype=np.float64) if args.start_joints is not None else None
        if reset_joints is not None:
            move_robot_to_reset_joints(env, reset_joints)

        gello_start_joints = reset_joints if reset_joints is not None else np.asarray(env.get_obs()["joint_positions"])
        agent = GelloAgent(port=args.gello_port, start_joints=gello_start_joints)
        align_with_gello(env, agent, args.ignore_gripper_during_startup)

        print(f"Saving demos to {dataset_path}")
        print(f"control_hz={args.control_hz}, record_hz={args.record_hz}")

        with h5py.File(dataset_path, "a") as h5_file:
            data_group = h5_file.require_group("data")
            create_or_update_root_attrs(data_group, args)
            update_summary_attrs(data_group)
            h5_file.flush()

            while True:
                action = prompt_session_action()
                if action == "quit":
                    break

                buffer, elapsed_s = record_episode(env, agent, args)
                if len(buffer) == 0:
                    print("Recorded 0 samples; discarding demo.")
                    continue

                print(summarize_episode(buffer, elapsed_s))
                review_action = prompt_review_action()
                if review_action == "d":
                    print("Discarded demo.")
                    continue
                if review_action == "q":
                    break

                metadata = prompt_episode_metadata(args.task_name_default)
                finalized = finalize_episode(buffer)
                demo_name = write_episode(h5_file, finalized, args, metadata)
                print(f"Saved {demo_name} to {dataset_path}")

    finally:
        if agent is not None:
            close = getattr(agent, "close", None)
            if close is not None:
                close()
        if robot_client is not None:
            close = getattr(robot_client, "close", None)
            if close is not None:
                close()
        for camera_client in camera_clients.values():
            close = getattr(camera_client, "close", None)
            if close is not None:
                close()


if __name__ == "__main__":
    main(tyro.cli(Args))
