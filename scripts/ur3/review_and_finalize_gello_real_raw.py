import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

for env_name in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH"):
    value = os.environ.get(env_name, "")
    if "cv2" in value:
        os.environ.pop(env_name, None)

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QAction, QImage, QKeySequence, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import cv2
import h5py
import numpy as np
import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CAMERA_NAMES = ("agentview", "eye_in_hand")
_RAW_SEGMENT_PATTERN = re.compile(r"raw_\d{5}\.hdf5$")
IMAGE_LABEL_MIN_WIDTH = 960
IMAGE_LABEL_MIN_HEIGHT = 540
REVIEW_ATTR_KEEP = "keep"
REVIEW_ATTR_LANGUAGE = "language"
REVIEW_ATTR_TASK_NAME = "task_name"
REVIEW_ATTR_SCENE_ID = "scene_id"
REVIEW_ATTR_NOTES = "notes"
REVIEW_ATTR_EXPORTED_PATH = "exported_path"
LEGACY_REVIEW_ATTR_KEEP = "review_keep"
LEGACY_REVIEW_ATTR_LANGUAGE = "review_language"
LEGACY_REVIEW_ATTR_TASK_NAME = "review_task_name"
LEGACY_REVIEW_ATTR_SCENE_ID = "review_scene_id"
LEGACY_REVIEW_ATTR_NOTES = "review_notes"
LEGACY_REVIEW_ATTR_EXPORTED_PATH = "review_exported_path"


@dataclass
class Args:
    input_path: str
    output_dir: str = ""
    compression: str = "gzip"
    header_height: int = 140
    gap: int = 8
    playback_fps: float = 0.0
    scene_id_default: str = "default"
    task_name_default: str = "unlabeled"
    instruction_default: str = "TBD"


@dataclass
class ReviewState:
    keep: bool
    trim_start: int
    trim_end: int
    language: str
    task_name: str
    scene_id: str
    notes: str
    exported_path: str = ""


@dataclass
class SegmentInfo:
    path: Path
    num_samples: int
    attrs: dict[str, object]


@dataclass
class RawSegment:
    path: Path
    agentview_rgb: np.ndarray
    eye_in_hand_rgb: np.ndarray
    joint_states: np.ndarray
    joint_velocities: np.ndarray
    gripper_states: np.ndarray
    ee_pos: np.ndarray
    ee_ori: np.ndarray
    ee_states: np.ndarray
    timestamps: np.ndarray
    actions_joint_position: np.ndarray
    attrs: dict[str, object]
    camera_info: dict[str, dict[str, object]]

    @property
    def num_samples(self) -> int:
        return int(self.timestamps.shape[0])


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


def quaternion_delta_to_rotvec_xyzw(curr: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    curr = normalize_quaternion_xyzw(curr)
    nxt = normalize_quaternion_xyzw(nxt)
    delta = quaternion_multiply_xyzw(nxt, quaternion_conjugate_xyzw(curr))
    delta = normalize_quaternion_xyzw(delta)

    if delta[3] < 0.0:
        delta = -delta

    xyz = delta[:3]
    w = float(np.clip(delta[3], -1.0, 1.0))
    xyz_norm = float(np.linalg.norm(xyz))
    if xyz_norm < 1e-12:
        return np.zeros(3, dtype=np.float32)

    angle = 2.0 * np.arctan2(xyz_norm, w)
    if angle > np.pi:
        angle -= 2.0 * np.pi
    axis = xyz / xyz_norm
    return (axis * angle).astype(np.float32)


def compute_ee_delta_actions(
    ee_pos: np.ndarray, ee_ori: np.ndarray, gripper_cmd: np.ndarray
) -> np.ndarray:
    num_steps = ee_pos.shape[0]
    deltas = np.zeros((num_steps, 7), dtype=np.float32)
    if num_steps == 0:
        return deltas

    if num_steps > 1:
        deltas[:-1, :3] = ee_pos[1:] - ee_pos[:-1]
        for idx in range(num_steps - 1):
            deltas[idx, 3:6] = quaternion_delta_to_rotvec_xyzw(
                ee_ori[idx], ee_ori[idx + 1]
            )
        deltas[-1, :6] = deltas[-2, :6]

    deltas[:, 6] = gripper_cmd.reshape(-1)
    return deltas


def read_camera_info(h5_file: h5py.File) -> dict[str, dict[str, object]]:
    if "camera_info" not in h5_file:
        return {}
    result: dict[str, dict[str, object]] = {}
    for name, group in h5_file["camera_info"].items():
        result[name] = {
            "serial": group.attrs.get("serial", ""),
            "resolution": np.asarray(group.attrs.get("resolution", (0, 0))),
            "fps": int(group.attrs.get("fps", 0)),
            "intrinsics": group["intrinsics"][()] if "intrinsics" in group else None,
        }
    return result


def load_segment_info(path: Path) -> SegmentInfo:
    with h5py.File(path, "r") as h5_file:
        attrs = {key: h5_file.attrs[key] for key in h5_file.attrs.keys()}
        num_samples = int(attrs.get("num_samples", h5_file["obs"]["timestamps"].shape[0]))
        return SegmentInfo(path=path, num_samples=num_samples, attrs=attrs)


def load_raw_segment(path: Path) -> RawSegment:
    with h5py.File(path, "r") as h5_file:
        obs_group = h5_file["obs"]
        return RawSegment(
            path=path,
            agentview_rgb=obs_group["agentview_rgb"][()],
            eye_in_hand_rgb=obs_group["eye_in_hand_rgb"][()],
            joint_states=obs_group["joint_states"][()],
            joint_velocities=obs_group["joint_velocities"][()],
            gripper_states=obs_group["gripper_states"][()],
            ee_pos=obs_group["ee_pos"][()],
            ee_ori=obs_group["ee_ori"][()],
            ee_states=obs_group["ee_states"][()],
            timestamps=obs_group["timestamps"][()],
            actions_joint_position=h5_file["actions_joint_position"][()],
            attrs={key: h5_file.attrs[key] for key in h5_file.attrs.keys()},
            camera_info=read_camera_info(h5_file),
        )


def default_review_state(segment: SegmentInfo | RawSegment, args: Args) -> ReviewState:
    return ReviewState(
        keep=True,
        trim_start=0,
        trim_end=max(segment.num_samples - 1, 0),
        language=args.instruction_default,
        task_name=args.task_name_default,
        scene_id=args.scene_id_default,
        notes="",
    )


def sidecar_path(raw_path: Path) -> Path:
    return raw_path.with_suffix(".json")


def clamp_review_state(state: ReviewState, segment: SegmentInfo | RawSegment) -> None:
    last = max(segment.num_samples - 1, 0)
    state.trim_start = int(np.clip(state.trim_start, 0, last))
    state.trim_end = int(np.clip(state.trim_end, 0, last))
    if state.trim_start > state.trim_end:
        state.trim_start = state.trim_end


def load_review_state(segment: SegmentInfo | RawSegment, args: Args) -> ReviewState:
    state = default_review_state(segment, args)

    attrs = segment.attrs
    state.keep = bool(
        attrs.get(REVIEW_ATTR_KEEP, attrs.get(LEGACY_REVIEW_ATTR_KEEP, state.keep))
    )
    state.language = str(
        attrs.get(
            REVIEW_ATTR_LANGUAGE,
            attrs.get(LEGACY_REVIEW_ATTR_LANGUAGE, state.language),
        )
    )
    state.task_name = str(
        attrs.get(
            REVIEW_ATTR_TASK_NAME,
            attrs.get(LEGACY_REVIEW_ATTR_TASK_NAME, state.task_name),
        )
    )
    state.scene_id = str(
        attrs.get(
            REVIEW_ATTR_SCENE_ID,
            attrs.get(LEGACY_REVIEW_ATTR_SCENE_ID, state.scene_id),
        )
    )
    state.notes = str(
        attrs.get(REVIEW_ATTR_NOTES, attrs.get(LEGACY_REVIEW_ATTR_NOTES, state.notes))
    )
    state.exported_path = str(
        attrs.get(
            REVIEW_ATTR_EXPORTED_PATH,
            attrs.get(LEGACY_REVIEW_ATTR_EXPORTED_PATH, state.exported_path),
        )
    )

    path = sidecar_path(segment.path)
    if path.exists():
        data = json.loads(path.read_text())
        state.trim_start = int(data.get("trim_start", state.trim_start))
        state.trim_end = int(data.get("trim_end", state.trim_end))
        if REVIEW_ATTR_KEEP not in segment.attrs and LEGACY_REVIEW_ATTR_KEEP not in segment.attrs:
            state.keep = bool(data.get("keep", state.keep))
        if REVIEW_ATTR_LANGUAGE not in segment.attrs and LEGACY_REVIEW_ATTR_LANGUAGE not in segment.attrs:
            state.language = str(data.get("language", state.language))
        if REVIEW_ATTR_TASK_NAME not in segment.attrs and LEGACY_REVIEW_ATTR_TASK_NAME not in segment.attrs:
            state.task_name = str(data.get("task_name", state.task_name))
        if REVIEW_ATTR_SCENE_ID not in segment.attrs and LEGACY_REVIEW_ATTR_SCENE_ID not in segment.attrs:
            state.scene_id = str(data.get("scene_id", state.scene_id))
        if REVIEW_ATTR_NOTES not in segment.attrs and LEGACY_REVIEW_ATTR_NOTES not in segment.attrs:
            state.notes = str(data.get("notes", state.notes))
        if REVIEW_ATTR_EXPORTED_PATH not in segment.attrs and LEGACY_REVIEW_ATTR_EXPORTED_PATH not in segment.attrs:
            state.exported_path = str(data.get("exported_path", state.exported_path))

    clamp_review_state(state, segment)
    return state


def save_review_state(raw_path: Path, state: ReviewState) -> None:
    with h5py.File(raw_path, "r+") as h5_file:
        h5_file.attrs[REVIEW_ATTR_KEEP] = bool(state.keep)
        h5_file.attrs[REVIEW_ATTR_LANGUAGE] = state.language
        h5_file.attrs[REVIEW_ATTR_TASK_NAME] = state.task_name
        h5_file.attrs[REVIEW_ATTR_SCENE_ID] = state.scene_id
        h5_file.attrs[REVIEW_ATTR_NOTES] = state.notes
        h5_file.attrs[REVIEW_ATTR_EXPORTED_PATH] = state.exported_path


def list_raw_segments(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    paths = []
    for path in sorted(input_path.glob("*.hdf5")):
        if path.name.endswith("_libero.hdf5"):
            continue
        if _RAW_SEGMENT_PATTERN.fullmatch(path.name) is None:
            continue
        paths.append(path)
    return paths


def estimate_playback_fps(segment: RawSegment, fallback_fps: float) -> float:
    if fallback_fps > 0:
        return fallback_fps
    if len(segment.timestamps) >= 2:
        dt = np.diff(segment.timestamps)
        dt = dt[dt > 0]
        if len(dt) > 0:
            return float(1.0 / np.median(dt))
    value = segment.attrs.get("record_freq", 0)
    return float(value) if value else 6.0


def current_trim_length(state: ReviewState) -> int:
    return max(0, state.trim_end - state.trim_start + 1)


def render_frame(
    segment: RawSegment,
    state: ReviewState,
    frame_idx: int,
    file_index: int,
    num_files: int,
    args: Args,
) -> np.ndarray:
    left = cv2.cvtColor(segment.agentview_rgb[frame_idx], cv2.COLOR_RGB2BGR)
    right = cv2.cvtColor(segment.eye_in_hand_rgb[frame_idx], cv2.COLOR_RGB2BGR)

    frame_height, frame_width = left.shape[0], left.shape[1]
    canvas_height = frame_height + args.header_height
    canvas_width = frame_width * 2 + args.gap
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[args.header_height :, :frame_width] = left
    canvas[args.header_height :, frame_width + args.gap :] = right

    status = "KEEP" if state.keep else "DISCARD"
    status_color = (0, 200, 0) if state.keep else (0, 0, 255)
    trim_flag = state.trim_start <= frame_idx <= state.trim_end
    trim_text = "IN TRIM" if trim_flag else "OUTSIDE TRIM"
    trim_color = (0, 220, 0) if trim_flag else (0, 140, 255)

    header_lines = [
        f"[{file_index + 1}/{num_files}] {segment.path.name}",
        (
            f"frame {frame_idx + 1}/{segment.num_samples} | trim "
            f"[{state.trim_start + 1}, {state.trim_end + 1}] => {current_trim_length(state)} frames"
        ),
        (
            f"status={status} | task={state.task_name!r} | "
            f"instruction={state.language!r}"
        ),
        f"scene={state.scene_id!r} | notes={state.notes!r}",
    ]

    for line_idx, text in enumerate(header_lines):
        cv2.putText(
            canvas,
            text,
            (10, 24 + line_idx * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        "agentview",
        (12, args.header_height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "eye_in_hand",
        (frame_width + args.gap + 12, args.header_height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        status,
        (canvas_width - 150, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        trim_text,
        (canvas_width - 210, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        trim_color,
        2,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def slice_segment(segment: RawSegment, state: ReviewState) -> RawSegment:
    sl = slice(state.trim_start, state.trim_end + 1)
    return RawSegment(
        path=segment.path,
        agentview_rgb=segment.agentview_rgb[sl],
        eye_in_hand_rgb=segment.eye_in_hand_rgb[sl],
        joint_states=segment.joint_states[sl],
        joint_velocities=segment.joint_velocities[sl],
        gripper_states=segment.gripper_states[sl],
        ee_pos=segment.ee_pos[sl],
        ee_ori=segment.ee_ori[sl],
        ee_states=segment.ee_states[sl],
        timestamps=segment.timestamps[sl],
        actions_joint_position=segment.actions_joint_position[sl],
        attrs=segment.attrs,
        camera_info=segment.camera_info,
    )


def finalize_segment(segment: RawSegment) -> FinalizedEpisode:
    actions = compute_ee_delta_actions(
        segment.ee_pos, segment.ee_ori, segment.actions_joint_position[:, -1:]
    )
    robot_states = np.concatenate(
        [
            segment.joint_states,
            segment.joint_velocities,
            segment.ee_pos,
            segment.ee_ori,
            segment.gripper_states,
        ],
        axis=-1,
    ).astype(np.float32)
    states = robot_states.copy()
    rewards = np.zeros((segment.num_samples,), dtype=np.uint8)
    dones = np.zeros((segment.num_samples,), dtype=np.uint8)
    rewards[-1] = 1
    dones[-1] = 1
    return FinalizedEpisode(
        obs={
            "agentview_rgb": segment.agentview_rgb.astype(np.uint8),
            "eye_in_hand_rgb": segment.eye_in_hand_rgb.astype(np.uint8),
            "joint_states": segment.joint_states.astype(np.float32),
            "joint_velocities": segment.joint_velocities.astype(np.float32),
            "gripper_states": segment.gripper_states.astype(np.float32),
            "ee_pos": segment.ee_pos.astype(np.float32),
            "ee_ori": segment.ee_ori.astype(np.float32),
            "ee_states": segment.ee_states.astype(np.float32),
            "timestamps": segment.timestamps.astype(np.float64),
        },
        actions=actions,
        actions_joint_position=segment.actions_joint_position.astype(np.float32),
        robot_states=robot_states,
        states=states,
        rewards=rewards,
        dones=dones,
    )


def create_dataset(
    group: h5py.Group, name: str, data: np.ndarray, compression: str
) -> None:
    kwargs = {}
    if data.ndim >= 1:
        kwargs["chunks"] = (1, *data.shape[1:])
    if compression:
        kwargs["compression"] = compression
    group.create_dataset(name, data=data, **kwargs)


def upsert_dataset(group: h5py.Group, name: str, data: np.ndarray) -> None:
    if name in group:
        del group[name]
    group.create_dataset(name, data=data)


def write_camera_info(h5_file: h5py.File, camera_info: dict[str, dict[str, object]]) -> None:
    if not camera_info:
        return
    root = h5_file.require_group("camera_info")
    for name, info in camera_info.items():
        group = root.require_group(name)
        group.attrs["serial"] = info.get("serial", "")
        group.attrs["resolution"] = np.asarray(
            info.get("resolution", (0, 0)), dtype=np.int32
        )
        group.attrs["fps"] = int(info.get("fps", 0))
        intrinsics = info.get("intrinsics")
        if intrinsics is not None:
            upsert_dataset(group, "intrinsics", np.asarray(intrinsics, dtype=np.float32))


def trim_raw_segment_in_place(
    raw_path: Path,
    state: ReviewState,
    compression: str,
) -> None:
    temp_path = raw_path.with_suffix(raw_path.suffix + ".tmp")
    trim_slice = slice(state.trim_start, state.trim_end + 1)

    with h5py.File(raw_path, "r") as src, h5py.File(temp_path, "w") as dst:
        original_num_samples = int(src.attrs.get("num_samples", 0))
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst.attrs["num_samples"] = int(state.trim_end - state.trim_start + 1)
        dst.attrs[REVIEW_ATTR_KEEP] = bool(state.keep)
        dst.attrs[REVIEW_ATTR_LANGUAGE] = state.language
        dst.attrs[REVIEW_ATTR_TASK_NAME] = state.task_name
        dst.attrs[REVIEW_ATTR_SCENE_ID] = state.scene_id
        dst.attrs[REVIEW_ATTR_NOTES] = state.notes
        dst.attrs[REVIEW_ATTR_EXPORTED_PATH] = state.exported_path

        for name, item in src.items():
            if isinstance(item, h5py.Dataset):
                data = item[()]
                if data.ndim >= 1 and data.shape[0] == original_num_samples:
                    data = data[trim_slice]
                create_dataset(dst, name, data, compression)
                continue

            group = dst.create_group(name)
            for attr_key, attr_value in item.attrs.items():
                group.attrs[attr_key] = attr_value

            for child_name, child in item.items():
                if isinstance(child, h5py.Dataset):
                    data = child[()]
                    if name == "obs" and data.ndim >= 1 and data.shape[0] == original_num_samples:
                        data = data[trim_slice]
                    create_dataset(group, child_name, data, compression)
                    continue

                child_group = group.create_group(child_name)
                for attr_key, attr_value in child.attrs.items():
                    child_group.attrs[attr_key] = attr_value
                for grandchild_name, grandchild in child.items():
                    upsert_dataset(child_group, grandchild_name, grandchild[()])

    os.replace(temp_path, raw_path)


def write_finalized_file(
    output_path: Path,
    finalized: FinalizedEpisode,
    metadata: ReviewState,
    raw_segment: RawSegment,
    compression: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5_file:
        data_group = h5_file.create_group("data")
        data_group.attrs["schema_version"] = "ur3_libero_real_v1"
        data_group.attrs["env_name"] = str(raw_segment.attrs.get("env_name", "UR3RealWorld"))
        data_group.attrs["robot"] = str(raw_segment.attrs.get("robot", "ur3"))
        data_group.attrs["gripper"] = str(raw_segment.attrs.get("gripper", "robotiq_2f"))
        data_group.attrs["teleop_device"] = str(
            raw_segment.attrs.get("teleop_device", "gello")
        )
        data_group.attrs["control_freq"] = int(raw_segment.attrs.get("control_freq", 0))
        data_group.attrs["record_freq"] = int(raw_segment.attrs.get("record_freq", 0))
        data_group.attrs["action_space"] = "ee_delta"
        data_group.attrs["camera_names"] = json.dumps(list(CAMERA_NAMES))
        data_group.attrs["num_demos"] = 1
        data_group.attrs["total"] = finalized.num_samples

        demo_group = data_group.create_group("demo_0")
        obs_group = demo_group.create_group("obs")
        for key, value in finalized.obs.items():
            create_dataset(obs_group, key, value, compression)

        create_dataset(demo_group, "actions", finalized.actions, compression)
        create_dataset(
            demo_group,
            "actions_joint_position",
            finalized.actions_joint_position,
            compression,
        )
        create_dataset(demo_group, "robot_states", finalized.robot_states, compression)
        create_dataset(demo_group, "states", finalized.states, compression)
        create_dataset(demo_group, "rewards", finalized.rewards, compression)
        create_dataset(demo_group, "dones", finalized.dones, compression)

        demo_group.attrs["num_samples"] = finalized.num_samples
        demo_group.attrs["task_name"] = metadata.task_name
        demo_group.attrs["language"] = metadata.language
        demo_group.attrs["success"] = np.uint8(1)
        demo_group.attrs["operator"] = str(raw_segment.attrs.get("operator", ""))
        demo_group.attrs["scene_id"] = metadata.scene_id
        demo_group.attrs["episode_id"] = raw_segment.path.stem
        demo_group.attrs["collection_date"] = str(
            raw_segment.attrs.get("collection_date", "")
        )
        demo_group.attrs["action_space"] = "ee_delta"
        demo_group.attrs["joint_action_space"] = "joint_position"
        demo_group.attrs["state_encoding"] = "robot_only"
        demo_group.attrs["notes"] = metadata.notes

        write_camera_info(h5_file, raw_segment.camera_info)
        h5_file.flush()


def resolve_output_path(raw_path: Path, args: Args) -> Path:
    base_dir = Path(args.output_dir).expanduser() if args.output_dir else raw_path.parent
    return base_dir / f"{raw_path.stem}_libero.hdf5"


class ReviewWindow(QMainWindow):
    def __init__(self, args: Args, segment_infos: list[SegmentInfo], states: list[ReviewState]):
        super().__init__()
        self.args = args
        self.segment_infos = segment_infos
        self.states = states
        self.index = 0
        self.frame_idx = 0
        self.playing = False
        self._ignore_widget_updates = False
        self._current_image: np.ndarray | None = None
        self._current_segment: RawSegment | None = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance_frame)

        self.setWindowTitle("GELLO Raw Review + Finalize")
        self.resize(1440, 980)
        self._build_ui()
        self._install_shortcuts()
        self.load_current_segment(reset_frame=True)

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        self.image_label = QLabel("No frame")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(IMAGE_LABEL_MIN_WIDTH, IMAGE_LABEL_MIN_HEIGHT)
        root.addWidget(self.image_label, stretch=1)

        self.status_label = QLabel("")
        root.addWidget(self.status_label)

        frame_row = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.valueChanged.connect(self.on_frame_slider_changed)
        self.frame_spin = QSpinBox()
        self.frame_spin.valueChanged.connect(self.on_frame_spin_changed)
        frame_row.addWidget(QLabel("Frame"))
        frame_row.addWidget(self.frame_slider, stretch=1)
        frame_row.addWidget(self.frame_spin)
        root.addLayout(frame_row)

        trim_group = QGroupBox("Trim / Status")
        trim_layout = QHBoxLayout(trim_group)
        self.keep_checkbox = QCheckBox("Keep")
        self.keep_checkbox.toggled.connect(self.on_metadata_changed)
        self.trim_start_spin = QSpinBox()
        self.trim_start_spin.valueChanged.connect(self.on_trim_changed)
        self.trim_end_spin = QSpinBox()
        self.trim_end_spin.valueChanged.connect(self.on_trim_changed)
        trim_layout.addWidget(self.keep_checkbox)
        trim_layout.addWidget(QLabel("Trim start"))
        trim_layout.addWidget(self.trim_start_spin)
        trim_layout.addWidget(QLabel("Trim end"))
        trim_layout.addWidget(self.trim_end_spin)
        trim_layout.addStretch(1)
        root.addWidget(trim_group)

        form_group = QGroupBox("Annotation")
        form_layout = QFormLayout(form_group)
        self.task_edit = QLineEdit()
        self.task_edit.editingFinished.connect(self.on_metadata_changed)
        self.instruction_edit = QLineEdit()
        self.instruction_edit.editingFinished.connect(self.on_metadata_changed)
        self.scene_edit = QLineEdit()
        self.scene_edit.editingFinished.connect(self.on_metadata_changed)
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setTabChangesFocus(True)
        self.notes_edit.textChanged.connect(self.on_metadata_changed)
        form_layout.addRow("Task", self.task_edit)
        form_layout.addRow("Instruction", self.instruction_edit)
        form_layout.addRow("Scene", self.scene_edit)
        form_layout.addRow("Notes", self.notes_edit)
        root.addWidget(form_group)

        button_grid = QGridLayout()
        self.prev_file_button = QPushButton("Prev File (P)")
        self.prev_file_button.clicked.connect(self.prev_file)
        self.next_file_button = QPushButton("Next File (N)")
        self.next_file_button.clicked.connect(self.next_file)
        self.prev_frame_button = QPushButton("Prev Frame (A)")
        self.prev_frame_button.clicked.connect(self.prev_frame)
        self.next_frame_button = QPushButton("Next Frame (D)")
        self.next_frame_button.clicked.connect(self.next_frame)
        self.play_button = QPushButton("Play/Pause (Space)")
        self.play_button.clicked.connect(self.toggle_playback)
        self.set_trim_start_button = QPushButton("Set Trim Start ([)")
        self.set_trim_start_button.clicked.connect(self.set_trim_start_to_current)
        self.set_trim_end_button = QPushButton("Set Trim End (])")
        self.set_trim_end_button.clicked.connect(self.set_trim_end_to_current)
        self.save_button = QPushButton("Save Review")
        self.save_button.clicked.connect(self.save_current_state)
        self.apply_trim_button = QPushButton("Apply Trim (T)")
        self.apply_trim_button.clicked.connect(self.apply_trim)
        self.export_button = QPushButton("Export Current (E)")
        self.export_button.clicked.connect(self.export_current)

        buttons = [
            self.prev_file_button,
            self.next_file_button,
            self.prev_frame_button,
            self.next_frame_button,
            self.play_button,
            self.set_trim_start_button,
            self.set_trim_end_button,
            self.save_button,
            self.apply_trim_button,
            self.export_button,
        ]
        for idx, button in enumerate(buttons):
            button_grid.addWidget(button, idx // 3, idx % 3)
        root.addLayout(button_grid)

    def _install_shortcuts(self) -> None:
        self.addAction(self._make_action("Space", self.toggle_playback))
        self.addAction(self._make_action("A", self.prev_frame))
        self.addAction(self._make_action("D", self.next_frame))
        self.addAction(self._make_action("P", self.prev_file))
        self.addAction(self._make_action("N", self.next_file))
        self.addAction(self._make_action("[", self.set_trim_start_to_current))
        self.addAction(self._make_action("]", self.set_trim_end_to_current))
        self.addAction(self._make_action("X", self.toggle_keep))
        self.addAction(self._make_action("E", self.export_current))
        self.addAction(self._make_action("T", self.apply_trim))
        self.addAction(self._make_action("Ctrl+S", self.save_current_state))

    def _make_action(self, shortcut: str, callback) -> QAction:
        action = QAction(self)
        action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(callback)
        return action

    @property
    def current_info(self) -> SegmentInfo:
        return self.segment_infos[self.index]

    @property
    def current_segment(self) -> RawSegment:
        info = self.current_info
        if self._current_segment is None or self._current_segment.path != info.path:
            self._current_segment = load_raw_segment(info.path)
        return self._current_segment

    @property
    def current_state(self) -> ReviewState:
        return self.states[self.index]

    def load_current_segment(self, reset_frame: bool) -> None:
        segment = self.current_segment
        state = self.current_state
        clamp_review_state(state, segment)
        self._ignore_widget_updates = True
        self.keep_checkbox.setChecked(state.keep)
        self.trim_start_spin.setRange(1, segment.num_samples)
        self.trim_end_spin.setRange(1, segment.num_samples)
        self.trim_start_spin.setValue(state.trim_start + 1)
        self.trim_end_spin.setValue(state.trim_end + 1)
        self.task_edit.setText(state.task_name)
        self.instruction_edit.setText(state.language)
        self.scene_edit.setText(state.scene_id)
        self.notes_edit.setPlainText(state.notes)
        self.frame_slider.setRange(1, segment.num_samples)
        self.frame_spin.setRange(1, segment.num_samples)
        if reset_frame:
            self.frame_idx = state.trim_start
        self.frame_idx = int(np.clip(self.frame_idx, 0, segment.num_samples - 1))
        self.frame_slider.setValue(self.frame_idx + 1)
        self.frame_spin.setValue(self.frame_idx + 1)
        self._ignore_widget_updates = False
        self.update_view()

    def update_view(self) -> None:
        segment = self.current_segment
        state = self.current_state
        clamp_review_state(state, segment)
        self.frame_idx = int(np.clip(self.frame_idx, 0, segment.num_samples - 1))
        image = render_frame(segment, state, self.frame_idx, self.index, len(self.segment_infos), self.args)
        self._current_image = image
        qimage = QImage(
            image.data,
            image.shape[1],
            image.shape[0],
            image.strides[0],
            QImage.Format.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimage)
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

        trim_len = current_trim_length(state)
        exported = state.exported_path if state.exported_path else "(not exported)"
        self.status_label.setText(
            f"file={segment.path.name} | frame={self.frame_idx + 1}/{segment.num_samples} | "
            f"trim=[{state.trim_start + 1}, {state.trim_end + 1}] ({trim_len} frames) | "
            f"keep={state.keep} | exported={exported}"
        )
        self.frame_slider.blockSignals(True)
        self.frame_spin.blockSignals(True)
        self.frame_slider.setValue(self.frame_idx + 1)
        self.frame_spin.setValue(self.frame_idx + 1)
        self.frame_slider.blockSignals(False)
        self.frame_spin.blockSignals(False)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._current_image is not None:
            self.update_view()

    def save_current_state(self) -> None:
        self.sync_state_from_widgets()
        save_review_state(self.current_segment.path, self.current_state)
        self.statusBar().showMessage(
            f"Saved review attrs in {self.current_segment.path.name}", 2000
        )

    def sync_state_from_widgets(self) -> None:
        if self._ignore_widget_updates:
            return
        state = self.current_state
        segment = self.current_segment
        state.keep = self.keep_checkbox.isChecked()
        state.trim_start = self.trim_start_spin.value() - 1
        state.trim_end = self.trim_end_spin.value() - 1
        state.task_name = self.task_edit.text().strip()
        state.language = self.instruction_edit.text().strip()
        state.scene_id = self.scene_edit.text().strip()
        state.notes = self.notes_edit.toPlainText().strip()
        clamp_review_state(state, segment)

    def on_metadata_changed(self) -> None:
        if self._ignore_widget_updates:
            return
        self.sync_state_from_widgets()
        save_review_state(self.current_segment.path, self.current_state)
        self.update_view()

    def on_trim_changed(self) -> None:
        if self._ignore_widget_updates:
            return
        self.sync_state_from_widgets()
        state = self.current_state
        if state.trim_start > state.trim_end:
            if self.sender() is self.trim_start_spin:
                state.trim_end = state.trim_start
            else:
                state.trim_start = state.trim_end
        clamp_review_state(state, self.current_segment)
        self._ignore_widget_updates = True
        self.trim_start_spin.setValue(state.trim_start + 1)
        self.trim_end_spin.setValue(state.trim_end + 1)
        self._ignore_widget_updates = False
        self.frame_idx = int(np.clip(self.frame_idx, state.trim_start, state.trim_end))
        self.update_view()

    def on_frame_slider_changed(self, value: int) -> None:
        if self._ignore_widget_updates:
            return
        self.frame_idx = value - 1
        self.update_view()

    def on_frame_spin_changed(self, value: int) -> None:
        if self._ignore_widget_updates:
            return
        self.frame_idx = value - 1
        self.update_view()

    def toggle_playback(self) -> None:
        self.playing = not self.playing
        if self.playing:
            fps = max(1.0, estimate_playback_fps(self.current_segment, self.args.playback_fps))
            self.timer.start(max(1, int(round(1000.0 / fps))))
            self.statusBar().showMessage("Playback started", 1000)
        else:
            self.timer.stop()
            self.statusBar().showMessage("Playback paused", 1000)

    def advance_frame(self) -> None:
        if self.frame_idx < self.current_segment.num_samples - 1:
            self.frame_idx += 1
            self.update_view()
        else:
            self.playing = False
            self.timer.stop()

    def prev_frame(self) -> None:
        self.stop_playback()
        self.frame_idx = max(0, self.frame_idx - 1)
        self.update_view()

    def next_frame(self) -> None:
        self.stop_playback()
        self.frame_idx = min(self.current_segment.num_samples - 1, self.frame_idx + 1)
        self.update_view()

    def set_trim_start_to_current(self) -> None:
        self.stop_playback()
        state = self.current_state
        state.trim_start = self.frame_idx
        if state.trim_start > state.trim_end:
            state.trim_end = state.trim_start
        self.load_current_segment(reset_frame=False)

    def set_trim_end_to_current(self) -> None:
        self.stop_playback()
        state = self.current_state
        state.trim_end = self.frame_idx
        if state.trim_end < state.trim_start:
            state.trim_start = state.trim_end
        self.load_current_segment(reset_frame=False)

    def toggle_keep(self) -> None:
        self.keep_checkbox.setChecked(not self.keep_checkbox.isChecked())

    def _reload_current_segment_from_disk(self, reset_frame: bool) -> None:
        path = self.current_segment.path
        self.segment_infos[self.index] = load_segment_info(path)
        self._current_segment = load_raw_segment(path)
        self.states[self.index] = load_review_state(self.segment_infos[self.index], self.args)
        self.load_current_segment(reset_frame=reset_frame)

    def apply_trim(self) -> None:
        self.stop_playback()
        self.sync_state_from_widgets()
        state = self.current_state
        segment = self.current_segment

        if current_trim_length(state) <= 0:
            QMessageBox.warning(self, "Trim blocked", "Trimmed segment must contain at least one frame.")
            return
        if state.trim_start == 0 and state.trim_end == segment.num_samples - 1:
            self.statusBar().showMessage("Trim already spans the full segment", 2000)
            return

        response = QMessageBox.question(
            self,
            "Apply trim",
            "This will permanently rewrite the raw HDF5 with only the selected frames. Continue?",
        )
        if response != QMessageBox.StandardButton.Yes:
            return

        save_review_state(segment.path, state)
        trim_raw_segment_in_place(segment.path, state, self.args.compression)
        legacy_sidecar = sidecar_path(segment.path)
        if legacy_sidecar.exists():
            legacy_sidecar.unlink()
        self._reload_current_segment_from_disk(reset_frame=True)
        self.statusBar().showMessage(f"Trimmed {segment.path.name}", 3000)

    def prev_file(self) -> None:
        self.stop_playback()
        self.save_current_state()
        self.index = max(0, self.index - 1)
        self._current_segment = None
        self.load_current_segment(reset_frame=True)

    def next_file(self) -> None:
        self.stop_playback()
        self.save_current_state()
        self.index = min(len(self.segment_infos) - 1, self.index + 1)
        self._current_segment = None
        self.load_current_segment(reset_frame=True)

    def stop_playback(self) -> None:
        if self.playing:
            self.playing = False
            self.timer.stop()

    def export_current(self) -> None:
        self.stop_playback()
        self.sync_state_from_widgets()
        state = self.current_state
        segment = self.current_segment

        if not state.keep:
            QMessageBox.warning(self, "Export blocked", "Current segment is marked discard.")
            return
        if not state.task_name.strip():
            QMessageBox.warning(self, "Export blocked", "task_name is required before export.")
            return
        if not state.language.strip():
            QMessageBox.warning(self, "Export blocked", "instruction/language is required before export.")
            return
        if current_trim_length(state) <= 0:
            QMessageBox.warning(self, "Export blocked", "Trimmed segment must contain at least one frame.")
            return

        trimmed = slice_segment(segment, state)
        finalized = finalize_segment(trimmed)
        output_path = resolve_output_path(segment.path, self.args)
        write_finalized_file(output_path, finalized, state, segment, self.args.compression)
        state.exported_path = str(output_path)
        save_review_state(segment.path, state)

        if state.trim_start != 0 or state.trim_end != segment.num_samples - 1:
            trim_raw_segment_in_place(segment.path, state, self.args.compression)
            legacy_sidecar = sidecar_path(segment.path)
            if legacy_sidecar.exists():
                legacy_sidecar.unlink()
            self._reload_current_segment_from_disk(reset_frame=True)
        else:
            self.update_view()

        self.statusBar().showMessage(f"Exported to {output_path}", 4000)

    def closeEvent(self, event) -> None:
        self.stop_playback()
        self.save_current_state()
        super().closeEvent(event)


def main(args: Args) -> None:
    input_path = Path(args.input_path).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    raw_paths = list_raw_segments(input_path)
    if not raw_paths:
        raise FileNotFoundError(f"No raw segment files found under {input_path}")

    segment_infos = [load_segment_info(path) for path in raw_paths]
    states = [load_review_state(segment_info, args) for segment_info in segment_infos]

    app = QApplication(sys.argv)
    window = ReviewWindow(args, segment_infos, states)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main(tyro.cli(Args))
