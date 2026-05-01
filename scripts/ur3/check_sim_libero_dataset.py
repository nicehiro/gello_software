from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import tyro

REQUIRED_ROOT_ATTRS = {
    "schema_version",
    "env_name",
    "robot",
    "gripper",
    "teleop_device",
    "control_freq",
    "record_freq",
    "action_space",
    "camera_names",
    "num_demos",
    "total",
}

REQUIRED_DEMO_ATTRS = {
    "num_samples",
    "task_name",
    "language",
    "success",
    "operator",
    "episode_id",
    "collection_date",
    "action_space",
    "joint_action_space",
    "state_encoding",
    "notes",
}

REQUIRED_OBS_KEYS = {
    "agentview_rgb",
    "eye_in_hand_rgb",
    "joint_states",
    "joint_velocities",
    "gripper_states",
    "ee_pos",
    "ee_ori",
    "ee_states",
    "timestamps",
}

REQUIRED_DEMO_KEYS = {
    "obs",
    "actions",
    "actions_joint_position",
    "robot_states",
    "states",
    "rewards",
    "dones",
}


@dataclass
class Args:
    dataset_path: str


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_demo(demo_name: str, demo_group: h5py.Group) -> int:
    missing_attrs = REQUIRED_DEMO_ATTRS - set(demo_group.attrs.keys())
    assert_true(not missing_attrs, f"{demo_name}: missing attrs {sorted(missing_attrs)}")

    missing_keys = REQUIRED_DEMO_KEYS - set(demo_group.keys())
    assert_true(not missing_keys, f"{demo_name}: missing keys {sorted(missing_keys)}")

    obs_group = demo_group["obs"]
    missing_obs_keys = REQUIRED_OBS_KEYS - set(obs_group.keys())
    assert_true(not missing_obs_keys, f"{demo_name}: missing obs keys {sorted(missing_obs_keys)}")

    num_samples = int(demo_group.attrs["num_samples"])
    lengths = {f"obs/{key}": int(obs_group[key].shape[0]) for key in obs_group.keys()}
    for key in ["actions", "actions_joint_position", "robot_states", "states", "rewards", "dones"]:
        lengths[key] = int(demo_group[key].shape[0])
    bad_lengths = {key: value for key, value in lengths.items() if value != num_samples}
    assert_true(not bad_lengths, f"{demo_name}: length mismatches {bad_lengths}, num_samples={num_samples}")

    assert_true(obs_group["agentview_rgb"].dtype == np.uint8, f"{demo_name}: agentview_rgb must be uint8")
    assert_true(obs_group["eye_in_hand_rgb"].dtype == np.uint8, f"{demo_name}: eye_in_hand_rgb must be uint8")
    assert_true(obs_group["joint_states"].shape[1] == 7, f"{demo_name}: joint_states must have dim 7")
    assert_true(obs_group["joint_velocities"].shape[1] == 7, f"{demo_name}: joint_velocities must have dim 7")
    assert_true(obs_group["gripper_states"].shape[1] == 1, f"{demo_name}: gripper_states must have dim 1")
    assert_true(obs_group["ee_pos"].shape[1] == 3, f"{demo_name}: ee_pos must have dim 3")
    assert_true(obs_group["ee_ori"].shape[1] == 4, f"{demo_name}: ee_ori must have dim 4")
    assert_true(obs_group["ee_states"].shape[1] == 7, f"{demo_name}: ee_states must have dim 7")
    assert_true(demo_group["actions"].shape[1] == 7, f"{demo_name}: actions must have dim 7")
    assert_true(demo_group["actions_joint_position"].shape[1] == 7, f"{demo_name}: actions_joint_position must have dim 7")
    assert_true(demo_group["robot_states"].shape[1] == 22, f"{demo_name}: robot_states must have dim 22")
    assert_true(demo_group["states"].shape[1] == 22, f"{demo_name}: states must have dim 22")

    timestamps = obs_group["timestamps"][()]
    assert_true(np.all(np.diff(timestamps) >= 0), f"{demo_name}: timestamps are not monotonic")
    assert_true(int(demo_group["dones"][()].sum()) == 1, f"{demo_name}: dones must sum to 1")
    assert_true(int(demo_group["rewards"][()].sum()) == 1, f"{demo_name}: rewards must sum to 1")
    assert_true(int(demo_group["dones"][-1]) == 1, f"{demo_name}: final done must be 1")
    assert_true(int(demo_group["rewards"][-1]) == 1, f"{demo_name}: final reward must be 1")

    ee_states = obs_group["ee_states"][()]
    ee_concat = np.concatenate([obs_group["ee_pos"][()], obs_group["ee_ori"][()]], axis=-1)
    assert_true(np.allclose(ee_states, ee_concat), f"{demo_name}: ee_states != concat(ee_pos, ee_ori)")

    robot_state_concat = np.concatenate(
        [
            obs_group["joint_states"][()],
            obs_group["joint_velocities"][()],
            obs_group["ee_pos"][()],
            obs_group["ee_ori"][()],
            obs_group["gripper_states"][()],
        ],
        axis=-1,
    )
    assert_true(np.allclose(demo_group["robot_states"][()], robot_state_concat), f"{demo_name}: robot_states layout mismatch")
    assert_true(np.allclose(demo_group["states"][()], demo_group["robot_states"][()]), f"{demo_name}: states != robot_states")

    for key in ["joint_states", "joint_velocities", "gripper_states", "ee_pos", "ee_ori", "ee_states"]:
        assert_true(np.isfinite(obs_group[key][()]).all(), f"{demo_name}: non-finite values in obs/{key}")
    for key in ["actions", "actions_joint_position", "robot_states", "states"]:
        assert_true(np.isfinite(demo_group[key][()]).all(), f"{demo_name}: non-finite values in {key}")

    print(
        f"[PASS] {demo_name}: T={num_samples}, "
        f"lang={demo_group.attrs['language']!r}, task={demo_group.attrs['task_name']!r}"
    )
    return num_samples


def main(args: Args) -> None:
    dataset_path = Path(args.dataset_path).expanduser()
    assert_true(dataset_path.exists(), f"Dataset file not found: {dataset_path}")

    with h5py.File(dataset_path, "r") as h5_file:
        assert_true("data" in h5_file, "Missing top-level 'data' group")
        data_group = h5_file["data"]

        missing_root_attrs = REQUIRED_ROOT_ATTRS - set(data_group.attrs.keys())
        assert_true(not missing_root_attrs, f"Missing root attrs {sorted(missing_root_attrs)}")

        demo_names = sorted(data_group.keys())
        assert_true(len(demo_names) > 0, "No demos found in /data")

        total = 0
        for demo_name in demo_names:
            total += check_demo(demo_name, data_group[demo_name])

        assert_true(int(data_group.attrs["num_demos"]) == len(demo_names), "Root num_demos mismatch")
        assert_true(int(data_group.attrs["total"]) == total, "Root total mismatch")

        print()
        print(f"Dataset: {dataset_path}")
        print(f"schema_version={data_group.attrs['schema_version']}")
        print(f"num_demos={len(demo_names)} total_frames={total}")
        print("All checks passed.")


if __name__ == "__main__":
    main(tyro.cli(Args))
