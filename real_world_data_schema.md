# UR3 Real-World Data Schema for Flower Fine-Tuning

This schema is for collecting **UR3** real-world data to fine-tune the current ThinkProprio codebase starting from the pretrained checkpoint:

- `mbreuss/flower_vla_pret`

This version is **UR3-only** and assumes:

- **UR3** single arm
- **parallel gripper**
- **1 external RGB camera**
- **language-conditioned manipulation**
- **16D proprio vector**
- **7D delta end-effector action + gripper**

This is preferred over joint-position actions for UR3 because the upstream Flower action registry supports **7D EEF delta control** directly, while UR3 joint-space would only give 6 joints instead of the 7-joint single-arm schema used in some other settings.

## Recommended setup

- **Robot**: UR3
- **Gripper**: parallel jaw gripper
- **Camera**: 1 fixed external RGB camera
- **Control rate**: **10 Hz** recommended
- **Action space**: **7D delta EEF + gripper**
- **Training action horizon**: **20 steps**
  - at 10 Hz, this corresponds to 2 seconds of future actions

## Dataset layout

```text
<dataset_root>/
  train/
    episode_000001/
      metadata.json
      trajectory.npz
      rgb_primary.mp4
    episode_000002/
      metadata.json
      trajectory.npz
      rgb_primary.mp4
  val/
    episode_000101/
      metadata.json
      trajectory.npz
      rgb_primary.mp4
```

Use explicit `train/` and `val/` folders.

## Episode files

### `metadata.json`

Store static metadata and language annotations.

Example:

```json
{
  "episode_id": "episode_000001",
  "robot_model": "ur3",
  "gripper_type": "parallel_jaw",
  "controller_type": "delta_end_effector",
  "control_frequency_hz": 10,
  "camera_name": "primary_external",
  "camera_resolution": [1280, 720],
  "scene_id": "table_scene_01",
  "task_id": "pick_place_block",
  "language_instruction": "pick up the red block and place it in the bowl",
  "language_instruction_2": "move the red block into the bowl",
  "language_instruction_3": "put the red block in the bowl",
  "success": true,
  "num_steps": 92,
  "time_start_unix": 1776712345.12,
  "notes": "clean teleop demo"
}
```

### `trajectory.npz`

Store all time-aligned low-dimensional arrays.

Required keys:

| key | shape | dtype | description |
|---|---:|---|---|
| `timestamp` | `[T]` | `float64` | timestamps in seconds |
| `joint_position` | `[T, 6]` | `float32` | measured UR3 joint positions in radians |
| `gripper_position` | `[T, 1]` | `float32` | normalized gripper open amount in `[0, 1]` |
| `ee_position` | `[T, 3]` | `float32` | end-effector XYZ in meters, robot base frame |
| `ee_quaternion_xyzw` | `[T, 4]` | `float32` | end-effector orientation in base frame |
| `robot_obs_16` | `[T, 16]` | `float32` | packed proprio vector defined below |
| `action_delta_position` | `[T, 3]` | `float32` | commanded Cartesian XYZ delta |
| `action_delta_euler` | `[T, 3]` | `float32` | commanded roll/pitch/yaw delta |
| `action_gripper` | `[T, 1]` | `float32` | commanded gripper action in `[0, 1]` |
| `action_7d` | `[T, 7]` | `float32` | concatenation of delta position + delta euler + gripper |
| `is_terminal` | `[T]` | `bool` | false except final step |
| `success` | `[T]` | `bool` | repeated episode success label |

Recommended optional keys:

| key | shape | dtype | description |
|---|---:|---|---|
| `joint_velocity` | `[T, 6]` | `float32` | measured joint velocities |
| `ee_pose_7d` | `[T, 7]` | `float32` | XYZ + quaternion XYZW |
| `gripper_width_m` | `[T, 1]` | `float32` | physical gripper width |
| `action_valid` | `[T]` | `bool` | whether the action should be used for training |
| `rgb_primary_timestamp` | `[T]` | `float64` | image timestamps if video alignment is imperfect |

### `rgb_primary.mp4`

- one RGB video per episode
- frames must align with `trajectory.npz`
- use constant frame rate matching `control_frequency_hz` when possible

## UR3 proprio definition: `robot_obs_16`

Use this exact 16D layout:

| indices | meaning |
|---|---|
| `0:6` | UR3 joint positions `q0..q5` |
| `6:7` | gripper open amount |
| `7:10` | end-effector position `(x, y, z)` |
| `10:14` | end-effector quaternion `(qx, qy, qz, qw)` |
| `14:16` | reserved padding, set to `0.0` |

Concretely:

```text
robot_obs_16[t] = [
  joint_position[t, 0:6],
  gripper_position[t, 0:1],
  ee_position[t, 0:3],
  ee_quaternion_xyzw[t, 0:4],
  pad_zeros[t, 0:2]
]
```

### Notes

- Keep quaternion convention fixed as **XYZW**.
- Use the **robot base frame** for `ee_position` and `ee_quaternion_xyzw`.
- Set the last two padded dimensions to zero for all steps.
- Do not mix conventions across episodes.

## UR3 action definition: `action_7d`

Use this exact 7D layout:

| indices | meaning |
|---|---|
| `0:3` | Cartesian position delta `(dx, dy, dz)` |
| `3:6` | Euler rotation delta `(droll, dpitch, dyaw)` |
| `6:7` | gripper open command |

Concretely:

```text
action_7d[t] = [
  dx,
  dy,
  dz,
  droll,
  dpitch,
  dyaw,
  gripper_open
]
```

### Action conventions

- These are **commanded actions**, not measured state differences.
- Units:
  - `dx, dy, dz`: meters
  - `droll, dpitch, dyaw`: radians
  - `gripper_open`: normalized `[0, 1]`
- Use:
  - `1.0` = open
  - `0.0` = closed
- Keep the convention identical in observations and actions.

## Sequence alignment

For each timestep `t`:

- observation at `t` must match the RGB frame at `t`
- `action_7d[t]` is the command issued after observing timestep `t`
- if needed for chunk extension, repeat the final valid action at the end

## Language annotation

Required:

- `language_instruction`

Recommended:

- `language_instruction_2`
- `language_instruction_3`

Guidelines:

- describe the intended task outcome
- keep language concrete and short
- use consistent object names across episodes
- avoid low-level controller wording

Good examples:

- `pick up the sponge and place it in the tray`
- `open the drawer`
- `put the green block into the bowl`

## Success labels

Store:

- episode-level success in `metadata.json`
- per-step repeated success flag in `trajectory.npz`

Recommended initial policy:

- collect and keep both success and failure episodes
- start fine-tuning using **successful demos only**
- later decide whether to include failures for other objectives

## Units and conventions

Use these consistently:

- **joint angles**: radians
- **Cartesian positions**: meters
- **Euler deltas**: radians
- **quaternion**: XYZW
- **gripper**: normalized `[0, 1]`
- **timestamps**: seconds

Do not mix:

- radians and degrees
- meters and millimeters
- XYZW and WXYZ
- opposite gripper conventions

## Mapping into current ThinkProprio codebase

For current training, the intended mapping is:

| dataset field | model batch field |
|---|---|
| decoded `rgb_primary.mp4` frames | `batch["rgb_obs"]["rgb_static"]` |
| no second camera | omit `rgb_gripper` or use a zero / duplicated placeholder if needed |
| `metadata.json -> language_instruction` | `batch["lang_text"]` |
| `trajectory.npz -> robot_obs_16` | `batch["robot_obs"]` |
| `trajectory.npz -> action_7d` | `batch["actions"]` |

## Important compatibility note with `flower_vla_pret`

The HF checkpoint we checked loads very well into the current codebase, but its stored config uses:

- `act_dim: 8`
- `obs_dim: 16`
- `act_seq_len: 20`
- `vlm_path: microsoft/Florence-2-large`
- `use_second_view: false`

For **UR3**, we intentionally choose **7D delta EEF actions** instead of 8D joint actions.

This is still valid because Flower supports multiple action spaces, including:

- **7D `eef_delta`**
- **8D `joint_single`**

So for UR3 fine-tuning, use a config close to:

```yaml
model.vlm_path: microsoft/Florence-2-large
model.use_second_view: false
act_dim: 7
proprio_dims: 16
act_seq_len: 20
model.sampling_type: uniform
model.dit_dim: 1024
model.n_heads: 16
model.n_layers: 12
model.use_rope: true
model.rope_theta: 1000.0
model.use_pre_vlm_selection: false
```

## Recommended first collection plan for UR3

Keep the first version simple:

- one external RGB camera
- one instruction per trajectory
- 10 Hz control
- 7D delta EEF + gripper actions
- 16D padded proprio
- successful clean demonstrations first

That is the cleanest UR3-specific path for fine-tuning from `mbreuss/flower_vla_pret`.
