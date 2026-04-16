# Real-World UR3 Dataset Schema (LIBERO-style, Hugging Face friendly)

## Scope

This document defines the dataset schema for real-world UR3 teleoperation data collected with GELLO.

Goals:
- use a LIBERO-style HDF5 layout
- keep the dataset easy to inspect and convert
- preserve both raw teleoperation commands and policy-friendly actions
- support upload and later conversion for Hugging Face / LeRobot workflows

## High-level decisions

- **Storage format**: HDF5
- **Organization**: one HDF5 file per task
- **Episodes per file**: multiple demonstrations per task, stored as `demo_0`, `demo_1`, ...
- **Observation modalities**:
  - agent-view RGB
  - eye-in-hand RGB
  - optional depth for both cameras
  - proprioception
  - timestamps
- **Action modalities**:
  - end-effector delta action
  - joint-space action
- **Canonical `actions` dataset**:
  - `actions` stores the **EE-delta action** for better LIBERO-style compatibility
  - `actions_joint_position` stores the corresponding joint-space action actually commanded or executed

---

## File organization

Recommended layout:

```text
dataset_root/
  ur3_libero_real_v1/
    pick_place_block_demo.hdf5
    open_drawer_demo.hdf5
    stack_cups_demo.hdf5
```

Each file contains all demonstrations for one task.

Example:
- `pick_place_block_demo.hdf5`
  - `demo_0`
  - `demo_1`
  - `demo_2`
  - ...

If a task grows very large, the same schema can be sharded later:

```text
pick_place_block_demo_000.hdf5
pick_place_block_demo_001.hdf5
```

---

## Top-level HDF5 structure

```text
/data                                      group
  attrs:
    schema_version
    env_name
    robot
    gripper
    teleop_device
    control_freq
    action_space
    camera_names
    num_demos
    total
    problem_info
    env_args
    bddl_file_name
    bddl_file_content

/camera_info                               group (optional but recommended)
  /agentview
    intrinsics
    extrinsics
    serial
    resolution
    fps
  /eye_in_hand
    intrinsics
    extrinsics
    serial
    resolution
    fps
```

---

## Per-episode structure

Each episode is stored as `/data/demo_i`.

```text
/data/demo_0                               group
  attrs:
    num_samples
    task_name
    language
    success
    operator
    scene_id
    episode_id
    collection_date
    action_space
    joint_action_space
    state_encoding
    notes

  /obs
    agentview_rgb
    eye_in_hand_rgb
    agentview_depth
    eye_in_hand_depth
    joint_states
    joint_velocities
    gripper_states
    ee_pos
    ee_ori
    ee_states
    timestamps

  actions
  actions_joint_position
  robot_states
  states
  rewards
  dones
```

---

## Canonical semantics

### Episodes
- one `demo_i` group = one teleoperated demonstration
- all demos in one file correspond to the same task
- task-level natural language is repeated per episode to keep each episode self-contained

### Actions
To support both LIBERO-style learning and exact replay/debugging:

- `actions`: canonical policy action, **EE-delta**
- `actions_joint_position`: joint-space action, typically the commanded joint target

This lets us:
- train policies with EE-delta actions
- retain the original low-level joint command for replay, auditing, and alternative policies

### States
Because this is real-world data rather than simulation:

- `robot_states`: canonical stacked robot state vector
- `states`: same as `robot_states` for compatibility

Root or episode metadata should mark:
- `state_encoding = "robot_only"`

---

## Dataset keys and shapes

### `/data` attrs

| Key | Type | Description |
|---|---|---|
| `schema_version` | string | Schema identifier, e.g. `ur3_libero_real_v1` |
| `env_name` | string | Environment name, e.g. `UR3RealWorld` |
| `robot` | string | Robot name, e.g. `ur3` |
| `gripper` | string | Gripper name, e.g. `robotiq_2f` |
| `teleop_device` | string | Teleop device, e.g. `gello` |
| `control_freq` | int | Nominal control loop frequency in Hz |
| `action_space` | string | Canonical action type, set to `ee_delta` |
| `camera_names` | JSON/string | Example: `["agentview", "eye_in_hand"]` |
| `num_demos` | int | Number of episodes in the file |
| `total` | int | Total number of frames across all demos |
| `problem_info` | string | Optional task description / metadata |
| `env_args` | JSON string | Optional environment/task configuration |
| `bddl_file_name` | string | Optional BDDL file name |
| `bddl_file_content` | string | Optional BDDL file contents |

### `/data/demo_i` attrs

| Key | Type | Description |
|---|---|---|
| `num_samples` | int | Number of timesteps in the episode |
| `task_name` | string | Canonical task identifier |
| `language` | string | Natural language instruction |
| `success` | uint8/bool | Episode success flag |
| `operator` | string | Person collecting the demo |
| `scene_id` | string | Scene/setup identifier |
| `episode_id` | string | Unique episode ID |
| `collection_date` | string | ISO-style date/time string |
| `action_space` | string | `ee_delta` |
| `joint_action_space` | string | `joint_position` |
| `state_encoding` | string | `robot_only` |
| `notes` | string | Optional free-form notes |

### `/data/demo_i/obs/*`

| Key | Shape | Dtype | Description |
|---|---:|---|---|
| `agentview_rgb` | `[T, H, W, 3]` | `uint8` | External camera RGB |
| `eye_in_hand_rgb` | `[T, H, W, 3]` | `uint8` | Wrist camera RGB |
| `agentview_depth` | `[T, H, W, 1]` | `float32` | External camera depth in meters |
| `eye_in_hand_depth` | `[T, H, W, 1]` | `float32` | Wrist camera depth in meters |
| `joint_states` | `[T, 7]` | `float32` | 6 arm joints + 1 gripper scalar |
| `joint_velocities` | `[T, 7]` | `float32` | Joint velocities + gripper velocity if available |
| `gripper_states` | `[T, 1]` | `float32` | Gripper opening / normalized state |
| `ee_pos` | `[T, 3]` | `float32` | End-effector position |
| `ee_ori` | `[T, 4]` | `float32` | End-effector orientation quaternion |
| `ee_states` | `[T, 7]` | `float32` | Concatenation of `ee_pos` and `ee_ori` |
| `timestamps` | `[T]` | `float64` | Per-step timestamps in seconds |

### `/data/demo_i/*`

| Key | Shape | Dtype | Description |
|---|---:|---|---|
| `actions` | `[T, 7]` | `float32` | Canonical EE-delta action: 6D delta + 1D gripper |
| `actions_joint_position` | `[T, 7]` | `float32` | Joint-space action: 6 arm joints + 1 gripper |
| `robot_states` | `[T, D]` | `float32` | Canonical concatenated robot state vector |
| `states` | `[T, D]` | `float32` | Same as `robot_states` for compatibility |
| `rewards` | `[T]` | `uint8` or `float32` | Sparse reward, typically final success reward |
| `dones` | `[T]` | `uint8` or `bool` | Episode termination flags |

---

## Action definition

### 1. Canonical action: `actions` (EE-delta)

```text
[T, 7] = [dx, dy, dz, d_rx, d_ry, d_rz, gripper]
```

Recommended semantics:
- translational delta in end-effector frame or base frame
- rotational delta in axis-angle / rotation vector form
- final dimension controls the gripper

This is the action intended for LIBERO-style or LeRobot-style policy training.

### 2. Raw / replay action: `actions_joint_position`

```text
[T, 7] = [q0, q1, q2, q3, q4, q5, g]
```

Recommended semantics:
- 6 UR3 joint targets in radians
- 1 gripper scalar, ideally normalized to `[0, 1]`

This action is intended for:
- replaying the robot command history
- debugging and auditability
- future joint-space policy training

### Synchronization rule
Both action arrays must have the same length `T` and correspond to the same timestep index.

---

## State definition

### `robot_states`
Recommended concatenation:

```text
robot_states[t] = concat(
  joint_states[t],
  joint_velocities[t],
  ee_pos[t],
  ee_ori[t],
  gripper_states[t],
)
```

Suggested dimension:
- `7 + 7 + 3 + 4 + 1 = 22`

So typically:

```text
robot_states: [T, 22]
states: [T, 22]
```

If gripper state is already included inside `joint_states`, we can still keep this explicit layout for clarity.

---

## Reward and done convention

For teleoperated imitation data:

- `dones[t] = 0` for all non-terminal steps
- `dones[T-1] = 1`

Recommended sparse rewards:
- successful episode: `rewards[T-1] = 1`, others `0`
- failed episode: all rewards `0`

Also store success as episode metadata:
- `success = 1` or `0`

---

## Camera metadata

Recommended optional camera metadata:

```text
/camera_info/agentview/intrinsics         [3, 3] float32
/camera_info/agentview/extrinsics         [4, 4] float32
/camera_info/eye_in_hand/intrinsics       [3, 3] float32
/camera_info/eye_in_hand/extrinsics       [4, 4] float32
```

Additional useful attrs or datasets:
- camera serial number
- image resolution
- frame rate
- depth scale
- distortion coefficients

If calibration changes over time, store calibration per file shard or per episode.

---

## Naming and compatibility notes

### LIBERO-style naming kept
We intentionally use:
- `agentview_rgb`
- `eye_in_hand_rgb`
- `agentview_depth`
- `eye_in_hand_depth`
- `joint_states`
- `gripper_states`
- `ee_pos`
- `ee_ori`
- `ee_states`
- `actions`
- `rewards`
- `dones`

This preserves a familiar LIBERO-like structure.

### Real-world extensions added
We additionally include:
- `actions_joint_position`
- `joint_velocities`
- `timestamps`
- richer episode metadata
- camera calibration info

These are important for real-world robotics and later dataset conversion.

---

## Hugging Face / LeRobot export mapping

Recommended mapping for later export:

| This schema | LeRobot-style field |
|---|---|
| `obs/agentview_rgb` | `observation.images.image` |
| `obs/eye_in_hand_rgb` | `observation.images.image2` |
| selected proprio from `robot_states` or `obs/*` | `observation.state` |
| `actions` | `action` |
| `language` | task / instruction text |

For LeRobot training, we may later define a compact `observation.state` vector, for example:
- `ee_pos`
- EE rotation in axis-angle
- gripper state

That export step should be separate from raw collection.

---

## Recommended HDF5 write settings

- RGB images: `uint8`, compressed
- depth images: `float32`, compressed
- state/action arrays: `float32`
- timestamps: `float64`
- use chunking along the time dimension

Example recommendations:
- image chunks: `[1, H, W, C]` or small time windows
- vector chunks: `[256, D]`
- compression: `gzip` or `lzf`

---

## Example tree

```text
pick_place_block_demo.hdf5
├── /data
│   ├── attrs
│   │   ├── schema_version = "ur3_libero_real_v1"
│   │   ├── env_name = "UR3RealWorld"
│   │   ├── robot = "ur3"
│   │   ├── teleop_device = "gello"
│   │   ├── control_freq = 100
│   │   ├── action_space = "ee_delta"
│   │   ├── num_demos = 24
│   │   └── total = 18342
│   ├── /demo_0
│   │   ├── attrs
│   │   │   ├── task_name = "pick_place_block"
│   │   │   ├── language = "Pick up the red block and place it into the blue bowl."
│   │   │   ├── success = 1
│   │   │   └── num_samples = 742
│   │   ├── /obs
│   │   │   ├── agentview_rgb
│   │   │   ├── eye_in_hand_rgb
│   │   │   ├── agentview_depth
│   │   │   ├── eye_in_hand_depth
│   │   │   ├── joint_states
│   │   │   ├── joint_velocities
│   │   │   ├── gripper_states
│   │   │   ├── ee_pos
│   │   │   ├── ee_ori
│   │   │   ├── ee_states
│   │   │   └── timestamps
│   │   ├── actions
│   │   ├── actions_joint_position
│   │   ├── robot_states
│   │   ├── states
│   │   ├── rewards
│   │   └── dones
│   └── /demo_1
└── /camera_info
    ├── /agentview
    └── /eye_in_hand
```

---

## Open implementation notes

Before implementing the recorder, we still need to pin down:

1. exact EE-delta convention
   - base-frame or tool-frame deltas
   - axis-angle vs quaternion delta
2. whether `actions_joint_position` is the commanded target or measured next joint state
3. whether `timestamps` are wall-clock, monotonic, or both
4. whether camera calibration is static per file or logged per episode

Current recommendation:
- `actions`: EE-delta in a fixed documented convention
- `actions_joint_position`: commanded joint target
- `timestamps`: monotonic seconds since episode start
- optional additional absolute wall-clock timestamp in metadata
