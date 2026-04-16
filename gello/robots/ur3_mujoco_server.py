import shutil
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional

import mujoco
import mujoco.viewer
import numpy as np
from dm_control import mjcf

from gello.cameras.mujoco_camera import MujocoCamera
from gello.robots.sim_robot import attach_hand_to_arm
from gello.safety.ur3_self_collision import UR3SelfCollisionChecker
from gello.safety.ur3_self_collision_config import (
    DEFAULT_TABLE_HEIGHT,
    DEFAULT_TABLE_WALL_HEIGHT,
)
from gello.zmq_core.camera_node import ZMQServerCamera
from gello.zmq_core.robot_node import ZMQServerRobot

DEFAULT_START_JOINTS = np.array(
    [1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 1.5708, 0.0], dtype=float
)
UR3_ARM_DOFS = 6
UR3_INTERFACE_DOFS = 7
UR3_ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
ROBOTIQ_CTRL_MAX = 255.0
ROBOTIQ_DRIVER_CLOSED = 0.8
DEFAULT_EYE_IN_HAND_CAMERA_PORT = 5000
DEFAULT_AGENTVIEW_CAMERA_PORT = 5001
DEFAULT_SIDEVIEW_CAMERA_PORT = 5002
DEFAULT_CAMERA_WIDTH = 128
DEFAULT_CAMERA_HEIGHT = 128
TABLE_SURFACE_LOCAL_Z = 0.36
TABLE_GEOM_CENTER_LOCAL_Z = 0.34
TABLE_KEEPOUT_HALF_HEIGHT = 0.0025
DEFAULT_VIEWER_DISTANCE = 1.35
DEFAULT_VIEWER_AZIMUTH = 35.0
DEFAULT_VIEWER_ELEVATION = -38.0
DEFAULT_VIEWER_LOOKAT = np.array([-0.08, 0.2, 0.28], dtype=float)
DEFAULT_OBJECT_FRICTION = "1.8 0.1 0.02"
DEFAULT_PAD_FRICTION_1 = "2.2"
DEFAULT_PAD_FRICTION_2 = "1.8"
ARM_ACTUATOR_NAMES = (
    "shoulder_pan_act",
    "shoulder_lift_act",
    "elbow_act",
    "wrist_1_act",
    "wrist_2_act",
    "wrist_3_act",
)
ARM_ACTUATOR_KP = (2200.0, 2200.0, 1800.0, 700.0, 700.0, 500.0)
ARM_ACTUATOR_KV = (120.0, 120.0, 100.0, 40.0, 40.0, 30.0)
ARM_ACTUATOR_FORCE_LIMIT = (220.0, 220.0, 180.0, 70.0, 70.0, 50.0)


def materialize_ur3_sim_urdf(
    output_dir: str | Path,
    source_urdf_path: str | Path,
    source_mesh_dir: str | Path,
) -> Path:
    output_dir = Path(output_dir)
    source_urdf_path = Path(source_urdf_path)
    source_mesh_dir = Path(source_mesh_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for mesh_path in source_mesh_dir.glob("*.stl"):
        shutil.copy2(mesh_path, output_dir / mesh_path.name)

    tree = ET.parse(source_urdf_path)
    root = tree.getroot()
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib["filename"]
        mesh.attrib["filename"] = Path(filename).name.replace(".dae", ".stl")

    output_path = output_dir / "ur3_sim.urdf"
    tree.write(output_path)
    return output_path


def materialize_ur3_mjcf(
    output_dir: str | Path,
    source_urdf_path: str | Path,
    source_mesh_dir: str | Path,
) -> Path:
    output_dir = Path(output_dir)
    urdf_path = materialize_ur3_sim_urdf(output_dir, source_urdf_path, source_mesh_dir)
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    mjcf_path = output_dir / "ur3_sim.xml"
    mujoco.mj_saveLastXML(str(mjcf_path), model)
    return mjcf_path


def add_arm_position_actuators(arm_mjcf: mjcf.RootElement) -> None:
    for actuator_name, joint_name, kp, kv, force_limit in zip(
        ARM_ACTUATOR_NAMES,
        UR3_ARM_JOINT_NAMES,
        ARM_ACTUATOR_KP,
        ARM_ACTUATOR_KV,
        ARM_ACTUATOR_FORCE_LIMIT,
    ):
        arm_mjcf.actuator.add(
            "position",
            name=actuator_name,
            joint=joint_name,
            kp=str(kp),
            kv=str(kv),
            ctrlrange="-6.2831 6.2831",
            forcerange=f"-{force_limit} {force_limit}",
        )


def add_default_wrist_camera(arm_mjcf: mjcf.RootElement) -> None:
    wrist_3_link = arm_mjcf.find("body", "wrist_3_link")
    if wrist_3_link is None:
        raise ValueError("wrist_3_link not found in generated UR3 MJCF")

    wrist_3_link.add(
        "site",
        name="attachment_site",
        pos="0 0 -0.007",
        quat="1 0 0 0",
    )
    wrist_3_link.add(
        "camera",
        name="eye_in_hand",
        mode="fixed",
        pos="0 -0.085 -0.02",
        euler="2.70526 0 0",
        fovy="60",
    )


def add_default_scene_objects(arena: mjcf.RootElement, table_body) -> None:
    arena.asset.add("material", name="cube_red", rgba="0.85 0.2 0.2 1")
    arena.asset.add("material", name="cube_blue", rgba="0.2 0.35 0.9 1")
    arena.asset.add("material", name="cylinder_green", rgba="0.2 0.75 0.3 1")
    arena.asset.add("material", name="tray_material", rgba="0.25 0.25 0.3 1")

    tray = table_body.add("body", name="tray", pos="-0.02 0.26 0.39")
    tray.add(
        "geom",
        name="tray_bottom",
        type="box",
        pos="0 0 -0.02",
        size="0.085 0.12 0.01",
        material="tray_material",
    )
    for name, x, y, sx, sy in (
        ("tray_wall_left", 0.0, 0.11, 0.085, 0.01),
        ("tray_wall_right", 0.0, -0.11, 0.085, 0.01),
        ("tray_wall_front", 0.075, 0.0, 0.01, 0.12),
        ("tray_wall_back", -0.075, 0.0, 0.01, 0.12),
    ):
        tray.add(
            "geom",
            name=name,
            type="box",
            pos=f"{x} {y} 0",
            size=f"{sx} {sy} 0.03",
            material="tray_material",
        )

    cube_red = arena.worldbody.add("body", name="cube_red_body", pos="-0.18 0.18 0.03")
    cube_red.add("freejoint")
    cube_red.add(
        "geom",
        name="cube_red_geom",
        type="box",
        size="0.02 0.02 0.02",
        mass="0.08",
        friction=DEFAULT_OBJECT_FRICTION,
        material="cube_red",
    )

    cube_blue = arena.worldbody.add("body", name="cube_blue_body", pos="0.32 0.18 0.03")
    cube_blue.add("freejoint")
    cube_blue.add(
        "geom",
        name="cube_blue_geom",
        type="box",
        size="0.02 0.02 0.02",
        mass="0.08",
        friction=DEFAULT_OBJECT_FRICTION,
        material="cube_blue",
    )

    cylinder_green = arena.worldbody.add(
        "body", name="cylinder_green_body", pos="-0.26 0.34 0.022"
    )
    cylinder_green.add("freejoint")
    cylinder_green.add(
        "geom",
        name="cylinder_green_visual",
        type="cylinder",
        size="0.022 0.022",
        material="cylinder_green",
        contype="0",
        conaffinity="0",
        group="2",
    )
    cylinder_green.add(
        "geom",
        name="cylinder_green_collision",
        type="box",
        size="0.018 0.018 0.022",
        mass="0.10",
        friction="0.9 0.05 0.01",
        rgba="0 0 0 0",
        group="3",
        solref="0.01 1",
        solimp="0.95 0.99 0.001",
    )


def add_default_scene_cameras(arena: mjcf.RootElement) -> None:
    arena.worldbody.add(
        "camera",
        name="agentview",
        mode="fixed",
        pos="0.9 -0.85 0.8",
        xyaxes="0.884 0.468 0 -0.288 0.543 0.789",
        fovy="50",
    )
    arena.worldbody.add(
        "camera",
        name="sideview",
        mode="fixed",
        pos="0.15 -1.05 0.6",
        xyaxes="0.962 -0.275 0 0.124 0.433 0.893",
        fovy="50",
    )


def build_ur3_robotiq_model(
    output_dir: str | Path,
    source_urdf_path: str | Path,
    source_mesh_dir: str | Path,
    robotiq_xml_path: str | Path,
    table_height: float = DEFAULT_TABLE_HEIGHT,
    table_wall_height: float = DEFAULT_TABLE_WALL_HEIGHT,
) -> mujoco.MjModel:
    mjcf_path = materialize_ur3_mjcf(output_dir, source_urdf_path, source_mesh_dir)
    arm_mjcf = mjcf.from_path(str(mjcf_path))
    add_arm_position_actuators(arm_mjcf)
    add_default_wrist_camera(arm_mjcf)

    gripper_mjcf = mjcf.from_path(str(robotiq_xml_path))
    attach_hand_to_arm(arm_mjcf, gripper_mjcf)

    arena = mjcf.RootElement()
    arena.asset.add("material", name="floor_material", rgba="0.18 0.18 0.18 1")
    arena.asset.add("material", name="table_material", rgba="0.55 0.42 0.3 1")
    arena.asset.add("material", name="keepout_material", rgba="0.8 0.1 0.1 0.2")
    arena.worldbody.add(
        "light",
        name="key_light",
        pos="1.5 -1.0 2.5",
        dir="-0.4 0.2 -1.0",
        directional="true",
        diffuse="0.9 0.9 0.9",
        specular="0.2 0.2 0.2",
        castshadow="true",
    )
    arena.worldbody.add(
        "light",
        name="fill_light",
        pos="-1.0 1.0 1.5",
        dir="0.2 -0.3 -1.0",
        directional="true",
        diffuse="0.35 0.35 0.35",
        specular="0.05 0.05 0.05",
        castshadow="false",
    )
    arena.worldbody.add(
        "geom",
        name="floor",
        type="plane",
        pos="0 0 -0.75",
        size="0 0 1",
        material="floor_material",
    )
    table_body_z = float(table_height) - TABLE_SURFACE_LOCAL_Z
    table_keepout_z = TABLE_SURFACE_LOCAL_Z + float(table_wall_height)
    table_body = arena.worldbody.add(
        "body", name="table_body", pos=f"0 0 {table_body_z}"
    )
    table_body.add(
        "geom",
        name="table",
        type="box",
        pos=f"0 0 {TABLE_GEOM_CENTER_LOCAL_Z}",
        size="0.45 0.6 0.02",
        material="table_material",
    )
    table_body.add(
        "geom",
        name="table_keepout",
        type="box",
        pos=f"0 0 {table_keepout_z}",
        size=f"0.45 0.6 {TABLE_KEEPOUT_HALF_HEIGHT}",
        material="keepout_material",
        contype="0",
        conaffinity="0",
    )
    for leg_name, x, y in (
        ("table_leg_front_left", 0.38, 0.53),
        ("table_leg_front_right", 0.38, -0.53),
        ("table_leg_back_left", -0.38, 0.53),
        ("table_leg_back_right", -0.38, -0.53),
    ):
        table_body.add(
            "geom",
            name=leg_name,
            type="box",
            pos=f"{x} {y} -0.01",
            size="0.03 0.03 0.33",
            material="table_material",
        )

    arena.worldbody.attach(arm_mjcf)
    add_default_scene_objects(arena, table_body)
    add_default_scene_cameras(arena)

    assets: Dict[str, bytes] = {}
    for asset in arena.asset.all_children():
        if asset.tag == "mesh":
            mesh_file = asset.file
            assets[mesh_file.get_vfs_filename()] = mesh_file.contents

    return mujoco.MjModel.from_xml_string(arena.to_xml_string(), assets)


class ServerThread(threading.Thread):
    def __init__(self, server: object):
        super().__init__()
        self._server = server

    def run(self) -> None:
        self._server.serve()


def _find_named_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    count: int,
    name: str,
) -> int:
    for object_id in range(count):
        object_name = mujoco.mj_id2name(model, object_type, object_id)
        if object_name == name or object_name.endswith(f"/{name}"):
            return object_id
    available = [
        mujoco.mj_id2name(model, object_type, object_id) for object_id in range(count)
    ]
    raise ValueError(f"{object_type.name} {name!r} not found. Available: {available}")


def _maybe_find_named_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    count: int,
    name: str,
) -> int | None:
    for object_id in range(count):
        object_name = mujoco.mj_id2name(model, object_type, object_id)
        if object_name == name or object_name.endswith(f"/{name}"):
            return object_id
    return None


def find_camera_id(model: mujoco.MjModel, name: str) -> int:
    return _find_named_id(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam, name)


def find_joint_id(model: mujoco.MjModel, name: str) -> int:
    return _find_named_id(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt, name)


def find_body_id(model: mujoco.MjModel, name: str) -> int:
    return _find_named_id(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody, name)


def maybe_find_site_id(model: mujoco.MjModel, name: str) -> int | None:
    return _maybe_find_named_id(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite, name)


def find_actuator_id(model: mujoco.MjModel, name: str) -> int:
    return _find_named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu, name)


class UR3MujocoServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5556,
        start_joints: np.ndarray = DEFAULT_START_JOINTS,
        collision_checker: Optional[UR3SelfCollisionChecker] = None,
        collision_debug: bool = False,
        print_joints: bool = False,
        enable_cameras: bool = False,
        eye_in_hand_camera_port: int = DEFAULT_EYE_IN_HAND_CAMERA_PORT,
        agentview_camera_port: int = DEFAULT_AGENTVIEW_CAMERA_PORT,
        sideview_camera_port: int = DEFAULT_SIDEVIEW_CAMERA_PORT,
        camera_width: int = DEFAULT_CAMERA_WIDTH,
        camera_height: int = DEFAULT_CAMERA_HEIGHT,
        show_camera_windows: bool = False,
        table_height: float | None = None,
        table_wall_height: float | None = None,
    ) -> None:
        self._temp_dir = tempfile.TemporaryDirectory(prefix="ur3-mujoco-")
        project_root = Path(__file__).resolve().parents[2]
        source_urdf_path = (
            project_root / "third_party" / "ur_description" / "urdf" / "ur3.urdf"
        )
        source_mesh_dir = (
            project_root
            / "third_party"
            / "ur_description"
            / "meshes"
            / "ur3"
            / "collision"
        )
        robotiq_xml_path = (
            project_root / "third_party" / "mujoco_menagerie" / "robotiq_2f85_v4" / "2f85.xml"
        )

        visual_table_height = (
            float(table_height)
            if table_height is not None
            else (
                collision_checker.table_height
                if collision_checker is not None
                else DEFAULT_TABLE_HEIGHT
            )
        )
        visual_table_wall_height = (
            float(table_wall_height)
            if table_wall_height is not None
            else (
                collision_checker.table_wall_height
                if collision_checker is not None
                else DEFAULT_TABLE_WALL_HEIGHT
            )
        )

        self._model = build_ur3_robotiq_model(
            self._temp_dir.name,
            source_urdf_path,
            source_mesh_dir,
            robotiq_xml_path,
            table_height=visual_table_height,
            table_wall_height=visual_table_wall_height,
        )
        self._model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self._data = mujoco.MjData(self._model)
        self._collision_checker = collision_checker
        self._collision_debug = collision_debug
        self._print_joints = print_joints
        self._show_camera_windows = show_camera_windows
        self._state_lock = threading.Lock()

        self._arm_dofs = UR3_ARM_DOFS
        self._interface_dofs = UR3_INTERFACE_DOFS
        self._joint_state = np.asarray(start_joints, dtype=float).copy()
        if self._joint_state.shape != (self._interface_dofs,):
            raise ValueError(
                f"Expected start_joints shape {(self._interface_dofs,)}, got {self._joint_state.shape}"
            )
        self._joint_cmd = self._joint_state.copy()
        self._joint_velocities = np.zeros(self._interface_dofs, dtype=float)
        self._ee_pos_quat = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=float)

        self._arm_joint_ids = np.array(
            [find_joint_id(self._model, name) for name in UR3_ARM_JOINT_NAMES],
            dtype=int,
        )
        self._arm_qpos_adrs = self._model.jnt_qposadr[self._arm_joint_ids]
        self._arm_dof_adrs = self._model.jnt_dofadr[self._arm_joint_ids]

        self._arm_actuator_ids = np.array(
            [find_actuator_id(self._model, name) for name in ARM_ACTUATOR_NAMES],
            dtype=int,
        )
        self._fingers_actuator_id = find_actuator_id(
            self._model, "fingers_actuator"
        )
        self._pinch_site_id = maybe_find_site_id(self._model, "pinch")
        self._right_driver_qpos_adr = self._model.jnt_qposadr[
            find_joint_id(self._model, "right_driver_joint")
        ]
        self._wrist_body_id = find_body_id(self._model, "wrist_3_link")

        with self._state_lock:
            self._data.qpos[self._arm_qpos_adrs] = self._joint_cmd[: self._arm_dofs]
            self._data.qvel[self._arm_dof_adrs] = 0.0
            self._data.ctrl[self._arm_actuator_ids] = self._joint_cmd[: self._arm_dofs]
            self._data.ctrl[self._fingers_actuator_id] = 0.0
            mujoco.mj_forward(self._model, self._data)
            self._apply_joint_cmd_locked(substeps=100)

        self._server = ZMQServerRobot(self, port=port, host=host)
        self._server_thread = ServerThread(self._server)
        self._cameras: list[tuple[str, MujocoCamera]] = []
        self._camera_servers: list[ZMQServerCamera] = []
        self._camera_threads: list[ServerThread] = []
        self._stopped = False
        if enable_cameras:
            camera_specs = (
                ("eye_in_hand", eye_in_hand_camera_port),
                ("agentview", agentview_camera_port),
                ("sideview", sideview_camera_port),
            )
            for camera_name, camera_port in camera_specs:
                camera = MujocoCamera(
                    self._model,
                    camera=find_camera_id(self._model, camera_name),
                    width=camera_width,
                    height=camera_height,
                )
                self._cameras.append((camera_name, camera))
                server = ZMQServerCamera(camera, port=camera_port, host=host)
                self._camera_servers.append(server)
                self._camera_threads.append(ServerThread(server))
                print(
                    f"Starting sim camera {camera_name} on {host}:{camera_port} at {camera_width}x{camera_height}"
                )

    def num_dofs(self) -> int:
        return self._interface_dofs

    def get_joint_state(self) -> np.ndarray:
        return self._joint_state.copy()

    def _current_gripper_position_locked(self) -> float:
        driver_pos = float(self._data.qpos[self._right_driver_qpos_adr])
        return float(np.clip(driver_pos / ROBOTIQ_DRIVER_CLOSED, 0.0, 1.0))

    def _update_observations_locked(self) -> None:
        ee_pos = np.zeros(3)
        ee_quat = np.array([1.0, 0.0, 0.0, 0.0])
        if self._pinch_site_id is not None:
            ee_pos = self._data.site_xpos[self._pinch_site_id].copy()
            ee_mat = self._data.site_xmat[self._pinch_site_id].reshape(3, 3)
            mujoco.mju_mat2Quat(ee_quat, ee_mat.reshape(-1))
        elif self._wrist_body_id >= 0:
            ee_pos = self._data.xpos[self._wrist_body_id].copy()
            ee_mat = self._data.xmat[self._wrist_body_id].reshape(3, 3)
            mujoco.mju_mat2Quat(ee_quat, ee_mat.reshape(-1))

        gripper_pos = self._current_gripper_position_locked()
        arm_positions = self._data.qpos[self._arm_qpos_adrs].copy()
        arm_velocities = self._data.qvel[self._arm_dof_adrs].copy()
        self._joint_state = np.concatenate([arm_positions, [gripper_pos]])
        self._joint_velocities[: self._arm_dofs] = arm_velocities
        self._ee_pos_quat = np.concatenate([ee_pos, ee_quat])

    def _apply_joint_cmd_locked(self, substeps: int) -> None:
        arm_cmd = self._joint_cmd[: self._arm_dofs]
        gripper_ctrl = float(np.clip(self._joint_cmd[-1], 0.0, 1.0) * ROBOTIQ_CTRL_MAX)
        prev_gripper = float(self._joint_state[-1])
        for _ in range(max(substeps, 1)):
            self._data.ctrl[self._arm_actuator_ids] = arm_cmd
            self._data.ctrl[self._fingers_actuator_id] = gripper_ctrl
            mujoco.mj_step(self._model, self._data)
        self._joint_velocities[-1] = self._current_gripper_position_locked() - prev_gripper
        self._update_observations_locked()

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        joint_state = np.asarray(joint_state, dtype=float)
        if joint_state.shape != (self._interface_dofs,):
            raise ValueError(
                f"Expected joint state of length {self._interface_dofs}, got {len(joint_state)}."
            )

        with self._state_lock:
            if self._collision_checker is not None:
                desired_arm = joint_state[: self._collision_checker.arm_dofs]
                current_arm = self._data.qpos[self._arm_qpos_adrs].copy()
                safe_arm = self._collision_checker.project_to_safe(current_arm, desired_arm)
                if self._collision_debug and not np.allclose(safe_arm, desired_arm):
                    result = self._collision_checker.check(desired_arm)
                    print(
                        "Self-collision filter clipped UR3 sim command:",
                        {
                            "current": np.round(current_arm, 4).tolist(),
                            "desired": np.round(desired_arm, 4).tolist(),
                            "safe": np.round(safe_arm, 4).tolist(),
                            "pairs": list(result.collision_pairs),
                            "minimum_distance": result.minimum_distance,
                        },
                    )
                joint_state = joint_state.copy()
                joint_state[: self._collision_checker.arm_dofs] = safe_arm

            self._joint_cmd = joint_state.copy()
            self._apply_joint_cmd_locked(substeps=6)

    def get_observations(self) -> Dict[str, np.ndarray]:
        with self._state_lock:
            return {
                "joint_positions": self._joint_state.copy(),
                "joint_velocities": self._joint_velocities.copy(),
                "ee_pos_quat": self._ee_pos_quat.copy(),
                "gripper_position": np.array([self._joint_state[-1]], dtype=float),
            }

    def _render_cameras_locked(self) -> None:
        for _, camera in self._cameras:
            camera.render(self._data)

    def _show_camera_windows_if_enabled(self) -> None:
        if not self._show_camera_windows or not self._cameras:
            return

        import cv2

        for camera_name, camera in self._cameras:
            image, _ = camera.read()
            cv2.imshow(f"sim_{camera_name}", image[:, :, ::-1])
        cv2.waitKey(1)

    def _configure_default_viewer_camera(self, viewer) -> None:
        viewer.cam.distance = DEFAULT_VIEWER_DISTANCE
        viewer.cam.azimuth = DEFAULT_VIEWER_AZIMUTH
        viewer.cam.elevation = DEFAULT_VIEWER_ELEVATION
        viewer.cam.lookat[:] = DEFAULT_VIEWER_LOOKAT

    def serve(self) -> None:
        self._server_thread.start()
        for thread in self._camera_threads:
            thread.start()
        try:
            with mujoco.viewer.launch_passive(self._model, self._data) as viewer:
                self._configure_default_viewer_camera(viewer)
                while viewer.is_running():
                    step_start = time.time()
                    with self._state_lock:
                        self._apply_joint_cmd_locked(substeps=1)
                        self._render_cameras_locked()
                        if self._print_joints:
                            print(self._joint_state)
                        viewer.sync()
                    self._show_camera_windows_if_enabled()
                    dt = max(1.0 / 120.0 - (time.time() - step_start), 0.0)
                    if dt > 0.0:
                        time.sleep(dt)
        finally:
            self.stop()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._server.stop()
        for server in self._camera_servers:
            server.stop()
        if self._server_thread.is_alive():
            self._server_thread.join(timeout=1.0)
        for thread in self._camera_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        if self._show_camera_windows and self._cameras:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass
        self._temp_dir.cleanup()

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
