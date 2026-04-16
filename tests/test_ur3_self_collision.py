import socket
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from gello.robots.ur3_mujoco_server import (
    UR3MujocoServer,
    build_ur3_robotiq_model,
    materialize_ur3_sim_urdf,
)
from gello.safety.ur3_self_collision import UR3SelfCollisionChecker


SAFE_Q = np.array([1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 1.5708])
COLLIDING_Q = np.array([5.674, 2.617, -2.935, -3.428, -4.51, 5.563])
TABLE_COLLIDING_Q = np.array([4.202, 2.122, 5.836, -1.885, 2.09, 6.245])


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class UR3SelfCollisionTest(unittest.TestCase):
    def test_checker_detects_collision_and_safe_pose(self) -> None:
        checker = UR3SelfCollisionChecker()

        safe_result = checker.check(SAFE_Q)
        colliding_result = checker.check(COLLIDING_Q)

        self.assertFalse(safe_result.in_collision)
        self.assertIsNotNone(safe_result.minimum_distance)
        self.assertGreater(safe_result.minimum_distance, 0.0)

        self.assertTrue(colliding_result.in_collision)
        self.assertIsNotNone(colliding_result.minimum_distance)
        self.assertLessEqual(colliding_result.minimum_distance, 0.0)
        self.assertTrue(colliding_result.collision_pairs)

    def test_checker_projects_to_safe_segment(self) -> None:
        checker = UR3SelfCollisionChecker()

        projected = checker.project_to_safe(SAFE_Q, COLLIDING_Q)

        self.assertTrue(checker.is_state_safe(projected))
        self.assertFalse(np.allclose(projected, COLLIDING_Q))
        self.assertGreater(np.linalg.norm(projected - SAFE_Q), 0.0)

    def test_checker_detects_table_wall_collision(self) -> None:
        checker = UR3SelfCollisionChecker(table_collision=True)

        safe_result = checker.check(SAFE_Q)
        table_result = checker.check(TABLE_COLLIDING_Q)
        projected = checker.project_to_safe(SAFE_Q, TABLE_COLLIDING_Q)

        self.assertFalse(safe_result.in_collision)
        self.assertTrue(table_result.in_collision)
        self.assertIn(("environment", "wrist_2_clearance"), table_result.collision_pairs)
        self.assertTrue(checker.is_state_safe(projected))
        self.assertFalse(np.allclose(projected, TABLE_COLLIDING_Q))

    def test_materialized_ur3_sim_urdf_loads_in_mujoco(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        urdf_path = project_root / "third_party" / "ur_description" / "urdf" / "ur3.urdf"
        mesh_dir = (
            project_root / "third_party" / "ur_description" / "meshes" / "ur3" / "collision"
        )
        robotiq_xml_path = (
            project_root / "third_party" / "mujoco_menagerie" / "robotiq_2f85" / "2f85.xml"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            sim_urdf_path = materialize_ur3_sim_urdf(tmp_dir, urdf_path, mesh_dir)
            urdf_model = mujoco.MjModel.from_xml_path(str(sim_urdf_path))
            composite_model = build_ur3_robotiq_model(
                tmp_dir, urdf_path, mesh_dir, robotiq_xml_path
            )

        self.assertEqual(urdf_model.nq, 6)
        self.assertEqual(urdf_model.njnt, 6)
        self.assertGreaterEqual(
            mujoco.mj_name2id(
                composite_model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                "ur3_robot/robotiq_2f85/fingers_actuator",
            ),
            0,
        )
        self.assertGreaterEqual(
            mujoco.mj_name2id(composite_model, mujoco.mjtObj.mjOBJ_GEOM, "table"), 0
        )
        self.assertGreaterEqual(
            mujoco.mj_name2id(composite_model, mujoco.mjtObj.mjOBJ_GEOM, "table_keepout"), 0
        )
        self.assertGreaterEqual(
            mujoco.mj_name2id(composite_model, mujoco.mjtObj.mjOBJ_LIGHT, "key_light"), 0
        )

    def test_build_ur3_model_syncs_table_visuals_with_config(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        urdf_path = project_root / "third_party" / "ur_description" / "urdf" / "ur3.urdf"
        mesh_dir = (
            project_root / "third_party" / "ur_description" / "meshes" / "ur3" / "collision"
        )
        robotiq_xml_path = (
            project_root / "third_party" / "mujoco_menagerie" / "robotiq_2f85" / "2f85.xml"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            composite_model = build_ur3_robotiq_model(
                tmp_dir,
                urdf_path,
                mesh_dir,
                robotiq_xml_path,
                table_height=-0.02,
                table_wall_height=0.01,
            )

        data = mujoco.MjData(composite_model)
        mujoco.mj_forward(composite_model, data)
        table_geom_id = mujoco.mj_name2id(composite_model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        keepout_geom_id = mujoco.mj_name2id(
            composite_model, mujoco.mjtObj.mjOBJ_GEOM, "table_keepout"
        )

        table_top_z = (
            float(data.geom_xpos[table_geom_id][2]) + float(composite_model.geom_size[table_geom_id][2])
        )
        keepout_center_z = float(data.geom_xpos[keepout_geom_id][2])

        self.assertAlmostEqual(table_top_z, -0.02, places=6)
        self.assertAlmostEqual(keepout_center_z, -0.01, places=6)

    def test_ur3_mujoco_server_tracks_7d_joint_state(self) -> None:
        server = UR3MujocoServer(host="127.0.0.1", port=get_free_port())
        target = np.array([0.2, -1.0, 1.1, -1.2, 0.3, 0.4, 0.7])
        try:
            self.assertEqual(server.num_dofs(), 7)
            initial_gripper = server.get_joint_state()[-1]
            for _ in range(20):
                server.command_joint_state(target)
            joint_state = server.get_joint_state()
            np.testing.assert_allclose(joint_state[:6], target[:6], atol=0.1)
            self.assertGreater(joint_state[-1], initial_gripper)
            self.assertLess(joint_state[-1], 1.0)
            np.testing.assert_allclose(
                server.get_observations()["joint_positions"][:6],
                target[:6],
                atol=0.1,
            )
            self.assertGreater(server.get_observations()["gripper_position"][0], initial_gripper)
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
