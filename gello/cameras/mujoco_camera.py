from threading import Lock
from typing import Optional, Tuple

import cv2
import mujoco
import numpy as np

from gello.cameras.camera import CameraDriver


class MujocoCamera(CameraDriver):
    def __init__(
        self,
        model: mujoco.MjModel,
        camera: int | str,
        width: int = 128,
        height: int = 128,
    ) -> None:
        self._model = model
        self._camera = camera
        self._width = width
        self._height = height
        self._renderer: mujoco.Renderer | None = None
        self._frame_lock = Lock()
        self._latest_image = np.zeros((height, width, 3), dtype=np.uint8)
        self._latest_depth = np.zeros((height, width, 1), dtype=np.float32)

    def __repr__(self) -> str:
        return (
            f"MujocoCamera(camera={self._camera}, width={self._width}, height={self._height})"
        )

    def _ensure_renderer(self) -> mujoco.Renderer:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self._model, height=self._height, width=self._width
            )
        return self._renderer

    def render(self, data: mujoco.MjData) -> None:
        renderer = self._ensure_renderer()

        renderer.disable_depth_rendering()
        renderer.update_scene(data, camera=self._camera)
        image = renderer.render().copy()

        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=self._camera)
        depth = renderer.render().copy()[:, :, None].astype(np.float32)
        renderer.disable_depth_rendering()

        with self._frame_lock:
            self._latest_image = image
            self._latest_depth = depth

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        with self._frame_lock:
            image = self._latest_image.copy()
            depth = self._latest_depth.copy()

        if img_size is not None:
            image = cv2.resize(image, img_size, interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth[:, :, 0], img_size, interpolation=cv2.INTER_LINEAR)[
                :, :, None
            ]

        return image, depth
