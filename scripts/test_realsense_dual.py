from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
import tyro


@dataclass
class Args:
    serials: Optional[Tuple[str, str]] = None
    width: int = 640
    height: int = 480
    fps: int = 30
    window_name: str = "realsense_dual_test"


def list_devices() -> list[dict[str, str]]:
    ctx = rs.context()
    devices = []
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


def start_pipeline(serial: str, width: int, height: int, fps: int) -> tuple[rs.pipeline, rs.align]:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    return pipeline, align


def render_camera_view(
    frames: rs.composite_frame,
    camera_label: str,
    serial: str,
) -> np.ndarray:
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        raise RuntimeError(f"Missing color/depth frame for {serial}")

    color = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data())
    depth_vis = cv2.convertScaleAbs(depth, alpha=0.03)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    color = color.copy()
    cv2.putText(
        color,
        f"{camera_label}: {serial}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return cv2.hconcat([color, depth_vis])


def main(args: Args) -> None:
    devices = list_devices()
    if not devices:
        raise SystemExit("No RealSense devices found.")

    print(f"Found {len(devices)} RealSense device(s):")
    for idx, dev in enumerate(devices):
        print(
            f"[{idx}] {dev['name']} | serial={dev['serial']} | "
            f"firmware={dev['firmware']} | usb={dev['usb']}"
        )

    if args.serials is None:
        if len(devices) < 2:
            raise SystemExit("Need 2 RealSense devices for this dual-camera test.")
        serials = (devices[0]["serial"], devices[1]["serial"])
    else:
        serials = args.serials

    known_serials = {dev["serial"] for dev in devices}
    missing = [serial for serial in serials if serial not in known_serials]
    if missing:
        raise SystemExit(f"Requested serial(s) not found: {missing}")

    pipelines: list[tuple[str, rs.pipeline, rs.align]] = []
    try:
        for serial in serials:
            pipeline, align = start_pipeline(serial, args.width, args.height, args.fps)
            pipelines.append((serial, pipeline, align))
            print(f"Started stream for {serial}")

        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
        print("Press q or ESC to quit.")

        while True:
            tiles = []
            for idx, (serial, pipeline, align) in enumerate(pipelines, start=1):
                frames = pipeline.wait_for_frames(timeout_ms=2000)
                aligned_frames = align.process(frames)
                tiles.append(render_camera_view(aligned_frames, f"cam{idx}", serial))

            dashboard = cv2.vconcat(tiles)
            cv2.imshow(args.window_name, dashboard)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        for _, pipeline, _ in pipelines:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(tyro.cli(Args))
