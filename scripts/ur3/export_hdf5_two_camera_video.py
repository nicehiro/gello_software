from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
import tyro


@dataclass
class Args:
    dataset_path: str
    output_path: str = ""
    demo: str = "demo_0"
    fps: float = 0.0
    gap: int = 8
    header_height: int = 36


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _estimate_fps(demo_group: h5py.Group, fallback_fps: float) -> float:
    if fallback_fps > 0:
        return fallback_fps

    if "record_freq" in demo_group.parent.attrs:
        return float(demo_group.parent.attrs["record_freq"])

    timestamps = demo_group["obs"]["timestamps"][()]
    if len(timestamps) >= 2:
        dt = np.diff(timestamps)
        dt = dt[dt > 0]
        if len(dt) > 0:
            return float(1.0 / np.median(dt))

    return 20.0


def main(args: Args) -> None:
    dataset_path = Path(args.dataset_path).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    output_path = Path(args.output_path).expanduser() if args.output_path else dataset_path.with_name(
        f"{dataset_path.stem}_{args.demo}_two_cams.mp4"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(dataset_path, "r") as h5_file:
        demo_group = h5_file["data"][args.demo]
        obs_group = demo_group["obs"]

        agentview = obs_group["agentview_rgb"]
        eye_in_hand = obs_group["eye_in_hand_rgb"]

        num_frames = min(len(agentview), len(eye_in_hand))
        if num_frames == 0:
            raise ValueError(f"No RGB frames found in {dataset_path} /data/{args.demo}")

        frame_height, frame_width = agentview.shape[1], agentview.shape[2]
        if eye_in_hand.shape[1] != frame_height or eye_in_hand.shape[2] != frame_width:
            raise ValueError("The two camera streams have different resolutions")

        fps = _estimate_fps(demo_group, args.fps)
        canvas_height = frame_height + args.header_height
        canvas_width = frame_width * 2 + args.gap

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (canvas_width, canvas_height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")

        try:
            for frame_idx in range(num_frames):
                left = _to_bgr(agentview[frame_idx])
                right = _to_bgr(eye_in_hand[frame_idx])

                canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
                canvas[args.header_height :, :frame_width] = left
                canvas[args.header_height :, frame_width + args.gap :] = right

                cv2.putText(
                    canvas,
                    "agentview",
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    "eye_in_hand",
                    (frame_width + args.gap + 12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    f"frame {frame_idx + 1}/{num_frames}",
                    (canvas_width - 170, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (220, 220, 220),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(canvas)
        finally:
            writer.release()

    print(f"Saved video to {output_path}")
    print(f"fps={fps:.3f}, frames={num_frames}, size={canvas_width}x{canvas_height}")


if __name__ == "__main__":
    main(tyro.cli(Args))
