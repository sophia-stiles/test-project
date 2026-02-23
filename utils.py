import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from einops import rearrange
import torch.nn.functional as F


def center_crop_video(video, crop_h: int, crop_w: int):
    """Apply center crop to video tensor.

    Args:
        video: Input video tensor.
        crop_h: Target crop height.
        crop_w: Target crop width.

    Returns:
        center_cropped_video: Center-cropped video tensor.

    Raises:
        ValueError: If crop dimensions are larger than video dimensions.
        ValueError: If crop dimensions are not positive.
    """
    h, w = video.shape[-3], video.shape[-2]
    if crop_h > h or crop_w > w:
        raise ValueError(
            f"Crop dimensions ({crop_h}, {crop_w}) cannot be larger than video dimensions ({h}, {w})"
        )
    if crop_h <= 0 or crop_w <= 0:
        raise ValueError(f"Crop dimensions must be positive, got ({crop_h}, {crop_w})")

    top = max((h - crop_h) // 2, 0)
    left = max((w - crop_w) // 2, 0)
    center_cropped_video = video[..., top : top + crop_h, left : left + crop_w, :]
    return center_cropped_video


def resize_video(video, new_h: int, new_w: int):
    """Resize video tensor using bilinear interpolation.

    Args:
        video: Input video tensor.
        new_h: Target height.
        new_w: Target width.

    Returns:
        resized_video: Resized video tensor.
    """
    video = rearrange(video, "... t h w c -> ... t c h w")
    video = F.interpolate(video, size=(new_h, new_w), mode="bilinear", align_corners=False)
    resized_video = rearrange(video, "... t c h w -> ... t h w c")
    return resized_video


def center_crop_resize_video(video, new_h: int, new_w: int):
    """Apply center crop and resize method to video tensor.

    This method first resizes the video so that the short side matches the target resolution,
    then applies a center crop to get the exact target resolution.

    Args:
        video: Input video tensor.
        new_h: Target height.
        new_w: Target width.

    Returns:
        resized_and_center_cropped_video: Resized and center-cropped video tensor.
    """
    h, w = video.shape[-3], video.shape[-2]
    scale = max(new_h / h, new_w / w)
    h = int(round(h * scale))
    w = int(round(w * scale))
    resized_video = resize_video(video, h, w)
    resized_and_center_cropped_video = center_crop_video(resized_video, new_h, new_w)
    return resized_and_center_cropped_video


def convert_video_output_format(video, output_range, dtype):
    """Convert video tensor to the desired output format.

    Args:
        video: Input video tensor.
        output_range: Output range ("unit" or "symmetric").
            - unit: values in range [0.0, 1.0]
            - symmetric: values in range [-1.0, 1.0].
        dtype: Output dtype (default: torch.float32).

    Returns:
        converted_video: Converted video tensor.

    Raises:
        ValueError: If output_range is not recognized.
    """
    if output_range == "unit":
        converted_video = (video.to(torch.float32) / 255.0).to(dtype)
    elif output_range == "symmetric":
        converted_video = (video.to(torch.float32) / 127.5 - 1.0).to(dtype)
    else:
        raise ValueError(f"unknown output_range: {output_range}")
    return converted_video


def select_video_timestamps(start_time_s: float, end_time_s: float, num_frames: int) -> list[float]:
    """Sample uniformly-spaced timestamps in seconds within a time interval [start_time_s, end_time_s].

    Args:
        start_time_s: Interval start time in seconds.
        end_time_s: Interval end time in seconds.
        num_frames: Number of timestamps to sample within the interval.

    Returns:
        List of timestamps (length = num_frames).
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be > 0, got {num_frames}")

    start_time_s = float(start_time_s)
    end_time_s = float(end_time_s)
    duration = end_time_s - start_time_s
    if duration <= 0:
        return []

    timestamps = np.linspace(start_time_s, end_time_s, num_frames).tolist()
    return timestamps


def decode_interval_with_pyav(video_path: Path | str, start_time_s: float, end_time_s: float, num_frames: int):
    """Decode a video clip using PyAV and sample frames by time.

    Some special cases of the return values:
    * If there are no frames in the requested interval, we return an empty tensor and None for the interval.
    * If errors occur, we return an empty tensor and None for the interval.
    * If there are less than num_frames frames in the interval, we repeat the last frame.
    """
    import av  # noqa: PLC0415

    if end_time_s <= start_time_s:
        return torch.empty((0, 0, 0, 0), dtype=torch.uint8), None

    container = None
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]

        start_time_s = max(0.0, float(start_time_s))
        end_time_s = float(end_time_s)

        if stream.time_base is not None:
            seek_offset = int(start_time_s / stream.time_base)  # the offset in units of ticks
            container.seek(seek_offset, stream=stream)
        else:
            container.seek(int(start_time_s * av.time_base))  # the offset in units of ticks

        timestamps = select_video_timestamps(start_time_s, end_time_s, num_frames)
        if len(timestamps) == 0:
            return torch.empty((0, 0, 0, 0), dtype=torch.uint8), None

        frames = []
        frame_ends = []
        frame_starts = []

        for frame in container.decode(stream):
            if frame.time is None:
                continue
            frame_start = float(frame.time)
            assert frame.time_base is not None

            frame_end = frame_start + (float(frame.duration * frame.time_base) if frame.duration else 0.0)
            if frame_end <= start_time_s:
                continue
            if frame_start >= end_time_s:
                break
            frame_nd = frame.to_ndarray(format="rgb24")
            frames.append(frame_nd)
            frame_starts.append(frame_start)
            frame_ends.append(frame_end)
    except Exception:
        return torch.empty((0, 0, 0, 0), dtype=torch.uint8), None
    finally:
        if container is not None:
            container.close()

    if len(frames) == 0:
        return torch.empty((0, 0, 0, 0), dtype=torch.uint8), None

    frame_ends_arr = np.asarray(frame_ends, dtype=np.float64)
    if frame_ends_arr.size > 1 and not np.all(frame_ends_arr[1:] >= frame_ends_arr[:-1]):
        frame_ends_arr = np.maximum.accumulate(frame_ends_arr)
    ts_array = np.asarray(timestamps, dtype=frame_ends_arr.dtype)
    indices = np.searchsorted(frame_ends_arr, ts_array, side="right")
    indices = np.clip(indices, 0, len(frame_ends_arr) - 1).astype(np.int64)
    interval_pts = None
    if indices.size > 0:
        interval_pts = (float(frame_starts[indices[0]]), float(frame_starts[indices[-1]]))
    selected_frames = [frames[i] for i in indices.tolist()]
    return torch.from_numpy(np.stack(selected_frames, axis=0)), interval_pts


def decode(
    video_path: Path | str,
    num_frames: int = 16,
    resolution: Sequence[int] = (288, 288),
    output_range: str = "unit",
    dtype: str | torch.dtype = torch.float32,
    interval: Sequence[float] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Decode a video, optionally within a temporal interval."""
    if interval is None:
        interval = (0.0, float("inf"))
    elif len(interval) != 2:
        raise ValueError(f"interval must have length 2, got {len(interval)}")

    rng = random.Random(42)
    start_time_s = float(interval[0])
    end_time_s = float(interval[1])
    video, interval_pts = decode_interval_with_pyav(video_path, start_time_s, end_time_s, num_frames)

    if video.numel() == 0:
        return video, {}

    video = convert_video_output_format(video, output_range, dtype)
    video = center_crop_resize_video(video, resolution[0], resolution[1])

    meta: dict[str, Any] = {}
    meta["video_path"] = str(video_path)
    if interval is not None:
        meta["interval"] = [float(interval[0]), float(interval[1])]
        if interval_pts is not None:
            meta["interval_pts"] = interval_pts
    return video, meta
