"""Render static, non-autoplay evidence comparing two video files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flashfilter.luminance import linear_luminance


def render(original_path: Path, filtered_path: Path, output_dir: Path) -> dict[str, float | int]:
    original_capture=cv2.VideoCapture(str(original_path)); filtered_capture=cv2.VideoCapture(str(filtered_path))
    if not original_capture.isOpened() or not filtered_capture.isOpened(): raise ValueError("Could not open both videos")
    fps=float(original_capture.get(cv2.CAP_PROP_FPS)) or 30.; filtered_fps=float(filtered_capture.get(cv2.CAP_PROP_FPS)) or 30.
    if not np.isclose(fps, filtered_fps, rtol=0, atol=.02):
        raise ValueError("Videos must have matching frame rates")
    previous_original=previous_filtered=None; original_scores=[]; filtered_scores=[]; strongest_score=-1.; strongest=0
    strongest_frames=None; total_modification=0.; modified_pixels=total_pixels=0; length=0
    while True:
        ok_original,bgr_original=original_capture.read(); ok_filtered,bgr_filtered=filtered_capture.read()
        if not ok_original or not ok_filtered: break
        original=cv2.cvtColor(bgr_original,cv2.COLOR_BGR2RGB).astype(np.float32)/255
        filtered=cv2.cvtColor(bgr_filtered,cv2.COLOR_BGR2RGB).astype(np.float32)/255
        if original.shape!=filtered.shape: raise ValueError("Videos must decode to the same shape")
        modification=np.max(np.abs(filtered-original),axis=-1); total_modification+=float(np.abs(filtered-original).sum())
        modified_pixels+=int(np.count_nonzero(modification>1/255)); total_pixels+=modification.size
        if previous_original is not None:
            original_change=np.abs(linear_luminance(original)-linear_luminance(previous_original)); filtered_change=np.abs(linear_luminance(filtered)-linear_luminance(previous_filtered))
            source_score=float(original_change.mean()); original_scores.append(source_score); filtered_scores.append(float(filtered_change.mean()))
            if source_score>strongest_score:
                strongest_score=source_score; strongest=length
                strongest_frames=(previous_original.copy(),original.copy(),original_change.copy(),previous_filtered.copy(),filtered.copy(),filtered_change.copy(),modification.copy())
        previous_original,previous_filtered=original,filtered; length+=1
    original_capture.release(); filtered_capture.release()
    if strongest_frames is None: raise ValueError("Videos need at least two frames")
    original_scores=np.asarray(original_scores); filtered_scores=np.asarray(filtered_scores)
    original_previous,original_current,original_change,filtered_previous,filtered_current,filtered_change,modification=strongest_frames

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    panels = [
        (original_previous, "Source: previous frame", None), (original_current, "Source: current frame", None),
        (original_change, "Source |Δ luminance|", "inferno"), (filtered_previous, "Filtered: previous frame", None),
        (filtered_current, "Filtered: current frame", None), (filtered_change, "Filtered |Δ luminance|", "inferno"),
    ]
    for axis, (image, title, color_map) in zip(axes.flat, panels):
        axis.imshow(image, cmap=color_map, vmin=0, vmax=1)
        axis.set_title(title); axis.axis("off")
    fig.suptitle(f"Strongest source transition at {strongest / fps:.3f} s (frame {strongest})")
    fig.savefig(output_dir / "strongest_transition.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].imshow(original_current); axes[0].set_title("Source frame")
    axes[1].imshow(filtered_current); axes[1].set_title("Filtered frame")
    heat = axes[2].imshow(modification, cmap="magma", vmin=0, vmax=max(float(modification.max()), 1 / 255))
    axes[2].set_title("Absolute RGB modification")
    for axis in axes: axis.axis("off")
    fig.colorbar(heat, ax=axes[2], fraction=.046, label="maximum channel difference")
    fig.savefig(output_dir / "source_filtered_difference.png", dpi=180)
    plt.close(fig)

    times = np.arange(1, length) / fps
    fig, axis = plt.subplots(figsize=(11, 4), constrained_layout=True)
    axis.plot(times, original_scores, label="source", linewidth=1.4)
    axis.plot(times, filtered_scores, label="filtered", linewidth=1.4)
    axis.axvline(strongest / fps, color="black", linestyle="--", linewidth=1, label="shown transition")
    axis.set(xlabel="Time in extracted segment (s)", ylabel="Mean |Δ luminance|", title="Frame-to-frame temporal activity")
    axis.grid(alpha=.25); axis.legend()
    fig.savefig(output_dir / "temporal_activity.png", dpi=180)
    plt.close(fig)

    metrics: dict[str, float | int] = {
        "frames": length,
        "fps": fps,
        "strongest_frame": strongest,
        "strongest_time_seconds": strongest / fps,
        "source_mean_activity": float(original_scores.mean()),
        "filtered_mean_activity": float(filtered_scores.mean()),
        "source_peak_activity": float(original_scores.max()),
        "filtered_peak_activity": float(filtered_scores.max()),
        "filtered_activity_at_source_peak": float(filtered_scores[strongest - 1]),
        "mean_absolute_rgb_modification": total_modification/(total_pixels*3), "modified_pixel_ratio": modified_pixels/total_pixels,
    }
    (output_dir / "frame_difference_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("original", type=Path)
    parser.add_argument("filtered", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.original, args.filtered, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
