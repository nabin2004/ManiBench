#!/usr/bin/env python3
"""
Download & Preprocess Reference Videos for ManiBench Visual Similarity
========================================================================
Uses Python SDKs (yt_dlp + opencv-python) to download 3Blue1Brown YouTube reference videos,
trim them to problem scene timestamps, and export standard 4 FPS / 512x512 MP4 clips.
"""

import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATASET_PATH = ROOT_DIR / "ManiBench_Pilot_Dataset.json"
REFERENCES_DIR = ROOT_DIR / "media" / "references"

# Default fallback timestamps (start_sec, end_sec) for 12 pilot problems
DEFAULT_TIMESTAMPS = {
    "MB-001": (15, 60),   # Colliding Blocks Compute pi
    "MB-002": (10, 45),   # Gradient Descent
    "MB-003": (10, 50),   # Convolution
    "MB-004": (12, 55),   # Eigenvectors & Eigenvalues
    "MB-005": (10, 40),   # The Determinant
    "MB-006": (15, 60),   # Central Limit Theorem
    "MB-007": (8, 45),    # Medical Test Paradox
    "MB-008": (10, 50),   # Visualizing Chain Rule
    "MB-009": (15, 60),   # Fundamental Theorem of Calculus
    "MB-010": (12, 50),   # Taylor Series
    "MB-011": (20, 70),   # Hairy Ball Theorem
    "MB-012": (15, 65),   # Windmill Problem
}


def download_and_trim_pythonic(problem: dict) -> Path | None:
    try:
        import cv2
        import yt_dlp
    except ImportError:
        print("ERROR: yt_dlp and opencv-python are required. Run: uv pip install -r requirements-vision.txt")
        sys.exit(1)

    prob_id = problem["id"]
    yt_id = problem.get("youtube_video_id")
    if not yt_id:
        print(f"[{prob_id}] Skipping: no youtube_video_id specified.")
        return None

    t_start, t_end = DEFAULT_TIMESTAMPS.get(prob_id, (0, 30))
    if "reference_start_timestamp" in problem and "reference_end_timestamp" in problem:
        t_start = problem["reference_start_timestamp"]
        t_end = problem["reference_end_timestamp"]

    out_file = REFERENCES_DIR / f"{prob_id.lower()}_ref.mp4"
    if out_file.exists():
        print(f"[{prob_id}] Reference clip already exists at {out_file.name}")
        return out_file

    url = f"https://www.youtube.com/watch?v={yt_id}"
    raw_file = REFERENCES_DIR / f"{prob_id.lower()}_raw.mp4"

    print(f"[{prob_id}] Downloading {url} (clip: {t_start}s - {t_end}s)...")
    try:
        # Download lightweight stream via yt_dlp Python API
        ydl_opts = {
            "format": "b[height<=480][ext=mp4]/b[height<=720][ext=mp4]/mp4/best[ext=mp4]/best",
            "outtmpl": str(raw_file),
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if not raw_file.exists():
            print(f"[{prob_id}] Error: Raw video download failed.")
            return None

        # Process frames using OpenCV (trim, resize to 512x512, sample at 4 FPS)
        cap = cv2.VideoCapture(str(raw_file))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        start_frame = int(t_start * fps)
        end_frame = int(t_end * fps)
        target_fps = 4
        frame_step = max(1, int(round(fps / target_fps)))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(out_file), fourcc, target_fps, (512, 512))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        curr_frame = start_frame
        written_count = 0

        while cap.isOpened() and curr_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if (curr_frame - start_frame) % frame_step == 0:
                resized = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
                out.write(resized)
                written_count += 1
            curr_frame += 1

        cap.release()
        out.release()

        if raw_file.exists():
            raw_file.unlink()

        print(f"[{prob_id}] Prepared standardized reference video ({written_count} frames): {out_file.name}")
        return out_file

    except Exception as e:
        print(f"[{prob_id}] Failed to process video: {e}")
        if raw_file.exists():
            raw_file.unlink()
        return None


def main():
    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    problems = data.get("problems", [])
    print(f"Processing reference videos for {len(problems)} ManiBench problems...")

    success_count = 0
    for prob in problems:
        res = download_and_trim_pythonic(prob)
        if res:
            success_count += 1

    print(f"\nDone! Successfully prepared {success_count}/{len(problems)} reference clips in {REFERENCES_DIR}")


if __name__ == "__main__":
    main()
