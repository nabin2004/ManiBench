#!/usr/bin/env python3
"""
Test Visual Embedding Similarity Pipeline (DINOv2 + DTW)
=========================================================
Runs a self-similarity test on a reference clip to verify DINOv2 feature extraction
and DTW alignment score.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from evaluation.metrics.visual_similarity import compute_visual_similarity, DINOv2FeatureExtractor

def main():
    ref_dir = ROOT_DIR / "media" / "references"
    ref_files = list(ref_dir.glob("*_ref.mp4"))

    if not ref_files:
        print("No prepared reference videos found in media/references/. Run scripts/download_references.py first.")
        sys.exit(1)

    sample_ref = ref_files[0]
    print(f"Testing visual similarity pipeline on {sample_ref.name}...")

    print("Initializing DINOv2 Feature Extractor (facebook/dinov2-base)...")
    extractor = DINOv2FeatureExtractor()

    # Self-similarity test: compare video against itself (expected score: ~1.0)
    print("Running self-similarity alignment test...")
    res = compute_visual_similarity(sample_ref, sample_ref, extractor=extractor, target_fps=4)

    print("\nVisual Similarity Test Results:")
    print("--------------------------------")
    print(f"Reference Video  : {res.get('ref_video')}")
    print(f"Candidate Video  : {res.get('cand_video')}")
    print(f"Frame Count      : {res.get('ref_frames')} frames")
    print(f"DTW Distance     : {res.get('dtw_distance'):.4f}")
    print(f"Alignment Score  : {res.get('alignment_score'):.4f}")

    if res.get("alignment_score", 0.0) > 0.95:
        print("\nPASSED! Visual similarity metric initialized and verified successfully.")
    else:
        print("\nWARNING: Unexpected alignment score on self-similarity test.")

if __name__ == "__main__":
    main()
