"""
Metric 5: Visual Embedding Similarity (DINOv2 + DTW)
======================================================
Evaluates dynamic visual-logic alignment of generated Manim CE animations (.mp4)
against 3Blue1Brown reference videos using DINOv2 self-supervised visual embeddings
and Dynamic Time Warping (DTW).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Lazy imports for ML dependencies to gracefully degrade if PyTorch/Transformers are missing
_TORCH_AVAILABLE = False
_TRANSFORMERS_AVAILABLE = False
_CV2_AVAILABLE = False
_DTW_AVAILABLE = False

try:
    import torch
    from torchvision import transforms
    _TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from transformers import AutoImageProcessor, AutoModel
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    pass

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    pass

try:
    from scipy.spatial.distance import cdist
    from dtaidistance import dtw
    _DTW_AVAILABLE = True
except ImportError:
    pass


class DINOv2FeatureExtractor:
    """Extract per-frame visual embeddings using DINOv2 ViT-B/14."""

    def __init__(self, model_name: str = "facebook/dinov2-base", device: str | None = None):
        if not (_TORCH_AVAILABLE and _TRANSFORMERS_AVAILABLE):
            raise ImportError(
                "PyTorch and HuggingFace Transformers are required for DINOv2 feature extraction. "
                "Install via: pip install -r requirements-vision.txt"
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract_frame_embedding(self, pil_img: Image.Image) -> np.ndarray:
        """Extract L2-normalized [CLS] token embedding for a single PIL image."""
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for extracting embeddings.")

        with torch.no_grad():
            inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            # Use [CLS] token (first spatial token)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            # L2 normalize
            norm = np.linalg.norm(cls_embedding)
            return cls_embedding / norm if norm > 0 else cls_embedding

    def extract_video_embeddings(self, video_path: str | Path, target_fps: int = 4) -> np.ndarray:
        """Read video, sample frames at target_fps, and extract embedding sequence [N, D]."""
        if not _CV2_AVAILABLE:
            raise ImportError("OpenCV and Pillow are required. Install via: pip install opencv-python Pillow")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(round(fps / target_fps)))

        embeddings = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                emb = self.extract_frame_embedding(pil_img)
                embeddings.append(emb)

            frame_idx += 1

        cap.release()
        return np.array(embeddings) if len(embeddings) > 0 else np.empty((0, 768))


def compute_dtw_alignment(ref_embeddings: np.ndarray, cand_embeddings: np.ndarray) -> dict[str, Any]:
    """
    Compute Dynamic Time Warping distance and cosine similarity between reference
    and candidate embedding sequences.
    """
    if len(ref_embeddings) == 0 or len(cand_embeddings) == 0:
        return {"dtw_distance": 1.0, "alignment_score": 0.0, "cosine_matrix": []}

    # Cosine distance matrix: D[i, j] = 1 - dot(ref[i], cand[j])
    cosine_sim = np.dot(ref_embeddings, cand_embeddings.T)
    cosine_dist = np.clip(1.0 - cosine_sim, 0.0, 2.0)

    # Compute DTW distance using dtaidistance or scipy fallback
    try:
        from dtaidistance import dtw_ndim
        dtw_dist = dtw_ndim.distance(ref_embeddings, cand_embeddings)
    except ImportError:
        # Fallback: simple matrix-based DTW alignment
        N, M = cosine_dist.shape
        cost = np.zeros((N, M))
        cost[0, 0] = cosine_dist[0, 0]
        for i in range(1, N):
            cost[i, 0] = cost[i - 1, 0] + cosine_dist[i, 0]
        for j in range(1, M):
            cost[0, j] = cost[0, j - 1] + cosine_dist[0, j]
        for i in range(1, N):
            for j in range(1, M):
                cost[i, j] = cosine_dist[i, j] + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
        dtw_dist = cost[-1, -1] / (N + M)

    # Convert DTW distance to a normalized similarity score in [0.0, 1.0]
    alignment_score = max(0.0, float(np.exp(-1.5 * dtw_dist)))

    return {
        "dtw_distance": float(dtw_dist),
        "alignment_score": round(alignment_score, 4),
        "ref_frames": len(ref_embeddings),
        "cand_frames": len(cand_embeddings),
    }


_DEFAULT_EXTRACTOR: "DINOv2FeatureExtractor | None" = None


def get_default_extractor(model_name: str = "facebook/dinov2-base") -> "DINOv2FeatureExtractor":
    """
    Return a process-wide cached DINOv2 extractor.

    Loading ViT-B/14 takes several seconds and ~350 MB of VRAM; without caching it
    would be reloaded for *every* trial of every model.
    """
    global _DEFAULT_EXTRACTOR
    if _DEFAULT_EXTRACTOR is None:
        _DEFAULT_EXTRACTOR = DINOv2FeatureExtractor(model_name=model_name)
    return _DEFAULT_EXTRACTOR


def compute_visual_similarity(
    ref_video_path: str | Path,
    cand_video_path: str | Path,
    extractor: DINOv2FeatureExtractor | None = None,
    target_fps: int = 4,
) -> dict[str, Any]:
    """
    Main entry point for computing DINOv2 + DTW Visual Embedding Similarity.

    NOTE the argument order: (reference, candidate) - both may be either a
    rendered .mp4 or a directory of frames.
    """
    ref_path = Path(ref_video_path)
    cand_path = Path(cand_video_path)

    if not ref_path.exists():
        return {"error": f"Reference video not found at {ref_path}", "alignment_score": None}
    if not cand_path.exists():
        return {"error": f"Candidate video not found at {cand_path}", "alignment_score": None}

    if extractor is None and not _CV2_AVAILABLE:
        return {"error": "OpenCV (cv2) is required for video visual similarity - "
                         "install it with: pip install opencv-python-headless",
                "alignment_score": None}

    if extractor is None:
        try:
            extractor = get_default_extractor()
        except Exception as e:
            return {"error": f"Failed to initialize DINOv2 feature extractor: {e}", "alignment_score": None}

    ref_emb = extractor.extract_video_embeddings(ref_path, target_fps=target_fps)
    cand_emb = extractor.extract_video_embeddings(cand_path, target_fps=target_fps)

    res = compute_dtw_alignment(ref_emb, cand_emb)
    res["ref_video"] = ref_path.name
    res["cand_video"] = cand_path.name
    return res
