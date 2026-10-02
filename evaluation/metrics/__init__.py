"""
ManiBench Evaluation — Metrics
================================
Four-tier scoring system:
  1. Executability — binary, does code run in Manim CE?
  2. Version-Conflict Error Rate — static analysis for GL/deprecated patterns
  3. Alignment Score — weighted visual event detection via AST + heuristics
  4. Coverage Score — pedagogical element density via code analysis
"""

from evaluation.metrics.executability import compute_executability
from evaluation.metrics.version_conflict import detect_version_conflicts, detect_specific_conflicts
from evaluation.metrics.alignment import compute_alignment
from evaluation.metrics.coverage import compute_coverage
from evaluation.metrics.visual_similarity import compute_visual_similarity
from evaluation.metrics.mathematical_fidelity import compute_mathematical_fidelity
from evaluation.metrics.code_quality import compute_code_quality
from evaluation.metrics.constraint_adherence import compute_constraint_adherence
from evaluation.metrics.temporal_similarity import compute_temporal_similarity

__all__ = [
    "compute_executability",
    "detect_version_conflicts",
    "detect_specific_conflicts",
    "compute_alignment",
    "compute_coverage",
    "compute_visual_similarity",
    "compute_mathematical_fidelity",
    "compute_code_quality",
    "compute_constraint_adherence",
    "compute_temporal_similarity",
]
