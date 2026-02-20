"""
ManiBench Evaluation — Configuration
======================================
Central configuration for models, paths, API settings, and evaluation parameters.
Loads secrets from .env file; everything else is hardcoded for reproducibility.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent          # /home/.../files
DATASET_PATH = ROOT_DIR / "ManiBench_Pilot_Dataset.json"
RAW_CODE_DIR = ROOT_DIR / "raw_code"
RESULTS_DIR = ROOT_DIR / "evaluation" / "results"
LOGS_DIR = ROOT_DIR / "evaluation" / "logs"
GENERATED_CODE_DIR = ROOT_DIR / "evaluation" / "generated_code"

# Ensure output dirs exist
for _d in (RESULTS_DIR, LOGS_DIR, GENERATED_CODE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# OpenRouter API
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://huggingface.co/datasets/nabin2004/ManiBench",
    "X-Title": "ManiBench Evaluation",
}

# ---------------------------------------------------------------------------
# Inference.net API
# ---------------------------------------------------------------------------
INFERENCE_API_KEY = os.getenv("INFERENCE_API_KEY", "")
INFERENCE_BASE_URL = os.getenv("INFERENCE_BASE_URL", "https://api.inference.net/v1")

# ---------------------------------------------------------------------------
# Supported providers
# ---------------------------------------------------------------------------
SUPPORTED_PROVIDERS = ["openrouter", "inference"]

# Per-request settings
REQUEST_TIMEOUT = 120          # seconds
MAX_RETRIES = 3
RETRY_DELAY = 5                # seconds between retries
MAX_TOKENS = 8192              # max generation length


# ---------------------------------------------------------------------------
# Models (OpenRouter model IDs)
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    """Specification for one model to evaluate."""
    id: str                         # OpenRouter model ID
    short_name: str                 # For tables/filenames
    provider: str                   # For paper attribution
    max_tokens: int = MAX_TOKENS
    temperature: float = 0.0        # Deterministic for reproducibility
    top_p: float = 1.0

# Default model roster — edit or override via CLI
DEFAULT_MODELS: list[ModelSpec] = [
    ModelSpec(
        id="openai/gpt-4o-2024-11-20",
        short_name="GPT-4o",
        provider="OpenAI",
    ),
    ModelSpec(
        id="anthropic/claude-sonnet-4",
        short_name="Claude-Sonnet-4",
        provider="Anthropic",
    ),
    ModelSpec(
        id="google/gemini-2.5-pro-preview",
        short_name="Gemini-2.5-Pro",
        provider="Google",
    ),
    ModelSpec(
        id="deepseek/deepseek-r1",
        short_name="DeepSeek-R1",
        provider="DeepSeek",
    ),
    ModelSpec(
        id="meta-llama/llama-4-maverick",
        short_name="Llama-4-Maverick",
        provider="Meta",
    ),
    ModelSpec(
        id="qwen/qwen-2.5-coder-32b-instruct",
        short_name="Qwen-2.5-Coder",
        provider="Alibaba",
    ),
]


# ---------------------------------------------------------------------------
# Evaluation parameters
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    """Runtime configuration for an evaluation run."""
    trials: int = 3                          # runs per (model, problem) pair
    problems: Optional[list[str]] = None     # None = all 12
    models: Optional[list[str]] = None       # None = DEFAULT_MODELS
    prompt_strategy: str = "zero_shot"       # zero_shot | few_shot | cot | constraint
    manim_timeout: int = 60                  # seconds for rendering
    skip_render: bool = False                # skip Manim execution (metrics 1-2 only via static)
    save_video: bool = False                 # keep rendered .mp4 files
    seed: int = 42                           # for reproducibility
    parallel_models: bool = False            # run models in parallel (careful with rate limits)
    provider: str = "openrouter"             # openrouter | inference


# ---------------------------------------------------------------------------
# Version-conflict detection patterns (from reference_code_analysis)
# ---------------------------------------------------------------------------
GL_ONLY_PATTERNS: list[str] = [
    # Imports
    r"from\s+manim_imports_ext",
    r"from\s+manimlib",
    r"from\s+manim_gl",
    r"import\s+manim_imports_ext",
    # Deprecated scene types
    r"class\s+\w+\s*\(\s*GraphScene\s*\)",
    r"class\s+\w+\s*\(\s*ReconfigurableScene\s*\)",
    r"class\s+\w+\s*\(\s*InteractiveScene\s*\)",
    r"class\s+\w+\s*\(\s*TeacherStudentsScene\s*\)",
    r"class\s+\w+\s*\(\s*PiCreatureScene\s*\)",
    r"class\s+\w+\s*\(\s*ExternallyAnimatedScene\s*\)",
    # CONFIG dict pattern
    r"^\s+CONFIG\s*=\s*\{",
    # Deprecated animations
    r"ShowCreation\s*\(",
    r"FadeInFrom\s*\(",
    r"FadeOutAndShift\s*\(",
    # GL-specific objects
    r"PiCreature\s*\(",
    r"PiCreatureSays\s*\(",
    r"Eyes\s*\(",
    r"GlowDot\s*\(",
    r"DieFace\s*\(",
    r"TrueDot\s*\(",
    # GL-specific methods
    r"\.embed\s*\(",
    r"force_skipping\s*\(",
    r"revert_to_original_skipping_status",
    r"apply_depth_test\s*\(",
    r"set_shading\s*\(",
    r"fix_in_frame\s*\(",
    r"set_backstroke\s*\(",
    # GL camera
    r"self\.frame\.",
    r"camera_frame",
    # Deprecated tex
    r"OldTex\s*\(",
    r"OldTexText\s*\(",
    r"TexMobject\s*\(",
    r"TextMobject\s*\(",
    # GL-specific rendering
    r"render_to_movie_file",
    r"set_renderer\s*\(",
]

# Manim CE import validation (expected patterns)
CE_VALID_IMPORTS = [
    r"from\s+manim\s+import",
    r"import\s+manim",
]


# ---------------------------------------------------------------------------
# Inference.net Models
# ---------------------------------------------------------------------------
INFERENCE_MODELS: list[ModelSpec] = [
    ModelSpec(
        id="meta-llama/llama-3.3-70b-instruct-fp8",
        short_name="Llama-3.3-70B",
        provider="Meta (via Inference.net)",
    ),
    ModelSpec(
        id="meta-llama/llama-3.1-8b-instruct",
        short_name="Llama-3.1-8B",
        provider="Meta (via Inference.net)",
    ),
    ModelSpec(
        id="meta-llama/llama-3.1-70b-instruct",
        short_name="Llama-3.1-70B",
        provider="Meta (via Inference.net)",
    ),
    ModelSpec(
        id="mistralai/mistral-nemo-instruct-2407",
        short_name="Mistral-Nemo",
        provider="Mistral (via Inference.net)",
    ),
    ModelSpec(
        id="nvidia/llama-3.1-nemotron-70b-instruct-hf",
        short_name="Nemotron-70B",
        provider="NVIDIA (via Inference.net)",
    ),
    ModelSpec(
        id="qwen/qwen-2.5-coder-32b-instruct",
        short_name="Qwen-2.5-Coder-Inf",
        provider="Alibaba (via Inference.net)",
    ),
]


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------
def get_models_for_provider(provider: str) -> list[ModelSpec]:
    """Return the default model list for a given provider name."""
    if provider == "inference":
        return INFERENCE_MODELS
    return DEFAULT_MODELS          # openrouter is the default


def get_model_by_short_name(
    name: str,
    provider: str = "openrouter",
) -> Optional[ModelSpec]:
    """Look up a model spec by short name (case-insensitive)."""
    roster = get_models_for_provider(provider)
    for m in roster:
        if m.short_name.lower() == name.lower():
            return m
    return None


def get_model_by_id(
    model_id: str,
    provider: str = "openrouter",
) -> Optional[ModelSpec]:
    """Look up a model spec by provider model ID."""
    roster = get_models_for_provider(provider)
    for m in roster:
        if m.id == model_id:
            return m
    return None
