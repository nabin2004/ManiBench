"""
ManiBench — Extensible Hugging Face Model Catalog
=================================================
Manages models trained on the Manim task (including nabin2004's models,
Qwen-Manimator, AOS-Qwen3, AOS-Qwen2.5, AOS-Gemma4, and baselines).

Features:
  - Strongly typed ModelSpec catalog with family, format, and training method
  - Auto-detection of LoRA vs Merged weights from Hugging Face hub
  - Dynamic registration of new models at runtime
  - Kaggle GPU T4*2 hardware allocation hints (FP16, 4-bit, sharding)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ModelSpec:
    id: str                                    # e.g., "nabin2004/qwen-Manimator-1-merged"
    short_name: str                            # e.g., "qwen-Manimator-1-merged"
    family: str                                # "qwen-manimator", "aos-qwen3", "aos-qwen2.5", "gemma", "baseline"
    format: str                                # "merged", "lora", "gguf", "base"
    base_model: Optional[str] = None           # e.g., "Qwen/Qwen3-8B"
    param_size: str = "8B"                     # e.g., "7B", "8B", "2B", "31B"
    training_method: str = "SFT"               # "Base", "SFT", "DPO", "GRPO", "Narrated"
    description: str = ""
    recommended_quant: Optional[str] = None    # None (FP16), "4bit", "8bit"
    chat_template_family: str = "chatml"       # "chatml", "gemma", "qwen"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def is_lora(self) -> bool:
        return self.format.lower() in ("lora", "adapter", "lora/adapter")

    @property
    def is_merged(self) -> bool:
        return self.format.lower() in ("merged", "base", "full")

    @property
    def is_gguf(self) -> bool:
        return self.format.lower() == "gguf"


# ===========================================================================
# Official Catalog of Models
# ===========================================================================

_DEFAULT_CATALOG: List[ModelSpec] = [
    # ── Family 1: Qwen-Manimator (Qwen3-8B Base) ──────────────────────────
    ModelSpec(
        id="nabin2004/qwen-Manimator-1-merged",
        short_name="qwen-Manimator-1-merged",
        family="qwen-manimator",
        format="merged",
        base_model="Qwen/Qwen3-8B",
        param_size="8B",
        training_method="SFT",
        description="Merged SFT weights for Manimator animation generation",
        chat_template_family="chatml",
    ),
    ModelSpec(
        id="nabin2004/qwen-Manimator-1-grpo-merged",
        short_name="qwen-Manimator-1-grpo-merged",
        family="qwen-manimator",
        format="merged",
        base_model="Qwen/Qwen3-8B",
        param_size="8B",
        training_method="GRPO",
        description="GRPO reinforcement-learned merged model for Manim CE",
        chat_template_family="chatml",
    ),
    ModelSpec(
        id="nabin2004/qwen-Manimator-1-sft",
        short_name="qwen-Manimator-1-sft",
        family="qwen-manimator",
        format="lora",
        base_model="Qwen/Qwen3-8B",
        param_size="8B",
        training_method="SFT",
        description="LoRA SFT adapter for Qwen3-8B base",
        chat_template_family="chatml",
    ),
    ModelSpec(
        id="nabin2004/qwen-Manimator-1-grpo",
        short_name="qwen-Manimator-1-grpo",
        family="qwen-manimator",
        format="lora",
        base_model="Qwen/Qwen3-8B",
        param_size="8B",
        training_method="GRPO",
        description="GRPO RL LoRA adapter checkpoint on Qwen3-8B",
        chat_template_family="chatml",
    ),
    ModelSpec(
        id="nabin2004/qwen-Manimator-1-grpo-clean",
        short_name="qwen-Manimator-1-grpo-clean",
        family="qwen-manimator",
        format="lora",
        base_model="Qwen/Qwen3-8B",
        param_size="8B",
        training_method="GRPO",
        description="Cleaned GRPO RL LoRA adapter on Qwen3-8B",
        chat_template_family="chatml",
    ),
    ModelSpec(
        id="nabin2004/qwen-Manimator-1-gguf",
        short_name="qwen-Manimator-1-gguf",
        family="qwen-manimator",
        format="gguf",
        base_model="nabin2004/qwen-Manimator-1-merged",
        param_size="8B",
        training_method="SFT",
        description="GGUF quantization (Q4_K_M) of Manimator-1",
        chat_template_family="chatml",
    ),

    # ── Family 2: AOS-Qwen3-8B (SFT, DPO, GRPO, Narrated) ──────────────────
    ModelSpec(
        id="nabin2004/AOS-qwen3-8b-narrated-sft-merged",
        short_name="AOS-qwen3-8b-narrated-sft-merged",
        family="aos-qwen3",
        format="merged",
        base_model="Qwen/Qwen3-8B",
        param_size="8B",
        training_method="SFT",
        description="Narrated Manim SFT merged model with voiceover alignment",
        chat_template_family="chatml",
    ),
    ModelSpec(
        id="nabin2004/AOS-qwen3-8b-narrated-merged",
        short_name="AOS-qwen3-8b-narrated-merged",
        family="aos-qwen3",
        format="merged",
        base_model="Qwen/Qwen3-8B",
        param_size="8B",
        training_method="DPO",
        description="DPO preference-tuned merged narrated model",
        chat_template_family="chatml",
    ),
    ModelSpec(
        id="nabin2004/AOS-qwen3-8b-grpo-merged",
        short_name="AOS-qwen3-8b-grpo-merged",
        family="aos-qwen3",
        format="merged",
        base_model="Qwen/Qwen3-8B",
        param_size="8B",
        training_method="GRPO",
        description="GRPO RL merged model for AOS Qwen3",
        chat_template_family="chatml",
    ),
    ModelSpec(
        id="nabin2004/AOS-qwen3-8b-narrated-adapter",
        short_name="AOS-qwen3-8b-narrated-adapter",
        family="aos-qwen3",
        format="lora",
        base_model="Qwen/Qwen3-8B",
        param_size="8B",
        training_method="SFT",
        description="Narrated animation adapter with manim-voiceover support",
        chat_template_family="chatml",
    ),
    ModelSpec(
        id="nabin2004/AOS-qwen3-8b-narrated-dpo",
        short_name="AOS-qwen3-8b-narrated-dpo",
        family="aos-qwen3",
        format="lora",
        base_model="Qwen/Qwen3-8B",
        param_size="8B",
        training_method="DPO",
        description="DPO preference LoRA adapter on Qwen3-8B",
        chat_template_family="chatml",
    ),
    ModelSpec(
        id="nabin2004/AOS-qwen3-8b-grpo",
        short_name="AOS-qwen3-8b-grpo",
        family="aos-qwen3",
        format="lora",
        base_model="Qwen/Qwen3-8B",
        param_size="8B",
        training_method="GRPO",
        description="GRPO LoRA adapter on Qwen3-8B",
        chat_template_family="chatml",
    ),
    ModelSpec(
        id="nabin2004/AOS-qwen3-8b-adapter",
        short_name="AOS-qwen3-8b-adapter",
        family="aos-qwen3",
        format="lora",
        base_model="Qwen/Qwen3-8B",
        param_size="8B",
        training_method="SFT",
        description="Standard SFT LoRA adapter for Qwen3-8B",
        chat_template_family="chatml",
    ),
    ModelSpec(
        id="nabin2004/AOS-qwen3-8b-narrated-sft-gguf",
        short_name="AOS-qwen3-8b-narrated-sft-gguf",
        family="aos-qwen3",
        format="gguf",
        base_model="nabin2004/AOS-qwen3-8b-narrated-sft-merged",
        param_size="8B",
        training_method="SFT",
        description="GGUF quantization of narrated SFT model",
        chat_template_family="chatml",
    ),
    ModelSpec(
        id="nabin2004/AOS-qwen3-8b-narrated-DPO-gguf",
        short_name="AOS-qwen3-8b-narrated-DPO-gguf",
        family="aos-qwen3",
        format="gguf",
        base_model="nabin2004/AOS-qwen3-8b-narrated-merged",
        param_size="8B",
        training_method="DPO",
        description="GGUF quantization of DPO narrated model",
        chat_template_family="chatml",
    ),
    ModelSpec(
        id="nabin2004/AOS-qwen3-8b-grpo-gguf",
        short_name="AOS-qwen3-8b-grpo-gguf",
        family="aos-qwen3",
        format="gguf",
        base_model="nabin2004/AOS-qwen3-8b-grpo-merged",
        param_size="8B",
        training_method="GRPO",
        description="GRPO GGUF format for Ollama / llama.cpp",
        chat_template_family="chatml",
    ),

    # ── Family 3: AOS-Qwen2.5-Coder-7B ─────────────────────────────────────
    ModelSpec(
        id="nabin2004/AOS-qwen25-coder-7b-manim-merged",
        short_name="AOS-qwen25-coder-7b-manim-merged",
        family="aos-qwen2.5",
        format="merged",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        param_size="7B",
        training_method="SFT",
        description="Merged SFT model based on Qwen2.5-Coder-7B",
        chat_template_family="chatml",
    ),
    ModelSpec(
        id="nabin2004/AOS-qwen2.5-coder-7b-manim-merged",
        short_name="AOS-qwen2.5-coder-7b-manim-merged",
        family="aos-qwen2.5",
        format="merged",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        param_size="7B",
        training_method="SFT",
        description="Alternate merged release of Qwen2.5-Coder Manim",
        chat_template_family="chatml",
    ),
    ModelSpec(
        id="nabin2004/qwen2.5-coder-7b-manim-merged",
        short_name="qwen2.5-coder-7b-manim-merged",
        family="aos-qwen2.5",
        format="merged",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        param_size="7B",
        training_method="SFT",
        description="Base merged release of Qwen2.5-Coder Manim",
        chat_template_family="chatml",
    ),
    ModelSpec(
        id="nabin2004/AOS-qwen25-coder-7b-manim-sft",
        short_name="AOS-qwen25-coder-7b-manim-sft",
        family="aos-qwen2.5",
        format="lora",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        param_size="7B",
        training_method="SFT",
        description="SFT LoRA adapter on Qwen2.5-Coder-7B-Instruct",
        chat_template_family="chatml",
    ),
    ModelSpec(
        id="nabin2004/AOS-qwen2.5-coder-7b-manim-sft",
        short_name="AOS-qwen2.5-coder-7b-manim-sft",
        family="aos-qwen2.5",
        format="lora",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        param_size="7B",
        training_method="SFT",
        description="SFT LoRA adapter checkpoint on Qwen2.5-Coder-7B",
        chat_template_family="chatml",
    ),
    ModelSpec(
        id="nabin2004/AOS-qwen25-coder-7b-manim-gguf",
        short_name="AOS-qwen25-coder-7b-manim-gguf",
        family="aos-qwen2.5",
        format="gguf",
        base_model="nabin2004/AOS-qwen25-coder-7b-manim-merged",
        param_size="7B",
        training_method="SFT",
        description="GGUF quantization for Qwen2.5-Coder Manim",
        chat_template_family="chatml",
    ),
    ModelSpec(
        id="nabin2004/AOS-qwen2.5-coder-7b-manim-sft-GGUF",
        short_name="AOS-qwen2.5-coder-7b-manim-sft-GGUF",
        family="aos-qwen2.5",
        format="gguf",
        base_model="nabin2004/AOS-qwen2.5-coder-7b-manim-merged",
        param_size="7B",
        training_method="SFT",
        description="GGUF quantization of SFT adapter merged weights",
        chat_template_family="chatml",
    ),

    # ── Family 4: AOS-Gemma4 (E2B and 31B) ─────────────────────────────────
    ModelSpec(
        id="nabin2004/AOS-gemma4-31b-manim-merged",
        short_name="AOS-gemma4-31b-manim-merged",
        family="gemma",
        format="merged",
        base_model="google/gemma-4-31B-it",
        param_size="31B",
        training_method="SFT",
        description="High-capacity 31B parameter Gemma 4 Manim merged model",
        recommended_quant="4bit",  # Fits in 32GB (2x T4 16GB) under 4-bit
        chat_template_family="gemma",
    ),
    ModelSpec(
        id="nabin2004/AOS-gemma4-31b-manim-sft",
        short_name="AOS-gemma4-31b-manim-sft",
        family="gemma",
        format="lora",
        base_model="google/gemma-4-31B-it",
        param_size="31B",
        training_method="SFT",
        description="31B Gemma 4 LoRA adapter",
        recommended_quant="4bit",
        chat_template_family="gemma",
    ),
    ModelSpec(
        id="nabin2004/AOS-gemma4-manim-sft",
        short_name="AOS-gemma4-manim-sft",
        family="gemma",
        format="lora",
        base_model="google/gemma-4-E2B-it",
        param_size="2B",
        training_method="SFT",
        description="SFT LoRA adapter for Gemma 4 E2B",
        chat_template_family="gemma",
    ),
    ModelSpec(
        id="nabin2004/gemma-4-e2b-manim-lora",
        short_name="gemma-4-e2b-manim-lora",
        family="gemma",
        format="lora",
        base_model="unsloth/gemma-4-e2b-it-unsloth-bnb-4bit",
        param_size="2B",
        training_method="SFT",
        description="Unsloth 4-bit tuned LoRA adapter for Gemma 4 E2B",
        chat_template_family="gemma",
    ),
    ModelSpec(
        id="nabin2004/AOS-gemma4-manim-gguf",
        short_name="AOS-gemma4-manim-gguf",
        family="gemma",
        format="gguf",
        base_model="google/gemma-4-E2B-it",
        param_size="2B",
        training_method="SFT",
        description="Lightweight GGUF model for fast local inference",
        chat_template_family="gemma",
    ),

    # ── Baselines for Paper Benchmark Comparison ───────────────────────────
    ModelSpec(
        id="Qwen/Qwen3-8B",
        short_name="Qwen3-8B-Base",
        family="baseline",
        format="base",
        base_model=None,
        param_size="8B",
        training_method="Base",
        description="Pretrained Qwen3-8B base foundation model",
        chat_template_family="chatml",
    ),
    ModelSpec(
        id="Qwen/Qwen2.5-Coder-7B-Instruct",
        short_name="Qwen2.5-Coder-7B-Instruct",
        family="baseline",
        format="base",
        base_model=None,
        param_size="7B",
        training_method="Base",
        description="Generalist code instruction-tuned baseline",
        chat_template_family="chatml",
    ),
]


class ModelCatalog:
    """Registry and manager for Manim benchmark models."""

    def __init__(self, initial_specs: Optional[List[ModelSpec]] = None):
        self._models: Dict[str, ModelSpec] = {}
        for spec in (initial_specs or _DEFAULT_CATALOG):
            self.register(spec)

    def register(self, spec: ModelSpec) -> None:
        """Register or update a model specification."""
        self._models[spec.id] = spec
        if spec.short_name:
            self._models[spec.short_name] = spec

    def get(self, identifier: str) -> Optional[ModelSpec]:
        """Look up a model by HF ID or short name."""
        return self._models.get(identifier)

    def list(
        self,
        family: Optional[str] = None,
        format_type: Optional[str] = None,
        training_method: Optional[str] = None,
        exclude_gguf: bool = False,
    ) -> List[ModelSpec]:
        """Return unique model specifications matching filters."""
        # De-duplicate by id
        unique_specs = {m.id: m for m in self._models.values()}
        results = list(unique_specs.values())

        if family:
            results = [m for m in results if m.family.lower() == family.lower()]
        if format_type:
            results = [m for m in results if m.format.lower() == format_type.lower()]
        if training_method:
            results = [m for m in results if m.training_method.lower() == training_method.lower()]
        if exclude_gguf:
            results = [m for m in results if not m.is_gguf]

        return results

    def get_or_create(
        self,
        identifier: str,
        base_model: Optional[str] = None,
        format_type: Optional[str] = None,
        family: Optional[str] = None,
        param_size: str = "8B",
    ) -> ModelSpec:
        """
        Get an existing model or auto-create/auto-detect a new ModelSpec.
        This provides zero-config extensibility for any arbitrary HF repo.
        """
        existing = self.get(identifier)
        if existing:
            return existing

        # Auto-detect from Hugging Face if possible
        detected = self.auto_detect(identifier, base_model_override=base_model)
        if detected:
            if family:
                detected.family = family
            if format_type:
                detected.format = format_type
            self.register(detected)
            return detected

        # Fallback manual specification
        short = identifier.split("/")[-1]
        is_lora = format_type == "lora" if format_type else "lora" in short.lower() or "adapter" in short.lower()
        new_spec = ModelSpec(
            id=identifier,
            short_name=short,
            family=family or "custom",
            format="lora" if is_lora else (format_type or "merged"),
            base_model=base_model or ("Qwen/Qwen3-8B" if "qwen" in short.lower() else None),
            param_size=param_size,
            training_method="Custom",
            description=f"User-specified model: {identifier}",
        )
        self.register(new_spec)
        return new_spec

    def auto_detect(self, hf_repo_id: str, base_model_override: Optional[str] = None) -> Optional[ModelSpec]:
        """
        Inspect Hugging Face model repository metadata to automatically determine
        whether it's a LoRA adapter or merged weights, and find its base model.
        """
        try:
            from huggingface_hub import HfApi, hf_hub_download
            api = HfApi()
            info = api.model_info(hf_repo_id)
            filenames = [s.rfilename for s in (info.siblings or [])]
            
            is_lora = "adapter_config.json" in filenames
            is_gguf = any(f.endswith(".gguf") for f in filenames)
            
            base_model = base_model_override
            if is_lora and not base_model:
                try:
                    cfg_path = hf_hub_download(hf_repo_id, "adapter_config.json")
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                        base_model = cfg.get("base_model_name_or_path")
                except Exception:
                    pass

            short_name = hf_repo_id.split("/")[-1]
            family = "custom"
            name_lower = short_name.lower()
            if "manimator" in name_lower:
                family = "qwen-manimator"
            elif "qwen3" in name_lower:
                family = "aos-qwen3"
            elif "qwen25" in name_lower or "qwen2.5" in name_lower:
                family = "aos-qwen2.5"
            elif "gemma" in name_lower:
                family = "gemma"

            method = "SFT"
            if "grpo" in name_lower:
                method = "GRPO"
            elif "dpo" in name_lower:
                method = "DPO"

            param_size = "8B"
            if "31b" in name_lower:
                param_size = "31B"
            elif "7b" in name_lower:
                param_size = "7B"
            elif "2b" in name_lower:
                param_size = "2B"

            fmt = "lora" if is_lora else ("gguf" if is_gguf else "merged")
            quant = "4bit" if "31b" in param_size.lower() else None

            return ModelSpec(
                id=hf_repo_id,
                short_name=short_name,
                family=family,
                format=fmt,
                base_model=base_model or ("Qwen/Qwen3-8B" if "qwen" in name_lower else None),
                param_size=param_size,
                training_method=method,
                description=f"Auto-detected from HuggingFace ({fmt}, {param_size})",
                recommended_quant=quant,
            )
        except Exception:
            return None

    def export_json(self, path: Path | str) -> None:
        """Export catalog as JSON."""
        unique_specs = [m.to_dict() for m in self.list()]
        Path(path).write_text(json.dumps(unique_specs, indent=2), encoding="utf-8")


# Default global singleton catalog instance
catalog = ModelCatalog()
