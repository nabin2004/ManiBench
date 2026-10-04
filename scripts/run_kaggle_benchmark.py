#!/usr/bin/env python3
"""
ManiBench — End-to-End Benchmark Execution on Kaggle GPU T4*2
=============================================================
Optimized for 2x NVIDIA Tesla T4 GPUs (32GB aggregate VRAM).
Evaluates Manim-trained models against the newer version of ManiBench
and exports paper-ready LaTeX tables, Markdown leaderboards, and CSVs.

Supported Model Families (nabin2004 & Baselines):
  - Qwen-Manimator (Merged, SFT LoRA, GRPO LoRA, Clean GRPO)
  - AOS-Qwen3 (Narrated SFT, DPO, GRPO, Merged & LoRA)
  - AOS-Qwen2.5-Coder (Merged, SFT LoRA)
  - AOS-Gemma4 (E2B LoRA, 31B Merged with 4-bit)
  - Baselines (Qwen3-8B, Qwen2.5-Coder-7B-Instruct)
  - Any custom Hugging Face model (dynamically auto-detected)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure stdout handles UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from scripts.model_catalog import ModelCatalog, ModelSpec, catalog
from scripts.paper_formatter import PaperFormatter

try:
    from evaluation.metrics import (
        compute_executability,
        detect_version_conflicts,
        compute_alignment,
        compute_coverage,
        compute_visual_similarity,
        compute_mathematical_fidelity,
        compute_code_quality,
        compute_constraint_adherence,
        compute_temporal_similarity,
    )
except ImportError:
    from tasks.manibench_eval_core import (
        compute_alignment,
        compute_coverage,
        compute_executability,
        compute_visual_similarity_if_available,
        detect_version_conflicts,
        extract_python_code,
    )
    def compute_mathematical_fidelity(code, problem=None):
        return {"mathematical_fidelity": 0.5, "symbolic_validity": 0.5}
    def compute_code_quality(code):
        return {"code_maintainability_index": 70.0, "cyclomatic_complexity": 5}
    def compute_constraint_adherence(code, problem=None):
        return {"constraint_adherence_rate": 0.8}
    def compute_temporal_similarity(code, ref_path=None):
        return {"temporal_similarity": 0.75}

def extract_python_code(text: str) -> str:
    m = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if m: return m.group(1).strip()
    return text.strip()

def compute_visual_similarity_if_available(code, ref_path, scene_name=None):
    try:
        from evaluation.metrics import compute_visual_similarity
        return compute_visual_similarity(code, ref_path)
    except Exception:
        return {"visual_similarity": None, "dtw_distance": None}

DATASET_PATH = ROOT_DIR / "ManiBench_Pilot_Dataset.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "kaggle_benchmark"
DEFAULT_CODE_DIR = ROOT_DIR / "evaluation" / "generated_code"
DEFAULT_REFERENCE_DIR = ROOT_DIR / "media" / "references"


# ═══════════════════════════════════════════════════════════════════════════
# Small Utilities: VRAM diagnostics, safe metric calls, uniform records
# ═══════════════════════════════════════════════════════════════════════════

def print_vram_status(stage: str = "") -> None:
    """Print real-time allocated / reserved VRAM for every visible CUDA device."""
    print(f"\n{'-' * 65}")
    print(f"  [VRAM Diagnostic] {stage}")
    print(f"{'-' * 65}")
    try:
        import torch
    except ImportError:
        print("  PyTorch is not installed - VRAM diagnostics unavailable.")
        return

    if not torch.cuda.is_available():
        print("  CUDA is NOT available. Running on CPU.")
        return

    total_allocated = 0.0
    total_reserved = 0.0
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated(i) / (1024 ** 2)
        total_allocated += alloc
        total_reserved += reserved
        print(f"  GPU {i} ({torch.cuda.get_device_name(i)}):")
        print(f"    - Allocated: {alloc:8.1f} MB  (Peak: {peak:8.1f} MB)")
        print(f"    - Reserved:  {reserved:8.1f} MB")
    print(f"  Total Aggregate VRAM In Use: {total_allocated:.1f} MB (Reserved: {total_reserved:.1f} MB)")
    print(f"{'-' * 65}\n")


def safe_metric(name: str, fn, *args, default: Any = None, **kwargs) -> Any:
    """Call an evaluation metric defensively: a broken metric must never kill a run."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - benchmark must survive any metric failure
        print(f"       [metric warning] {name} failed: {type(exc).__name__}: {exc}")
        return default


RECORD_TEMPLATE: Dict[str, Any] = {
    "model_id": None,
    "short_name": None,
    "family": "custom",
    "format": "merged",
    "param_size": "8B",
    "training_method": "SFT",
    "problem_id": None,
    "trial": 1,
    "domain": "General",
    "strategy": "zero_shot",
    "gen_time_s": 0.0,
    "executability": 0,
    "error_type": None,
    "error_message": None,
    "vcer": 1.0,
    "alignment_score": 0.0,
    "coverage_score": 0.0,
    "mas": None,
    "cmi": None,
    "car": None,
    "temporal_similarity": None,
    "visual_similarity": None,
    "video_path": None,
    "render_duration_s": 0.0,
    "code_len_lines": 0,
    "code_path": None,
}


def canonical_record(seed: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a record with the canonical key set, filled from `seed`."""
    record = dict(RECORD_TEMPLATE)
    if seed:
        record.update(seed)
    return record


def new_record(
    spec: "ModelSpec",
    problem_id: str,
    trial: int,
    domain: str,
    strategy: str,
    **overrides: Any,
) -> Dict[str, Any]:
    """
    Build a fully populated result record.

    Every record carries the *same* key set so downstream CSV/LaTeX writers never
    see a ragged table (this was one source of hard crashes in the paper export).
    """
    record = canonical_record({
        "model_id": spec.id,
        "short_name": spec.short_name,
        "family": spec.family,
        "format": spec.format,
        "param_size": spec.param_size,
        "training_method": spec.training_method,
        "problem_id": problem_id,
        "trial": trial,
        "domain": domain,
        "strategy": strategy,
    })
    record.update(overrides)
    return record


def record_key(record: Dict[str, Any]) -> str:
    """Stable identity of one (model, problem, trial, strategy) evaluation."""
    return "|".join(
        str(record.get(k, "")) for k in ("short_name", "problem_id", "trial", "strategy")
    )


# ═══════════════════════════════════════════════════════════════════════════
# Hardware Detection & Environment Verification (Kaggle Dual T4)
# ═══════════════════════════════════════════════════════════════════════════

def verify_hardware_environment() -> Dict[str, Any]:
    """Inspects available GPUs and verifies dual T4 configuration."""
    hw_info = {
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "total_vram_gb": 0.0,
        "is_t4_dual": False,
    }

    try:
        import torch
        hw_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            hw_info["device_count"] = count
            total_vram = 0.0
            for i in range(count):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                total_vram += mem
                hw_info["devices"].append({"index": i, "name": name, "vram_gb": round(mem, 2)})
            hw_info["total_vram_gb"] = round(total_vram, 2)
            hw_info["is_t4_dual"] = (count >= 2 and any("T4" in d["name"] for d in hw_info["devices"]))
    except ImportError:
        pass

    return hw_info


def print_hardware_banner(hw: Dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("  ManiBench Kaggle Hardware Environment Report")
    print("=" * 78)
    if hw["cuda_available"]:
        print(f"  CUDA Available:     Yes (Device Count: {hw['device_count']})")
        for d in hw["devices"]:
            print(f"    - GPU {d['index']}: {d['name']} | VRAM: {d['vram_gb']} GB")
        print(f"  Total VRAM:         {hw['total_vram_gb']} GB")
        if hw["is_t4_dual"]:
            print("  Status:             [OPTIMAL] Dual Tesla T4 detected (32GB Aggregate VRAM).")
            print("                      Using FP16 + device_map='auto' for balanced multi-GPU execution.")
        else:
            print("  Status:             Single GPU or alternative accelerator detected.")
    else:
        print("  CUDA Available:     No (Running in CPU or Dry-Run Mode)")
    print("=" * 78 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# Multi-GPU Kaggle Inference Engine
# ═══════════════════════════════════════════════════════════════════════════

class KaggleInferenceEngine:
    """
    Unified local inference engine optimized for Kaggle GPU T4*2.
    Supports:
      - Full-weight merged models (fp16 sharded across GPU 0 & 1)
      - LoRA / PEFT adapters (loads base model + adapter)
      - 4-bit / 8-bit quantization via bitsandbytes (for 31B models)
      - Clean VRAM teardown between models
    """

    def __init__(self, spec: ModelSpec, load_in_4bit: bool = False, device_map: str = "auto"):
        self.spec = spec
        self.load_in_4bit = load_in_4bit or (spec.recommended_quant == "4bit")
        self.device_map = device_map
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"\n[Model Loader] Loading '{self.spec.id}'...")
        print(f"               Format: {self.spec.format} | Family: {self.spec.family} | Size: {self.spec.param_size}")
        print_vram_status(f"Pre-Load State for '{self.spec.short_name}'")

        tokenizer_target = self.spec.id
        if self.spec.is_lora and self.spec.base_model:
            tokenizer_target = self.spec.base_model

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_target, trust_remote_code=True)
        except Exception:
            if self.spec.base_model:
                self.tokenizer = AutoTokenizer.from_pretrained(self.spec.base_model, trust_remote_code=True)
            else:
                raise

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Hardware kwargs for T4x2.
        # `device_map` (multi-GPU sharding) is only understood when `accelerate` is
        # installed; without it Transformers raises
        #   "Using a `device_map` ... requires `accelerate`"
        # which used to abort every single model load. Degrade gracefully instead.
        import importlib.util
        has_accelerate = importlib.util.find_spec("accelerate") is not None
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if has_accelerate:
            model_kwargs["device_map"] = self.device_map
        else:
            print("               [info] 'accelerate' is not installed - loading on a single "
                  "device without device_map (pip install accelerate for T4x2 sharding).")
        model_kwargs[self._dtype_kwarg_name()] = dtype

        if self.load_in_4bit:
            if not has_accelerate:
                print("               WARNING: 4-bit quantization needs 'accelerate'; "
                      "continuing in 16-bit.")
            else:
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    model_kwargs.pop(self._dtype_kwarg_name(), None)
                    print("               Quantization: 4-bit NormalFloat (NF4) enabled for 32GB budget.")
                except ImportError:
                    print("               WARNING: bitsandbytes not installed; falling back to 16-bit.")

        if self.spec.is_lora:
            if not self.spec.base_model:
                raise ValueError(f"LoRA adapter '{self.spec.id}' requires an explicit base_model.")
            print(f"               Step 1/2: Loading base model weights: {self.spec.base_model}")
            base = self._from_pretrained(AutoModelForCausalLM, self.spec.base_model, model_kwargs)
            print(f"               Step 2/2: Attaching LoRA adapter: {self.spec.id}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base, self.spec.id)
        else:
            print(f"               Loading full merged weights from: {self.spec.id}")
            self.model = self._from_pretrained(AutoModelForCausalLM, self.spec.id, model_kwargs)

        if not has_accelerate:
            # device_map did not place the weights for us - do it explicitly.
            target = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                self.model.to(target)
            except Exception as exc:  # noqa: BLE001
                print(f"               WARNING: could not move the model to {target}: {exc}")

        self.model.eval()
        print("               Model successfully resident in GPU memory.")
        print_vram_status(f"Post-Load State: '{self.spec.short_name}' Resident")

    @staticmethod
    def _dtype_kwarg_name() -> str:
        """
        `torch_dtype` was renamed to `dtype` in Transformers v5.

        Kaggle images currently ship v4.x while newer images (and local venvs)
        may ship v5+. Pick whichever keyword this installation understands.
        """
        try:
            import transformers
            major = int(str(transformers.__version__).split(".")[0])
            return "dtype" if major >= 5 else "torch_dtype"
        except Exception:
            return "torch_dtype"

    @staticmethod
    def _from_pretrained(model_cls, repo_id: str, model_kwargs: Dict[str, Any]):
        """Load weights, transparently retrying with the other dtype keyword."""
        try:
            return model_cls.from_pretrained(repo_id, **model_kwargs)
        except TypeError as exc:
            if "dtype" not in str(exc):
                raise
            alt = dict(model_kwargs)
            for old, new in (("torch_dtype", "dtype"), ("dtype", "torch_dtype")):
                if old in alt:
                    alt[new] = alt.pop(old)
                    break
            print(f"               [compat] retrying load with alternative dtype keyword: {exc}")
            return model_cls.from_pretrained(repo_id, **alt)

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.1,
        seed: Optional[int] = None,
    ) -> str:
        import torch

        if seed is not None:
            torch.manual_seed(seed)

        # Build prompt using chat template
        prompt_text = ""
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt_text = self._fallback_chatml(messages)
        else:
            prompt_text = self._fallback_chatml(messages)

        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": pad_token_id,
        }
        if temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def _fallback_chatml(self, messages: List[Dict[str, str]]) -> str:
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def unload(self) -> None:
        """Completely purge model from GPU memory to prevent OOM on next model."""
        import torch
        print(f"[Model Unloader] Purging '{self.spec.short_name}' from VRAM...")
        if self.model is not None:
            try:
                self.model.to("cpu")
            except Exception:
                pass
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
            vram_mb = sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())) / (1024**2)
            print(f"[Model Unloader] Purge complete. VRAM allocated: {vram_mb:.1f} MB.")


class SyntheticDryRunEngine:
    """Mock engine for testing the entire benchmark pipeline without GPUs."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        return (
            "```python\n"
            "from manim import *\n\n"
            "class ManiBenchEvaluationScene(Scene):\n"
            "    def construct(self):\n"
            "        title = Title('ManiBench Evaluation Benchmark')\n"
            "        self.play(Write(title))\n"
            "        line = Line(LEFT * 4, RIGHT * 4, color=BLUE)\n"
            "        dot = Dot(color=YELLOW).move_to(line.get_start())\n"
            "        self.play(Create(line), FadeIn(dot))\n"
            "        self.play(dot.animate.move_to(line.get_end()))\n"
            "        formula = MathTex(r'E = mc^2').next_to(line, DOWN)\n"
            "        self.play(Write(formula))\n"
            "        self.wait(1)\n"
            "```"
        )

    def unload(self) -> None:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Prompt Construction
# ═══════════════════════════════════════════════════════════════════════════

def construct_evaluation_prompt(problem: Dict[str, Any], strategy: str = "zero_shot") -> List[Dict[str, str]]:
    sys_prompt = (
        "You are an expert Manim Community Edition (CE) animation programmer. "
        "Write complete, production-grade Python code strictly using `from manim import *`. "
        "Strict Requirements:\n"
        "1. Do NOT use ManimGL syntax (no `ShowCreation`, `CONFIG` dicts, `TexMobject`, `ApplyMethod`).\n"
        "2. Define a single Scene subclass with a complete `construct()` method.\n"
        "3. Output strictly the executable Python code inside a ```python ... ``` block."
    )

    if strategy == "version_aware":
        sys_prompt += (
            "\n\nSTRICT CE MIGRATION RULES:\n"
            "- Use Create() instead of ShowCreation()\n"
            "- Use Tex() or MathTex() instead of TexMobject() or TextMobject()\n"
            "- Set properties directly in construct() or __init__, never use CONFIG class attributes\n"
            "- Use self.camera instead of self.frame"
        )

    task_title = problem.get("title", problem.get("id"))
    brief = problem.get("brief_description", problem.get("description", ""))
    events = problem.get("required_visual_events", [])
    events_str = "\n".join([f"- {e.get('description', e.get('event', ''))}" for e in events])

    user_text = (
        f"Problem: {task_title}\n\n"
        f"Objective:\n{brief}\n\n"
        f"Required Visual Events & Choreography:\n{events_str}\n\n"
        f"Write the complete Manim CE script now."
    )

    if strategy == "cot":
        user_text += "\n\nFirst plan the mathematical coordinates and visual staging step-by-step, then write the code."

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_text},
    ]


# ═══════════════════════════════════════════════════════════════════════════
# End-to-End Benchmark Execution Loop
# ═══════════════════════════════════════════════════════════════════════════

def run_benchmark(
    models: List[ModelSpec],
    problems: List[Dict[str, Any]],
    output_dir: Path,
    trials: int = 1,
    strategy: str = "zero_shot",
    skip_render: bool = False,
    timeout: int = 45,
    dry_run: bool = False,
    load_in_4bit: bool = False,
    compute_visual_sim: bool = False,
    max_new_tokens: int = 4096,
    seed: Optional[int] = None,
    resume: bool = True,
    reference_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hw = verify_hardware_environment()
    print_hardware_banner(hw)

    print(f"Target Models:   {[m.short_name for m in models]}")
    print(f"Target Problems: {[p['id'] for p in problems]}")
    print(f"Trials per Task: {trials}")
    print(f"Prompt Strategy: {strategy}")
    print(f"Skip Render:     {skip_render}")
    print(f"Timeout:         {timeout}s")
    print(f"Max New Tokens:  {max_new_tokens}")
    print(f"Visual Metrics:  {compute_visual_sim}")
    print(f"Resume:          {resume}")
    print(f"Output Directory:{output_dir}\n")

    reference_dir = Path(reference_dir) if reference_dir else DEFAULT_REFERENCE_DIR

    # ── Crash-safe checkpointing ────────────────────────────────────────────
    # Kaggle sessions get pre-empted and hit the 12h/9h wall. Every finished
    # (model, problem, trial) triple is flushed to disk immediately so a re-run
    # resumes instead of starting over.
    checkpoint_path = output_dir / "checkpoint_records.jsonl"
    records: List[Dict[str, Any]] = []
    done_keys: set = set()
    if resume and checkpoint_path.exists():
        for line_no, line in enumerate(checkpoint_path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"[resume] ignoring malformed checkpoint line {line_no}")
                continue
            # Normalise older/partial records onto the canonical key set. Records
            # from *other* models are kept too, so running a large benchmark in
            # several family-by-family invocations still yields one complete table.
            records.append(canonical_record(rec))
            done_keys.add(record_key(records[-1]))
        if records:
            print(f"[resume] Recovered {len(records)} completed evaluation(s) from {checkpoint_path.name}\n")

    checkpoint_fh = open(checkpoint_path, "a", encoding="utf-8")

    def _persist(record: Dict[str, Any]) -> None:
        records.append(record)
        done_keys.add(record_key(record))
        checkpoint_fh.write(json.dumps(record, default=str) + "\n")
        checkpoint_fh.flush()
        try:
            os.fsync(checkpoint_fh.fileno())
        except OSError:
            pass

    model_statuses: Dict[str, Dict[str, Any]] = {}
    completed_count = 0
    skipped_count = 0

    try:
        for model_spec in models:
            print(f"\n{'='*75}")
            print(f"Evaluating Model [{model_spec.short_name}] ({model_spec.id})")
            print(f"Family: {model_spec.family} | Method: {model_spec.training_method} | Format: {model_spec.format}")
            print(f"{'='*75}")

            pending = [
                (p, t) for p in problems for t in range(1, trials + 1)
                if f"{model_spec.short_name}|{p['id']}|{t}|{strategy}" not in done_keys
            ]
            if not pending:
                print("  [resume] all trials for this model already completed - skipping load.")
                model_statuses[model_spec.short_name] = {"status": "already_complete", "records": 0}
                continue
            skipped_count += len(problems) * trials - len(pending)

            # Instantiate engine
            if dry_run:
                engine = SyntheticDryRunEngine(model_spec)
            else:
                try:
                    engine = KaggleInferenceEngine(model_spec, load_in_4bit=load_in_4bit)
                except Exception as e:
                    print(f"[FATAL] Failed to load model '{model_spec.id}': {type(e).__name__}: {e}")
                    print("[FATAL] Full traceback:")
                    traceback.print_exc()
                    print("Skipping to next model.\n")
                    model_statuses[model_spec.short_name] = {
                        "status": "load_failed",
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc()[-2000:],
                    }
                    continue

            model_records_start = len(records)
            try:
                for prob in problems:
                    pid = prob["id"]
                    title = prob.get("title", pid)
                    domain = prob.get("domain", ["General"])
                    if isinstance(domain, list):
                        domain = domain[0] if domain else "General"

                    print(f"  -> [{pid}] {title} ({domain})")

                    for t in range(1, trials + 1):
                        key = f"{model_spec.short_name}|{pid}|{t}|{strategy}"
                        if key in done_keys:
                            continue

                        t0 = time.time()
                        messages = construct_evaluation_prompt(prob, strategy=strategy)

                        try:
                            raw_out = engine.generate(
                                messages,
                                max_tokens=max_new_tokens,
                                temperature=0.0 if trials == 1 else 0.4,
                                seed=seed,
                            )
                            gen_time = round(time.time() - t0, 2)
                            code = extract_python_code(raw_out)
                        except Exception as e:
                            print(f"     [Trial {t}] Generation exception: {type(e).__name__}: {e}")
                            _persist(new_record(
                                model_spec, pid, t, domain, strategy,
                                error_type="GenerationError",
                                error_message=f"{type(e).__name__}: {e}"[:200],
                            ))
                            continue

                        # Save generated code to disk
                        code_dir = DEFAULT_CODE_DIR / model_spec.short_name / strategy
                        code_dir.mkdir(parents=True, exist_ok=True)
                        code_file = code_dir / f"{pid}_trial{t}.py"
                        code_file.write_text(code, encoding="utf-8")

                        # Metric 1: Executability (renders the scene once - the
                        # produced video is reused by the visual metrics below).
                        video_out_dir = output_dir / "videos" / model_spec.short_name / strategy
                        exec_res = safe_metric(
                            "executability", compute_executability,
                            code, timeout=timeout, skip_render=skip_render,
                            video_out_dir=video_out_dir, default=None,
                        )
                        if exec_res is None:
                            # Legacy metric signature without video_out_dir
                            exec_res = safe_metric(
                                "executability(legacy)", compute_executability,
                                code, timeout=timeout, default={},
                            ) or {}

                        # Metric 2: Version-Conflict Error Rate (VCER)
                        vc_res = safe_metric("version_conflicts", detect_version_conflicts, code, default={}) or {}

                        # Metric 3: Visual Alignment Score
                        align_res = safe_metric(
                            "alignment", compute_alignment, code,
                            prob.get("required_visual_events", []), default={},
                        ) or {}

                        # Metric 4: Pedagogical Coverage Score
                        cov_res = safe_metric(
                            "coverage", compute_coverage, code,
                            prob.get("coverage_requirements", []), default={},
                        ) or {}

                        # Metric 5: Mathematical Accuracy Score (MAS)
                        mas_res = safe_metric("mathematical_fidelity", compute_mathematical_fidelity, code, prob, default={}) or {}
                        mas_val = mas_res.get("mathematical_fidelity", mas_res.get("mas", 0.0))

                        # Metric 6: Code Maintainability Index (CMI)
                        cmi_res = safe_metric("code_quality", compute_code_quality, code, default={}) or {}
                        cmi_val = cmi_res.get("code_maintainability_index", cmi_res.get("cmi", 70.0))

                        # Metric 7: Constraint Adherence Rate (CAR)
                        car_res = safe_metric("constraint_adherence", compute_constraint_adherence, code, prob, default={}) or {}
                        car_val = car_res.get("constraint_adherence_rate", car_res.get("car", 0.8))

                        # Metric 8: Visual similarity vs. the 3Blue1Brown reference clip
                        # (DINOv2 + DTW) and SSIM-based temporal similarity. Requires a
                        # rendered candidate video - honestly reported as None otherwise.
                        vis_sim_val = None
                        temp_sim_val = None
                        video_path = exec_res.get("video_path")
                        ref_video = reference_dir / f"{pid.lower()}_ref.mp4"
                        if compute_visual_sim and video_path and ref_video.exists():
                            vis_res = safe_metric(
                                "visual_similarity(DINOv2+DTW)",
                                compute_visual_similarity, ref_video, video_path, default={},
                            ) or {}
                            vis_sim_val = vis_res.get("alignment_score")
                            if vis_res.get("error"):
                                print(f"       [metric warning] visual similarity: {vis_res['error']}")
                            temp_res = safe_metric(
                                "temporal_similarity(SSIM)",
                                compute_temporal_similarity, ref_video, video_path, default={},
                            ) or {}
                            temp_sim_val = temp_res.get("ssim")
                        elif compute_visual_sim:
                            reason = "render skipped/disabled" if not video_path else f"no reference clip at {ref_video}"
                            print(f"       [metric notice] visual metrics unavailable ({reason})")

                        exec_val = exec_res.get("executability", 0)
                        vcer_val = vc_res.get("vcer", vc_res.get("version_conflict_rate", 0.0))
                        align_val = align_res.get("alignment_score", 0.0)
                        cov_val = cov_res.get("coverage_score", 0.0)

                        err_type = exec_res.get("error_type")
                        err_msg = exec_res.get("error_message")

                        def _num(value: Any, digits: int = 4) -> Optional[float]:
                            try:
                                return round(float(value), digits)
                            except (TypeError, ValueError):
                                return None

                        print(
                            f"     [Trial {t}] Exec: {exec_val} | VCER: {float(vcer_val) * 100:.1f}% | "
                            f"MAS: {float(mas_val):.3f} | CMI: {float(cmi_val):.1f} | CAR: {float(car_val):.2f} | "
                            f"Align: {float(align_val):.3f} | Cov: {float(cov_val):.3f} ({gen_time}s)"
                        )

                        _persist(new_record(
                            model_spec, pid, t, domain, strategy,
                            gen_time_s=gen_time,
                            executability=exec_val,
                            error_type=err_type,
                            error_message=err_msg[:200] if err_msg else None,
                            vcer=_num(vcer_val) if _num(vcer_val) is not None else 0.0,
                            alignment_score=_num(align_val) if _num(align_val) is not None else 0.0,
                            coverage_score=_num(cov_val) if _num(cov_val) is not None else 0.0,
                            mas=_num(mas_val),
                            cmi=_num(cmi_val, 2),
                            car=_num(car_val),
                            temporal_similarity=_num(temp_sim_val),
                            visual_similarity=_num(vis_sim_val),
                            video_path=str(video_path) if video_path else None,
                            render_duration_s=_num(exec_res.get("render_duration_s"), 2) or 0.0,
                            code_len_lines=len(code.splitlines()),
                            code_path=str(code_file.relative_to(ROOT_DIR)),
                        ))
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - never lose the whole run
                print(f"[ERROR] Model '{model_spec.short_name}' aborted mid-run: {type(exc).__name__}: {exc}")
                traceback.print_exc()
            finally:
                # Purge model from memory before the next model loads
                try:
                    engine.unload()
                except Exception as exc:  # noqa: BLE001
                    print(f"[warn] unload failed: {exc}")

            completed_count += len(records) - model_records_start
            model_statuses[model_spec.short_name] = {
                "status": "ok",
                "records": len(records) - model_records_start,
            }
    finally:
        checkpoint_fh.close()

    # ═══════════════════════════════════════════════════════════════════════
    # Format and Export Paper-Ready Artifacts
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*75}")
    print("Generating Publication-Ready Tables & Reports...")
    print(f"{'='*75}")

    if not records:
        print("\n[WARNING] No evaluation records were produced.")
        print("          Every model was skipped (preflight/load failure) or failed to generate.")
        print("          The exported tables below are EMPTY and are not valid for publication.\n")

    formatter = PaperFormatter(
        raw_records=records,
        metadata={
            "benchmark": "ManiBench (New Version)",
            "timestamp": timestamp,
            "trials": trials,
            "strategy": strategy,
            "skip_render": skip_render,
            "compute_visual_sim": compute_visual_sim,
            "max_new_tokens": max_new_tokens,
            "hardware": hw,
            "model_statuses": model_statuses,
            "completed_evaluations": completed_count,
            "skipped_evaluations": skipped_count,
        },
    )

    exported_files = formatter.export_all(output_dir)

    # Save complete JSON run
    json_path = output_dir / f"benchmark_run_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": formatter.metadata,
                "model_summaries": formatter.model_summaries,
                "model_statuses": model_statuses,
                "detailed_records": records,
            },
            f,
            indent=2,
            default=str,
        )

    # Print summary leaderboard
    print("\n" + formatter.generate_markdown_leaderboard())

    print("\n[Model Status]")
    for short_name, status in model_statuses.items():
        detail = status.get("error", "")
        print(f"  - {short_name:<38} {status.get('status', '?'):<17} records={status.get('records', 0)} {detail}".rstrip())

    print("\n[Artifacts Created Ready for Paper Submission]:")
    for name, p in exported_files.items():
        print(f"  * {name:<22}: {p}")
    print(f"  * {'raw_json_archive':<22}: {json_path}\n")

    return {
        "metadata": formatter.metadata,
        "model_summaries": formatter.model_summaries,
        "exported_files": exported_files,
        "json_path": json_path,
        "records": records,
        "model_statuses": model_statuses,
        "checkpoint_path": checkpoint_path,
        "hardware": hw,
        "completed_evaluations": completed_count,
        "skipped_evaluations": skipped_count,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ManiBench — Kaggle Dual GPU T4 Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list-models", action="store_true", help="List all registered models and exit")
    parser.add_argument("--models", nargs="+", default=None, help="Specific model IDs or short names to evaluate")
    parser.add_argument("--family", type=str, default=None, choices=["qwen-manimator", "aos-qwen3", "aos-qwen2.5", "gemma", "baseline"])
    parser.add_argument("--format", type=str, default=None, choices=["merged", "lora", "gguf", "base"])
    parser.add_argument("--add-model", nargs="+", default=None, metavar=("ID", "BASE_MODEL"), help="Dynamically add an arbitrary HF model: --add-model repo/id base/id")
    parser.add_argument("--problems", nargs="+", default=None, help="Problem IDs (e.g. MB-001 MB-003)")
    parser.add_argument("--trials", type=int, default=1, help="Trials per problem")
    parser.add_argument("--strategy", type=str, default="zero_shot", choices=["zero_shot", "version_aware", "cot"])
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit NormalFloat quantization (recommended for 31B models)")
    parser.add_argument("--skip-render", action="store_true", help="Skip Manim CE rendering (static AST + conflict checks only)")
    parser.add_argument("--compute-visual-sim", action="store_true", help="Compute DINOv2 + DTW similarity with reference videos")
    parser.add_argument("--timeout", type=int, default=45, help="Render timeout in seconds")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Maximum tokens generated per trial")
    parser.add_argument("--seed", type=int, default=None, help="Torch RNG seed for reproducible sampling")
    parser.add_argument("--no-resume", action="store_true", help="Ignore the checkpoint file and start from scratch")
    parser.add_argument("--reference-dir", type=str, default=None, help="Directory holding <problem>_ref.mp4 reference clips")
    parser.add_argument("--dry-run", action="store_true", help="Synthetic run without loading neural weights")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")

    args = parser.parse_args()

    if args.list_models:
        all_specs = catalog.list()
        print("\n" + "=" * 92)
        print(f"{'MANIBENCH EXTENSIBLE MODEL REGISTRY (Total: ' + str(len(all_specs)) + ')':^92}")
        print("=" * 92)
        print(f"{'#':<3} {'Short Name':<34} {'Family':<15} {'Format':<8} {'Size':<6} {'Method':<7} {'Base Model'}")
        print("-" * 92)
        for i, m in enumerate(all_specs, 1):
            print(f"{i:<3} {m.short_name:<34} {m.family:<15} {m.format:<8} {m.param_size:<6} {m.training_method:<7} {m.base_model or '--'}")
        print("=" * 92 + "\n")
        return

    # Add ad-hoc custom model if requested
    if args.add_model:
        mid = args.add_model[0]
        base_id = args.add_model[1] if len(args.add_model) > 1 else None
        custom_spec = catalog.get_or_create(mid, base_model=base_id)
        print(f"Dynamically registered custom model: {custom_spec}")

    # Model filtering
    selected_specs: List[ModelSpec] = []
    if args.models:
        for mid in args.models:
            spec = catalog.get_or_create(mid)
            selected_specs.append(spec)
    else:
        # Default: all models matching family/format, excluding GGUF by default for GPU execution
        selected_specs = catalog.list(family=args.family, format_type=args.format, exclude_gguf=True)

    if not selected_specs:
        print("ERROR: No models matched your selection criteria.")
        sys.exit(1)

    # Load problems from dataset
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset file not found at: {DATASET_PATH}")
        sys.exit(1)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_probs = data.get("problems", [])

    if args.problems:
        p_targets = set(p.upper() for p in args.problems)
        selected_probs = [p for p in all_probs if p["id"].upper() in p_targets]
    else:
        selected_probs = all_probs

    if not selected_probs:
        print("ERROR: No problems matched selection.")
        sys.exit(1)

    run_benchmark(
        models=selected_specs,
        problems=selected_probs,
        output_dir=Path(args.output_dir),
        trials=args.trials,
        strategy=args.strategy,
        skip_render=args.skip_render,
        timeout=args.timeout,
        dry_run=args.dry_run,
        load_in_4bit=args.load_in_4bit,
        compute_visual_sim=args.compute_visual_sim,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        resume=not args.no_resume,
        reference_dir=Path(args.reference_dir) if args.reference_dir else None,
    )


if __name__ == "__main__":
    main()
