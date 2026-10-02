#!/usr/bin/env python3
"""
ManiBench — Hugging Face Manim Models Evaluation Script
======================================================
Automated evaluation pipeline for Manim-specialized LLMs from Hugging Face
(specifically nabin2004's models, including Qwen-Manimator, AOS-Qwen3, AOS-Qwen2.5,
and AOS-Gemma4 families).

Evaluates models against the ManiBench benchmark across 5 core metrics:
  1. Executability (Pass@1, Manim CE render success)
  2. Version-Conflict Error Rate (VCER - ManimGL/legacy API hallucination rate)
  3. Visual Alignment Score (Presence and sequencing of required visual events)
  4. Rubric Coverage Score (Mathematical & visual animation completeness)
  5. Visual Similarity (Optional DINOv2 + DTW similarity against reference videos)

Outputs structured results in:
  - JSON (detailed machine-readable data)
  - CSV (flattened trial-level & model-level metrics)
  - Markdown (formatted leaderboard and evaluation report)
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure stdout handles UTF-8 where possible without crashing on cp1252
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Load environment variables (.env) if python-dotenv is present
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass
DATASET_PATH = ROOT_DIR / "ManiBench_Pilot_Dataset.json"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "hf_evaluation"
DEFAULT_CODE_DIR = ROOT_DIR / "evaluation" / "generated_code"

# Curated catalog of Manim-related models by nabin2004 from extensible registry
from scripts.model_catalog import catalog, ModelSpec

MANIM_MODELS_CATALOG = [m.to_dict() for m in catalog.list()]



# ---------------------------------------------------------------------------
# Evaluation Core Metrics (Executability, VCER, Alignment, Coverage)
# ---------------------------------------------------------------------------

try:
    from tasks.manibench_eval_core import (
        extract_python_code,
        compute_executability,
        detect_version_conflicts,
        compute_alignment,
        compute_coverage,
        compute_visual_similarity_if_available,
    )
except ImportError:
    # Fallback to evaluation.metrics
    from evaluation.metrics import (
        compute_executability,
        detect_version_conflicts,
        compute_alignment,
        compute_coverage,
        compute_visual_similarity,
    )
    def extract_python_code(text: str) -> str:
        m = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
        if m: return m.group(1).strip()
        m = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if m: return m.group(1).strip()
        return text.strip()


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("problems", [])


# ---------------------------------------------------------------------------
# Backend Clients
# ---------------------------------------------------------------------------

class BaseLLMClient:
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(BaseLLMClient):
    """Client for local servers: Ollama, vLLM, LM Studio, llama.cpp server, etc."""
    def __init__(self, base_url: str = "http://localhost:11434/v1", api_key: str = "not-needed", model_id: str = ""):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        self.client = OpenAI(base_url=base_url, api_key=api_key or "local")
        self.model_id = model_id

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 4096)
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


class HuggingFaceInferenceClient(BaseLLMClient):
    """Client for Hugging Face Inference API / Serverless Endpoint."""
    def __init__(self, model_id: str, hf_token: Optional[str] = None):
        import requests
        self.model_id = model_id
        self.token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        import requests
        # Convert chat messages to standard format
        prompt_parts = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        full_prompt = "\n".join(prompt_parts)

        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", 4096),
                "temperature": kwargs.get("temperature", 0.1),
                "return_full_text": False,
            }
        }
        res = requests.post(self.api_url, headers=self.headers, json=payload, timeout=120)
        if res.status_code != 200:
            raise RuntimeError(f"HF API Error ({res.status_code}): {res.text}")
        result = res.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "")
        return str(result)


class TransformersLocalClient(BaseLLMClient):
    """Local Hugging Face Transformers execution optimized for Kaggle GPU T4*2 with LoRA support."""
    def __init__(
        self,
        model_id: str,
        base_model: Optional[str] = None,
        is_lora: bool = False,
        device: str = "auto",
        load_in_4bit: bool = False,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("Please install transformers, torch, accelerate, and peft")
        
        print(f"Loading local model '{model_id}' via transformers...")
        tokenizer_target = base_model if (is_lora and base_model) else model_id
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_target, trust_remote_code=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": device,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                print("Notice: bitsandbytes not available, continuing with float16.")

        if is_lora:
            if not base_model:
                raise ValueError(f"LoRA adapter '{model_id}' requires an explicit base_model.")
            base = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base, model_id)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

        self.model.eval()

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        import torch
        prompt_text = ""
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            try:
                prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt_text = "\n".join([f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages]) + "\n<|im_start|>assistant\n"
        else:
            prompt_text = "\n".join([f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages]) + "\n<|im_start|>assistant\n"

        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_tokens", 4096),
                do_sample=kwargs.get("temperature", 0.0) > 0,
                temperature=kwargs.get("temperature", 0.1),
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def unload(self) -> None:
        import gc, torch
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



class DryRunClient(BaseLLMClient):
    """Dry-run synthetic client for testing benchmark orchestration and reporting."""
    def __init__(self, model_id: str):
        self.model_id = model_id

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # Returns a valid Manim CE template code
        return (
            "from manim import *\n\n"
            "class CollidingBlocksScene(Scene):\n"
            "    def construct(self):\n"
            "        title = Title('Colliding Blocks & Pi')\n"
            "        self.play(Write(title))\n"
            "        wall = Line(LEFT * 5 + DOWN * 2, LEFT * 5 + UP * 2)\n"
            "        floor = Line(LEFT * 6 + DOWN * 2, RIGHT * 6 + DOWN * 2)\n"
            "        b1 = Square(side_length=1.0, color=BLUE).next_to(wall, RIGHT, buff=1)\n"
            "        b2 = Square(side_length=2.0, color=RED).next_to(b1, RIGHT, buff=2)\n"
            "        self.play(Create(wall), Create(floor))\n"
            "        self.play(FadeIn(b1), FadeIn(b2))\n"
            "        counter = Integer(0).to_corner(UP + RIGHT)\n"
            "        self.play(Create(counter))\n"
            "        self.wait(1)\n"
        )


# ---------------------------------------------------------------------------
# Prompt Building
# ---------------------------------------------------------------------------

def build_prompt_messages(problem: Dict[str, Any], strategy: str = "zero_shot") -> List[Dict[str, str]]:
    sys_prompt = (
        "You are an expert Manim CE (Community Edition) animation programmer. "
        "Write clean, executable Python code strictly using `from manim import *`. "
        "Do NOT use ManimGL, ShowCreation, CONFIG dicts, or deprecated APIs. "
        "Output ONLY the Python code in a ```python ... ``` markdown code block."
    )
    task_desc = problem.get("description", "")
    req_events = problem.get("required_visual_events", [])
    events_text = "\n".join([f"- {ev.get('event', ev.get('id', ''))}" for ev in req_events])

    user_content = (
        f"Problem: {problem.get('title', problem.get('id', ''))}\n\n"
        f"Description:\n{task_desc}\n\n"
        f"Required Visual Elements & Animation Flow:\n{events_text}\n\n"
        f"Produce a complete, self-contained Manim CE Scene subclass with a construct() method."
    )

    if strategy == "version_aware":
        sys_prompt += (
            "\n\nSTRICT CE REQUIREMENTS:\n"
            "- Use Create() instead of ShowCreation()\n"
            "- Use Tex/MathTex instead of TexMobject\n"
            "- Do not use CONFIG class dictionaries"
        )
    elif strategy == "cot":
        user_content += "\n\nFirst briefly reason step-by-step about scene timing, geometry, and transitions, then write the code."

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------

def run_evaluation(
    selected_models: List[Dict[str, Any]],
    problems: List[Dict[str, Any]],
    backend: str = "openai",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "not-needed",
    trials: int = 1,
    strategy: str = "zero_shot",
    skip_render: bool = False,
    timeout: int = 45,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> Dict[str, Any]:
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*75}")
    print(f"ManiBench HF Model Evaluation Pipeline")
    print(f"{'='*75}")
    print(f"Backend:         {backend}")
    if backend == "openai":
        print(f"Base URL:        {base_url}")
    print(f"Models ({len(selected_models)}):    {[m['short_name'] for m in selected_models]}")
    print(f"Problems ({len(problems)}):  {[p['id'] for p in problems]}")
    print(f"Trials:          {trials}")
    print(f"Strategy:        {strategy}")
    print(f"Skip Render:     {skip_render}")
    print(f"Timeout:         {timeout}s")
    print(f"{'='*75}\n")

    eval_records: List[Dict[str, Any]] = []

    for model_meta in selected_models:
        model_id = model_meta["id"]
        short_name = model_meta["short_name"]
        print(f"\n>>> Evaluating Model: {short_name} ({model_id})")

        # Instantiate backend client
        if backend == "openai":
            client = OpenAICompatibleClient(base_url=base_url, api_key=api_key, model_id=model_id)
        elif backend == "hf-api":
            client = HuggingFaceInferenceClient(model_id=model_id)
        elif backend == "transformers":
            client = TransformersLocalClient(
                model_id=model_id,
                base_model=model_meta.get("base_model"),
                is_lora=(model_meta.get("format") in ("lora", "LoRA/Adapter")),
            )
        elif backend == "dry-run":
            client = DryRunClient(model_id=model_id)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        for problem in problems:
            pid = problem["id"]
            title = problem.get("title", pid)
            print(f"  -> Problem {pid}: {title}...")

            for trial_idx in range(1, trials + 1):
                t0 = time.time()
                messages = build_prompt_messages(problem, strategy=strategy)
                
                try:
                    raw_output = client.generate(messages, temperature=0.0 if trials == 1 else 0.4)
                    gen_time = round(time.time() - t0, 2)
                    code = extract_python_code(raw_output)
                except Exception as e:
                    print(f"     [Trial {trial_idx}] Generation failed: {e}")
                    eval_records.append({
                        "model_id": model_id,
                        "short_name": short_name,
                        "family": model_meta["family"],
                        "format": model_meta["format"],
                        "problem_id": pid,
                        "trial": trial_idx,
                        "strategy": strategy,
                        "gen_time_s": 0.0,
                        "executability": 0,
                        "error_type": "GenerationError",
                        "error_message": str(e)[:200],
                        "vcer": 1.0,
                        "alignment_score": 0.0,
                        "coverage_score": 0.0,
                        "code_len_lines": 0,
                    })
                    continue

                # Save generated code to disk
                code_save_dir = DEFAULT_CODE_DIR / short_name / strategy
                code_save_dir.mkdir(parents=True, exist_ok=True)
                code_file = code_save_dir / f"{pid}_trial{trial_idx}.py"
                code_file.write_text(code, encoding="utf-8")

                # Metric 1: Executability
                try:
                    exec_res = compute_executability(code, timeout=timeout, skip_render=skip_render)
                except TypeError:
                    # In case compute_executability only takes code, timeout
                    exec_res = compute_executability(code, timeout=timeout)
                
                # Metric 2: Version-Conflict Error Rate (VCER)
                vcn = problem.get("version_conflict_notes", {})
                known_incompat = vcn.get("known_incompatibilities", []) if isinstance(vcn, dict) else []
                vc_res = detect_version_conflicts(code, known_incompat)

                # Metric 3: Alignment Score
                align_res = compute_alignment(code, problem.get("required_visual_events", []))

                # Metric 4: Coverage Score
                cov_res = compute_coverage(code, problem.get("coverage_requirements", []))

                # Extract metric values safely
                exec_val = exec_res.get("executability", 0)
                vcer_val = vc_res.get("vcer", vc_res.get("version_conflict_rate", 0.0))
                align_val = align_res.get("alignment_score", 0.0)
                cov_val = cov_res.get("coverage_score", 0.0)

                err_type = exec_res.get("error_type")
                err_msg = exec_res.get("error_message")

                print(f"     [Trial {trial_idx}] Exec: {exec_val} | VCER: {vcer_val:.2f} | Align: {align_val:.2f} | Cov: {cov_val:.2f} ({gen_time}s)")

                eval_records.append({
                    "model_id": model_id,
                    "short_name": short_name,
                    "family": model_meta["family"],
                    "format": model_meta["format"],
                    "problem_id": pid,
                    "trial": trial_idx,
                    "strategy": strategy,
                    "gen_time_s": gen_time,
                    "executability": exec_val,
                    "error_type": err_type,
                    "error_message": err_msg[:200] if err_msg else None,
                    "vcer": round(float(vcer_val), 4),
                    "alignment_score": round(float(align_val), 4),
                    "coverage_score": round(float(cov_val), 4),
                    "code_len_lines": len(code.splitlines()),
                    "code_path": str(code_file.relative_to(ROOT_DIR)),
                })

        if hasattr(client, "unload"):
            client.unload()

    # -----------------------------------------------------------------------
    # Aggregation & Structuring Results
    # -----------------------------------------------------------------------
    model_summaries = {}
    for m in selected_models:
        sname = m["short_name"]
        m_recs = [r for r in eval_records if r["short_name"] == sname]
        n_samples = len(m_recs)
        if n_samples == 0:
            continue
        
        exec_mean = sum(r["executability"] for r in m_recs) / n_samples
        vcer_mean = sum(r["vcer"] for r in m_recs) / n_samples
        align_mean = sum(r["alignment_score"] for r in m_recs) / n_samples
        cov_mean = sum(r["coverage_score"] for r in m_recs) / n_samples
        
        model_summaries[sname] = {
            "model_id": m["id"],
            "family": m["family"],
            "format": m["format"],
            "samples": n_samples,
            "executability_mean": round(exec_mean, 4),
            "pass_rate_pct": round(exec_mean * 100, 2),
            "vcer_mean": round(vcer_mean, 4),
            "alignment_mean": round(align_mean, 4),
            "coverage_mean": round(cov_mean, 4),
        }

    full_output = {
        "metadata": {
            "benchmark": "ManiBench (New Version)",
            "timestamp": timestamp,
            "backend": backend,
            "trials": trials,
            "strategy": strategy,
            "total_records": len(eval_records),
        },
        "model_summaries": model_summaries,
        "detailed_results": eval_records,
    }

    # 1. Save JSON
    json_path = results_dir / f"hf_eval_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    # 2. Save CSV
    csv_path = results_dir / f"hf_eval_{timestamp}.csv"
    if eval_records:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(eval_records[0].keys()))
            writer.writeheader()
            writer.writerows(eval_records)

    # 3. Save Markdown Report
    md_path = results_dir / f"hf_eval_{timestamp}.md"
    _generate_markdown_report(full_output, md_path)

    # 4. Export Paper-Ready LaTeX Tables & Summaries
    try:
        from scripts.paper_formatter import PaperFormatter
        formatter = PaperFormatter(raw_records=eval_records, metadata=full_output["metadata"])
        exported_tables = formatter.export_all(results_dir)
        print("Paper-Ready LaTeX & CSV tables successfully exported to:")
        for tname, tpath in exported_tables.items():
            print(f"  * {tname:<20}: {tpath}")
    except Exception as e:
        print(f"Notice: PaperFormatter export encountered: {e}")

    # Print Terminal Table
    _print_terminal_summary(model_summaries)

    print(f"\nStructured results successfully saved to:")
    print(f"  * JSON:     {json_path}")
    print(f"  * CSV:      {csv_path}")
    print(f"  * Markdown: {md_path}\n")

    return full_output


def _print_terminal_summary(model_summaries: Dict[str, Any]):
    print(f"\n{'='*82}")
    print(f"{'MANIBENCH EVALUATION SUMMARY LEADERBOARD':^82}")
    print(f"{'='*82}")
    print(f"{'Model':<34} {'Format':<10} {'Exec(%)':>8} {'VCER':>8} {'Align':>8} {'Cover':>8}")
    print(f"{'-'*34} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    # Sort by executability desc, then VCER asc
    sorted_models = sorted(
        model_summaries.items(),
        key=lambda x: (x[1]["executability_mean"], -x[1]["vcer_mean"], x[1]["alignment_mean"]),
        reverse=True,
    )
    for name, s in sorted_models:
        print(f"{name:<34} {s['format']:<10} {s['pass_rate_pct']:>7.1f}% {s['vcer_mean']:>8.3f} {s['alignment_mean']:>8.3f} {s['coverage_mean']:>8.3f}")
    print(f"{'='*82}\n")


def _generate_markdown_report(full_output: Dict[str, Any], path: Path):
    meta = full_output["metadata"]
    summaries = full_output["model_summaries"]

    lines = [
        f"# ManiBench Evaluation Report — Hugging Face Manim Models",
        f"",
        f"- **Benchmark:** {meta['benchmark']}",
        f"- **Evaluation Timestamp:** {meta['timestamp']}",
        f"- **Backend:** `{meta['backend']}`",
        f"- **Trials per Problem:** {meta['trials']}",
        f"- **Prompt Strategy:** `{meta['strategy']}`",
        f"",
        f"## Leaderboard & Performance Metrics",
        f"",
        f"| Model | Family | Format | Pass@1 (Exec) | VCER (Conflict Rate) | Alignment Score | Coverage Score |",
        f"| :--- | :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    sorted_models = sorted(
        summaries.items(),
        key=lambda x: (x[1]["executability_mean"], -x[1]["vcer_mean"], x[1]["alignment_mean"]),
        reverse=True,
    )

    for name, s in sorted_models:
        lines.append(
            f"| **`{name}`** | `{s['family']}` | `{s['format']}` | **{s['pass_rate_pct']:.1f}%** | `{s['vcer_mean']:.3f}` | `{s['alignment_mean']:.3f}` | `{s['coverage_mean']:.3f}` |"
        )

    lines.extend([
        f"",
        f"### Metric Explanations",
        f"1. **Pass@1 (Executability):** Fraction of runs where generated Manim code compiles, instantiates a valid `Scene.construct()`, imports CE, and renders without exceptions.",
        f"2. **VCER (Version-Conflict Error Rate):** Percentage of code lines hallucinating deprecated ManimGL / 3Blue1Brown-fork syntax (e.g. `ShowCreation`, `CONFIG` dicts, `TexMobject`). Lower is better.",
        f"3. **Visual Alignment Score:** Weighted detection of required mathematical animation events specified in the 3Blue1Brown video rubric.",
        f"4. **Coverage Score:** Metric spanning 4 critical dimensions: mathematical equations, visual mappings, numeric trackers, and structural pacing.",
        f"",
        f"---",
        f"*Generated automatically by ManiBench Evaluation Suite.*",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ManiBench Evaluation for Hugging Face Manim Models (nabin2004)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List all 28 Manim models in the catalog and exit",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Specific model ID(s) or short name(s) to evaluate (default: all or filtered by family/format)",
    )
    parser.add_argument(
        "--family", type=str, default=None,
        choices=["qwen-manimator", "aos-qwen3", "aos-qwen2.5", "gemma"],
        help="Filter models by family",
    )
    parser.add_argument(
        "--format", type=str, default=None,
        choices=["merged", "gguf", "lora"],
        help="Filter models by format",
    )
    parser.add_argument(
        "--backend", type=str, default="openai",
        choices=["openai", "transformers", "hf-api", "dry-run"],
        help="Backend inference engine (default: openai for Ollama/vLLM/LM Studio)",
    )
    parser.add_argument(
        "--base-url", type=str, default="http://localhost:11434/v1",
        help="Base URL for OpenAI-compatible server (e.g., http://localhost:11434/v1 for Ollama, http://localhost:8000/v1 for vLLM)",
    )
    parser.add_argument(
        "--api-key", type=str, default="not-needed",
        help="API Key for OpenAI-compatible or HF API backend",
    )
    parser.add_argument(
        "--problems", nargs="+", default=None,
        help="Problem IDs to evaluate (e.g. MB-001 MB-002, default: all 12 problems)",
    )
    parser.add_argument(
        "--trials", type=int, default=1,
        help="Number of evaluation trials per problem (default: 1)",
    )
    parser.add_argument(
        "--strategy", type=str, default="zero_shot",
        choices=["zero_shot", "few_shot", "cot", "constraint", "version_aware"],
        help="Prompt strategy (default: zero_shot)",
    )
    parser.add_argument(
        "--skip-render", action="store_true",
        help="Skip full Manim rendering (static syntax, AST, and conflict checking only)",
    )
    parser.add_argument(
        "--timeout", type=int, default=45,
        help="Manim execution render timeout in seconds (default: 45)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_RESULTS_DIR),
        help="Output directory for JSON, CSV, and Markdown results",
    )

    args = parser.parse_args()

    # If --list-models requested
    if args.list_models:
        print(f"\n{'='*88}")
        print(f"{'CATALOG OF NABIN2004 MANIM MODELS (28 Total)':^88}")
        print(f"{'='*88}")
        print(f"{'#':<3} {'Model ID':<46} {'Family':<16} {'Format':<8} {'Base Model'}")
        print(f"{'-'*3} {'-'*46} {'-'*16} {'-'*8} {'-'*15}")
        for idx, m in enumerate(MANIM_MODELS_CATALOG, 1):
            print(f"{idx:<3} {m['id']:<46} {m['family']:<16} {m['format']:<8} {m['base_model']}")
        print(f"{'='*88}\n")
        return

    # Filter catalog
    selected = list(MANIM_MODELS_CATALOG)
    if args.family:
        selected = [m for m in selected if m["family"] == args.family]
    if args.format:
        selected = [m for m in selected if m["format"] == args.format]
    if args.models:
        targets = set(args.models)
        selected = [m for m in selected if m["id"] in targets or m["short_name"] in targets]
        if not selected:
            # Allow user to specify custom model ID not yet in catalog
            for mid in args.models:
                selected.append({
                    "id": mid,
                    "short_name": mid.split("/")[-1],
                    "family": "custom",
                    "format": "custom",
                    "base_model": "Unknown",
                    "description": "User-specified custom model",
                    "recommended_backend": args.backend,
                })

    if not selected:
        print("ERROR: No models matched your filter criteria.")
        sys.exit(1)

    # Load problems
    all_problems = load_dataset(DATASET_PATH)
    if args.problems:
        prob_set = set(args.problems)
        problems = [p for p in all_problems if p["id"] in prob_set]
    else:
        problems = all_problems

    if not problems:
        print("ERROR: No problems found matching criteria.")
        sys.exit(1)

    run_evaluation(
        selected_models=selected,
        problems=problems,
        backend=args.backend,
        base_url=args.base_url,
        api_key=args.api_key,
        trials=args.trials,
        strategy=args.strategy,
        skip_render=args.skip_render,
        timeout=args.timeout,
        results_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
