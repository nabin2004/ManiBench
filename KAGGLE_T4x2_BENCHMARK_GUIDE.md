# ManiBench: Kaggle Dual GPU (T4×2) Benchmark Guide
===================================================

This guide details the complete end-to-end setup for evaluating **Manim-specialized Large Language Models** on **Kaggle GPU Dual Tesla T4 (2×16GB VRAM)** using the updated **ManiBench** benchmark suite, producing **publication-ready LaTeX tables, Markdown leaderboards, and CSV exports**.

> **TL;DR (updated flow).** The whole pipeline is now a single resumable command — see
> **[KAGGLE_QUICKSTART.md](KAGGLE_QUICKSTART.md)**. In a Kaggle notebook:
>
> ```python
> !git clone https://github.com/nabin2004/ManiBench.git
> %cd ManiBench
> !python scripts/kaggle_run.py --preset paper
> ```
>
> The rest of this document describes the architecture, model families, metrics and
> those same phases in detail.

---

## 1. Overview & Architecture

### Hardware Allocation Strategy on Kaggle (2× NVIDIA Tesla T4)
| Hardware Component | Specification | Allocation in ManiBench |
| :--- | :--- | :--- |
| **GPUs** | 2× NVIDIA Tesla T4 (Turing, CC 7.5) | Distributed via `device_map="auto"` across GPU 0 & 1 |
| **VRAM** | 16 GB per GPU (32 GB aggregate) | Native `torch.float16` precision (avoids BF16 emulation slowdowns) |
| **Model Size Budget** | 7B / 8B parameters | Loaded unquantized in FP16 (~14–16GB VRAM across 2 GPUs) |
| **Large Models** | 31B parameters (e.g. Gemma-4-31B) | Loaded with 4-bit NF4 (`--load-in-4bit`) taking ~18GB VRAM |
| **LoRA Adapters** | 13 adapter checkpoints | Base model loaded on dual GPUs + PEFT adapter attached |
| **VRAM Cleanup** | Sequential multi-model loop | Automatic garbage collection & `torch.cuda.empty_cache()` between models |

---

## 2. Supported Hugging Face Models (`nabin2004` & Baselines)

The benchmark includes a curated and extensible catalog of **30 models** (28 Manim-trained models + 2 baselines):

### 1. Qwen-Manimator Family (Qwen3-8B Base)
- `nabin2004/qwen-Manimator-1-merged` (Merged SFT weights)
- `nabin2004/qwen-Manimator-1-grpo-merged` (GRPO Reinforcement-Learned Merged)
- `nabin2004/qwen-Manimator-1-sft` (LoRA SFT Adapter on Qwen3-8B)
- `nabin2004/qwen-Manimator-1-grpo` (GRPO RL LoRA Adapter)
- `nabin2004/qwen-Manimator-1-grpo-clean` (Cleaned GRPO RL Adapter)
- `nabin2004/qwen-Manimator-1-gguf` (Quantized GGUF Q4_K_M)

### 2. AOS-Qwen3-8B Family (Narrated, SFT, DPO, GRPO)
- `nabin2004/AOS-qwen3-8b-narrated-sft-merged` (Merged with voiceover alignment)
- `nabin2004/AOS-qwen3-8b-narrated-merged` (DPO Preference-Tuned Merged)
- `nabin2004/AOS-qwen3-8b-grpo-merged` (GRPO RL Merged Model)
- `nabin2004/AOS-qwen3-8b-narrated-adapter` (LoRA Adapter with narration sync)
- `nabin2004/AOS-qwen3-8b-narrated-dpo` (DPO Preference LoRA Adapter)
- `nabin2004/AOS-qwen3-8b-grpo` (GRPO RL LoRA Adapter)
- `nabin2004/AOS-qwen3-8b-adapter` (Standard SFT LoRA Adapter)
- GGUF Quantizations: `AOS-qwen3-8b-narrated-sft-gguf`, `AOS-qwen3-8b-narrated-DPO-gguf`, `AOS-qwen3-8b-grpo-gguf`

### 3. AOS-Qwen2.5-Coder Family (7B Base)
- `nabin2004/AOS-qwen25-coder-7b-manim-merged`
- `nabin2004/AOS-qwen2.5-coder-7b-manim-merged`
- `nabin2004/qwen2.5-coder-7b-manim-merged`
- `nabin2004/AOS-qwen25-coder-7b-manim-sft` (LoRA SFT Adapter)
- `nabin2004/AOS-qwen2.5-coder-7b-manim-sft` (LoRA Checkpoint)
- GGUF Quantizations: `AOS-qwen25-coder-7b-manim-gguf`, `AOS-qwen2.5-coder-7b-manim-sft-GGUF`

### 4. AOS-Gemma4 Family (E2B and 31B)
- `nabin2004/AOS-gemma4-31b-manim-merged` (31B Parameter Merged — requires `--load-in-4bit`)
- `nabin2004/AOS-gemma4-31b-manim-sft` (31B LoRA Adapter — requires `--load-in-4bit`)
- `nabin2004/AOS-gemma4-manim-sft` (2B LoRA Adapter)
- `nabin2004/gemma-4-e2b-manim-lora` (Unsloth 4-bit LoRA Adapter)
- `nabin2004/AOS-gemma4-manim-gguf` (Lightweight GGUF)

### 5. Baselines for Paper Comparisons
- `Qwen/Qwen3-8B` (Pretrained Foundation Base)
- `Qwen/Qwen2.5-Coder-7B-Instruct` (Generalist Code LLM)

---

## 3. Quickstart: Running the Kaggle Notebook

1. **Create or Open Notebook on Kaggle**:
   - Go to [kaggle.com/code](https://www.kaggle.com/code) -> **New Notebook**.
   - Under **Notebook Options** in the right-side panel:
     - **Accelerator**: Select **GPU T4 x 2**.
     - **Internet**: Toggle **On** (required to download model weights from Hugging Face).
2. **Upload `ManiBench_Kaggle_T4x2_Benchmark.ipynb`**:
   - Click **File -> Upload Notebook** and select `ManiBench_Kaggle_T4x2_Benchmark.ipynb`.
3. **Execute Cells**:
   - Run the cells sequentially.
   - At Step 5, customize `SELECTED_MODELS` if you want to benchmark specific models or run the entire suite.
4. **Download Paper Bundle**:
   - The final cell generates `manibench_paper_submission_bundle.zip` containing all `.tex`, `.csv`, `.md`, and `.pdf` files. Download it directly from the notebook output panel.

---

## 4. Running via Terminal / CLI

You can also run the benchmark directly via command-line in Kaggle terminal or any Linux/Windows environment:

### List All Models in Registry
```bash
python scripts/run_kaggle_benchmark.py --list-models
```

### Benchmark Top Models across all 12 Pilot Tasks
```bash
python scripts/run_kaggle_benchmark.py \
    --models qwen-Manimator-1-grpo-merged AOS-qwen3-8b-narrated-sft-merged AOS-qwen25-coder-7b-manim-merged Qwen3-8B-Base \
    --trials 1 \
    --output-dir results/paper_run
```

### Benchmark an Entire Family (e.g. Qwen-Manimator)
```bash
python scripts/run_kaggle_benchmark.py \
    --family qwen-manimator \
    --trials 3 \
    --strategy zero_shot \
    --output-dir results/manimator_ablation
```

### Evaluate 31B Parameter Models with 4-bit Quantization
```bash
python scripts/run_kaggle_benchmark.py \
    --models nabin2004/AOS-gemma4-31b-manim-merged \
    --load-in-4bit \
    --output-dir results/gemma31b_run
```

### Fast Syntax & Conflict Check (Skip Video Render)
```bash
python scripts/run_kaggle_benchmark.py \
    --models qwen-Manimator-1-grpo-merged \
    --skip-render
```

### Dry-Run Test (Verifies complete pipeline in 3 seconds without GPU)
```bash
python scripts/run_kaggle_benchmark.py \
    --dry-run \
    --models qwen-Manimator-1-grpo-merged Qwen3-8B-Base \
    --problems MB-001 MB-002
```

---

## 5. Adding New Models (Zero-Config Extensibility)

The registry is completely extensible. You can benchmark any new Hugging Face model without modifying codebase files.

### Method 1: Dynamically via CLI
```bash
python scripts/run_kaggle_benchmark.py \
    --add-model your_username/your-manim-model Qwen/Qwen3-8B \
    --problems MB-001
```
*Note: If the model is a LoRA adapter, the system automatically inspects `adapter_config.json` on Hugging Face to resolve the base model if omitted.*

### Method 2: In Python Script or Notebook
```python
from scripts.model_catalog import catalog, ModelSpec

catalog.register(
    ModelSpec(
        id="your_username/your-new-model",
        short_name="your-new-model",
        family="qwen-manimator",
        format="merged", # or "lora"
        base_model="Qwen/Qwen3-8B",
        param_size="8B",
        training_method="GRPO",
        description="Newly fine-tuned animation model",
    )
)
```

---

## 6. The 8 Evaluation Metrics (Publication-Ready ManiBench)

Each generated Manim CE animation is evaluated against 8 comprehensive benchmark dimensions:

1. **Executability (Pass@1 / Pass@k)**:
   - Binary $1$ or $0$.
   - Must parse valid AST, instantiate a `Scene` subclass with `construct()`, use valid CE imports, and render cleanly under headless Manim CLI (`manim -ql -s --media_dir <tmp> <script> <scene>`).
2. **Version-Conflict Error Rate (VCER)**:
   - Evaluates the fraction of code lines triggering legacy ManimGL / 3Blue1Brown fork API hallucinations (145 tracked incompatibilities, including `ShowCreation`, `CONFIG` dictionaries, `TexMobject`, `ApplyMethod`, `self.frame`).
   - Formula: $\text{VCER} = \frac{\text{Conflicting Lines}}{\text{Total Code Lines}}$ (Lower is better).
3. **Mathematical Accuracy Score (MAS)**:
   - Symbolic AST parsing and equation equivalence testing via SymPy, verifying that mathematical formulas rendered in animations match ground-truth mathematical theorems.
4. **Code Maintainability Index (CMI)**:
   - Quantitative software engineering assessment combining cyclomatic complexity, Halstead volume, and PEP8 conformity via `radon` and `pycodestyle`.
5. **Constraint Adherence Rate (CAR)**:
   - Evaluates adherence to problem visual constraints (specified color palettes, frame runtimes, element counts, coordinate bounds).
6. **Visual Alignment Score**:
   - Event detection algorithm that tracks the sequential presence of problem-specific choreographic milestones from 3Blue1Brown reference videos (e.g. collision counter, coordinate phase plane, velocity vectors).
7. **Pedagogical Coverage Score**:
   - Measures element density across 4 weighted dimensions:
     - Math Annotations ($0.35$): LaTeX symbols, Tex, MathTex, equations.
     - Visual Mapping ($0.30$): Color encodings, vector fields, dot markers, dynamic updaters.
     - Numeric Evidence ($0.20$): DecimalNumber trackers, coordinate grids, function plots.
     - Structural Clarity ($0.15$): VGroup arrangements, scene pacing, transition sequencing.
8. **Temporal Visual Similarity (DINOv2 + DTW / SSIM)**:
   - Evaluates frame sequence progression and visual dynamics against ground truth reference clips.

---

## 7. Paper-Ready Publication Artifacts

Every benchmark run produces a dedicated `tables/` directory with publication-ready files:

| File | Format | Description / Usage in Research Paper |
| :--- | :--- | :--- |
| `table1_main_benchmark.tex` | LaTeX `booktabs` | **Table 1 of Paper**: Model comparison grouped by family, bolding best scores, showing Pass@1, VCER, MAS, CMI, CAR, Alignment, Coverage. |
| `table2_domain_breakdown.tex` | LaTeX `booktabs` | **Table 2 of Paper**: Breakdown across Calculus, Linear Algebra, Probability, Physics, and Geometry. |
| `table3_error_taxonomy.tex` | LaTeX `booktabs` | **Table 3 of Paper**: Distribution of failure modes (GL syntax, import errors, timeout, runtime). |
| `table4_statistical_significance.tex` | LaTeX `booktabs` | **Table 4 of Paper**: Pairwise Wilcoxon signed-rank and paired t-tests vs. foundation baseline with 95% CIs and p-values ($^*p < 0.05, ^{**}p < 0.01$). |
| `leaderboard.md` | Markdown | Formatted GitHub leaderboard for repository README. |
| `results_summary.csv` | CSV | Model-level aggregated metric table for statistical testing. |
| `results_per_trial.csv` | CSV | Trial-level granular results for pandas/R dataframes. |
| `paper_results_section.md` | Markdown/Text | Draft prose paragraphs for the experimental section of the paper. |
| `figures/fig1_vcer_vs_executability.pdf` | Vector PDF (300 DPI) | Camera-ready scatter plot with Pareto frontier. |
| `figures/fig2_family_comparison.pdf` | Vector PDF (300 DPI) | Clustered bar chart of metric scores by model family. |

### Including in LaTeX Manuscripts
In your Overleaf or LaTeX project:
```latex
\usepackage{booktabs}
\usepackage{amsmath}

% Include Table 1 (Main Benchmark Results)
\input{tables/table1_main_benchmark.tex}

% Include Table 2 (Domain Breakdown)
\input{tables/table2_domain_breakdown.tex}

% Include Table 4 (Statistical Significance)
\input{tables/table4_statistical_significance.tex}
```
