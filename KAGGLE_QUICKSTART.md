# ManiBench on Kaggle — One-Command Quickstart

Everything the old notebook did cell-by-cell is now a **single resumable script**:
`scripts/kaggle_run.py`.

```bash
python scripts/kaggle_run.py --preset paper
```

---

## 1. Kaggle notebook setup (2 minutes)

> **Prerequisite:** the notebook clones the repository, so the revision containing
> `scripts/kaggle_run.py` must be pushed to GitHub first (`git push`). If you prefer not
> to push, either upload the updated notebook plus the `scripts/` + `evaluation/`
> folders as a Kaggle Dataset, or set `REPO_URL` in the setup cell to your own fork.

1. Create a notebook, then **Settings → Accelerator → GPU T4 × 2**, **Internet → On**.
2. Run this cell — it clones the repo and starts the full paper run:

```python
!git clone https://github.com/nabin2004/ManiBench.git
%cd ManiBench
!python scripts/kaggle_run.py --preset paper
```

That is the entire procedure. Or simply upload
`ManiBench_Kaggle_T4x2_Benchmark.ipynb` and press **Run All** — it is a thin launcher
with a configuration cell at the top.

Private/gated checkpoints (e.g. Gemma): add an HF token first.

```python
from kaggle_secrets import UserSecretsClient
import os
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

---

## 2. What the runner does

| Phase | Action |
| :--- | :--- |
| 1 | Detects Kaggle / Colab / local, prints the GPU and VRAM inventory |
| 2 | `apt` (ffmpeg, Cairo, Pango) + `pip` install with **NumPy/PyTorch pinned via a constraints file**, then verifies the stack and repairs/restarts if it is corrupt |
| 3 | Resolves the repository (shallow `git clone` when needed) |
| 4 | Loads `ManiBench_Pilot_Dataset.json` and the model registry |
| 5 | Hugging Face preflight — nonexistent or gated repos are skipped with a reason |
| 6 | Runs inference + the 8 metrics, flushing every trial to a checkpoint |
| 7 | Writes 300-DPI publication figures (PDF + PNG) |
| 8 | Builds `manibench_paper_submission_bundle.zip` and `run_manifest.json` |

Outputs land in `results/kaggle_paper_run/`:

```
results/kaggle_paper_run/
├── tables/           table1..table4.tex, leaderboard.md, results_summary.csv,
│                     results_per_trial.csv, paper_results_section.md
├── figures/          fig1_vcer_vs_executability.{pdf,png}, fig2_family_comparison.{pdf,png}
├── videos/           rendered candidate clips (when rendering is enabled)
├── checkpoint_records.jsonl   resume state (one JSON per finished trial)
├── benchmark_run_<ts>.json    complete archive
└── run_manifest.json          config + hardware + statuses + artifact index
```

The zip is also copied to the working directory (`/kaggle/working/...`), so it appears
directly in the notebook **Output** panel for download.

---

## 3. Presets and common flags

| Preset | Meaning |
| :--- | :--- |
| `paper` *(default)* | 8 headline models × 12 problems × 1 trial, rendering on |
| `trio` | base vs. SFT-merged vs. GRPO-adapter on Qwen3-8B |
| `full` | Every non-GGUF model in the registry (long; run it across several sessions) |
| `quick` | 1 baseline model × 2 problems, rendering off — environment sanity check |
| `smoke` | Synthetic code, no weights, no rendering — pipeline test on CPU |
| `smoke-render` | Synthetic code with real Manim rendering — validates Cairo/FFmpeg |

```bash
# only two tasks, three trials, 4-bit, keep every artifact
python scripts/kaggle_run.py --preset paper --problems MB-001 MB-005 --trials 3 \
    --load-in-4bit --seed 42

# base vs SFT-merged vs GRPO-adapter trio
python scripts/kaggle_run.py --preset trio --trials 3

# any Hugging Face model, including LoRA adapters
python scripts/kaggle_run.py --models your-org/your-manim-model

# the whole registry, family by family (each command resumes the same run)
python scripts/kaggle_run.py --family qwen-manimator
python scripts/kaggle_run.py --family aos-qwen3
python scripts/kaggle_run.py --family aos-qwen2.5
python scripts/kaggle_run.py --family baseline
```

Full option list: `python scripts/kaggle_run.py --help`.

### Publication bundle for the trio

For the extended package — methodology diagram, metric radar, per-task heatmaps,
failure taxonomy, significance forest plot, metric correlations, a narrative
`REPORT.md` and paste-ready LaTeX includes — use the dedicated trio script (it
reuses the same engine, checkpointing and metrics):

```bash
!python scripts/run_model_trio.py --trials 3
```

It writes `results/trio_publication/` with 9 figures (PDF + 300-DPI PNG), LaTeX
tables, CSVs, `REPORT.md`, `latex_includes.tex`, `ARTIFACTS.md`, a `run_manifest.json`
and a zip bundle. Figures alone can be rebuilt later on any machine (no GPU):

```bash
python scripts/run_model_trio.py --from-json results/trio_publication/benchmark_run_*.json
```

---

## 4. Resuming after a Kaggle timeout

Nothing special is required — run the same command again:

```bash
python scripts/kaggle_run.py --preset paper
```

The runner reads `checkpoint_records.jsonl`, skips every completed
`(model, problem, trial, strategy)` triple, and prints how many evaluations were
recovered. Use `--fresh` to deliberately discard previous progress.

---

## 5. The NumPy failure this runner fixes

The previous notebook died on every model with:

```
cannot import name '_center' from 'numpy._core.umath'
(/usr/local/lib/python3.12/dist-packages/numpy/_core/umath.py)
```

**Cause.** Kaggle ships a working NumPy. During the notebook's `pip install` step a
dependency resolver replaced or partially overwrote it, leaving *newer* pure-Python
modules (`numpy/_core/strings.py`, which imports `_center`) next to an *older* compiled
extension. The already-running Python process then cannot import the half-updated
package — every model load fails, and the run silently produces empty tables
("Total Model Evaluations: 0").

**Fix, in three layers.**

1. **Prevent** — before installing anything, the runner records the healthy versions of
   `numpy`, `torch`, `torchvision`, `torchaudio`, writes them to
   `manibench_constraints.txt`, and passes `-c manibench_constraints.txt` to every
   `pip install`. No resolver can move them anymore.
2. **Detect** — a subprocess probe imports the full stack and exercises the exact API
   that broke (`numpy._core.umath._center`, `numpy._core.strings`, `pandas`) before and
   after installation.
3. **Repair + restart** — if the probe fails, NumPy is force-reinstalled at the pinned
   version and the runner **re-executes itself** (`os.execv`) so the fixed libraries are
   imported by a pristine interpreter, without you restarting the Kaggle kernel.

Verify all of this offline at any time:

```bash
python scripts/selftest_kaggle_run.py
```

It even simulates the corrupted NumPy (via a `sitecustomize.py` shim that deletes
`_center`) and asserts the probe detects it.

---

## 6. Local dry run (no GPU, no weights, no network)

```bash
python scripts/kaggle_run.py --preset smoke --no-install
python scripts/kaggle_run.py --preset smoke-render --no-install   # needs FFmpeg
python scripts/kaggle_run.py --quick --no-install                 # real 8B model on GPU
```

---

## 7. Troubleshooting

| Symptom | Action |
| :--- | :--- |
| `CUDA is NOT available` | Accelerator is not set to **GPU T4 × 2** (Settings → Accelerator) |
| Model preflight skips everything | Internet is off, or an HF token is missing for a gated repo |
| `Rendering exceeded Ns` | Raise `--timeout 90`, or use `--skip-render` for static-only metrics |
| Out of VRAM on a 31B model | Add `--load-in-4bit` |
| Long run interrupted | Re-run the same command; it resumes from the checkpoint |
| Want a clean slate | `--fresh` (or delete `results/kaggle_paper_run/`) |
| Import errors after manual `pip install` in the notebook | Do not install manually — the runner handles it; otherwise restart the kernel |
