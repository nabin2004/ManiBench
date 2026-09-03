# ManiBench on Kaggle Benchmarks: Setup & Execution Guide

This guide explains how to configure, write, validate, push, execute, and publish **ManiBench** tasks on **Kaggle Benchmarks** using the `kaggle` CLI and the `kaggle-benchmarks` Python SDK (`kbench`).

---

## 1. Prerequisites & Installation

Ensure you have Python 3.10+ installed along with the Kaggle CLI and `kaggle-benchmarks` SDK:

```bash
pip install kaggle kaggle-benchmarks
```

Ensure your Kaggle API token (`kaggle.json`) is placed in `~/.kaggle/kaggle.json`.

---

## 2. Quick Command Reference

```bash
kaggle b (alias for kaggle benchmarks)
├── init              — Initialize credentials and local environment (.env)
├── auth              — Refresh Model Proxy credentials
└── tasks (alias: t)  — Manage benchmark tasks
    ├── push          — Upload a task from a Python file
    ├── run           — Run a task against specified LLM model(s)
    ├── list          — List your benchmark tasks
    ├── status        — Show task details and per-model run status
    ├── log           — Stream live or fetch execution logs
    ├── download      — Download completed run outputs and results
    └── publish       — Make a benchmark task public on Kaggle
```

---

## 3. Environment Setup & Initialization

Run the initialization command to fetch credentials and populate your `.env` file with `MODEL_PROXY_*` settings:

```bash
kaggle b init -y
```

> [!NOTE]
> `MODEL_PROXY_API_KEY` is short-lived. If local validation or server-side runs report authentication errors, refresh credentials using:
> ```bash
> kaggle b auth -y
> ```

---

## 4. Step-by-Step Task Workflow

### Step 1: Generate Task Files

Generate Jupytext-formatted (`# %%`) task files for all 12 ManiBench pilot problems from `ManiBench_Pilot_Dataset.json`:

```bash
python tasks/generate_tasks.py
```

This creates self-contained task scripts in `tasks/` (e.g. `tasks/mb_001_colliding_blocks_compute_pi.py`) and saves a manifest to `tasks/tasks_manifest.json`.

### Step 2: Validate Locally

Before pushing to Kaggle, validate any task script locally. Running the script directly executes `llm.prompt()` using local `.env` settings and outputs a `*.run.json` execution file:

```bash
python tasks/mb_001_colliding_blocks_compute_pi.py
```

Confirm that `*.run.json` was created cleanly:

```bash
ls -1 *.run.json
```

### Step 3: Push Task to Kaggle

Push the task file to Kaggle Benchmarks:

```bash
kaggle b t push manibench-mb-001-colliding-blocks-compute-pi -f tasks/mb_001_colliding_blocks_compute_pi.py --wait
```

#### Attaching Kaggle Datasets (Optional)

To attach the backing `ManiBench` dataset:

```bash
kaggle b t push manibench-mb-001-colliding-blocks-compute-pi -f tasks/mb_001_colliding_blocks_compute_pi.py -d owner/manibench-dataset --wait
```

### Step 4: Run Task against LLMs

Run the task on Kaggle's infrastructure against specific LLM models (e.g. `gemini-3.5-flash`, `claude-haiku-4-5`):

```bash
# Run against a specific model and wait for completion
kaggle b t run manibench-mb-001-colliding-blocks-compute-pi -m gemini-3.5-flash --wait

# Run against multiple models (repeat -m flag)
kaggle b t run manibench-mb-001-colliding-blocks-compute-pi -m gemini-3.5-flash -m claude-haiku-4-5 --wait
```

To view available models on Kaggle:

```bash
kaggle b t models
```

### Step 5: Check Task & Run Status

Check details, version info, state, and per-model run results:

```bash
kaggle b t status manibench-mb-001-colliding-blocks-compute-pi
```

### Step 6: View & Stream Logs

Stream live execution logs or view logs for completed runs:

```bash
kaggle b t log manibench-mb-001-colliding-blocks-compute-pi -m gemini-3.5-flash
```

### Step 7: Download Results

Download execution artifacts and results to a local folder:

```bash
kaggle b t download manibench-mb-001-colliding-blocks-compute-pi -o ./results
```

To include source notebooks for debugging:

```bash
kaggle b t download manibench-mb-001-colliding-blocks-compute-pi -o ./results -s -f
```

### Step 8: Publish Task

Publish the task and make it public on Kaggle:

```bash
kaggle b t publish manibench-mb-001-colliding-blocks-compute-pi
```

---

## 5. ManiBench Pilot Task Inventory

| Problem ID | Task Slug | Task File |
|------------|-----------|-----------|
| MB-001 | `manibench-mb-001-colliding-blocks-compute-pi` | `tasks/mb_001_colliding_blocks_compute_pi.py` |
| MB-002 | `manibench-mb-002-gradient-descent-how-neural-networks-learn` | `tasks/mb_002_gradient_descent_how_neural_networks_learn.py` |
| MB-003 | `manibench-mb-003-but-what-is-a-convolution` | `tasks/mb_003_but_what_is_a_convolution.py` |
| MB-004 | `manibench-mb-004-eigenvectors-eigenvalues-chapter-14` | `tasks/mb_004_eigenvectors_eigenvalues_chapter_14.py` |
| MB-005 | `manibench-mb-005-the-determinant-chapter-6` | `tasks/mb_005_the_determinant_chapter_6.py` |
| MB-006 | `manibench-mb-006-but-what-is-the-central-limit-theorem` | `tasks/mb_006_but_what_is_the_central_limit_theorem.py` |
| MB-007 | `manibench-mb-007-the-medical-test-paradox` | `tasks/mb_007_the_medical_test_paradox.py` |
| MB-008 | `manibench-mb-008-visualizing-the-chain-rule` | `tasks/mb_008_visualizing_the_chain_rule.py` |
| MB-009 | `manibench-mb-009-integration-and-the-fundamental-theorem-of-calculus` | `tasks/mb_009_integration_and_the_fundamental_theorem_of_calculus.py` |
| MB-010 | `manibench-mb-010-taylor-series` | `tasks/mb_010_taylor_series.py` |
| MB-011 | `manibench-mb-011-the-hairy-ball-theorem` | `tasks/mb_011_the_hairy_ball_theorem.py` |
| MB-012 | `manibench-mb-012-the-unexpectedly-hard-windmill-problem` | `tasks/mb_012_the_unexpectedly_hard_windmill_problem.py` |

---

## 6. Important Rules & Gotchas

1. **Must call `.run()`**: Every task function in Python must end with `task_fn.run(kbench.llm)` to ensure outputs are recorded.
2. **Repeated flags**: For flags accepting multiple values (like `-m` for models or `-d` for datasets), pass the flag multiple times (e.g., `-m model-a -m model-b`), **not** space-separated.
3. **Dataset persistence**: When re-pushing a task, repeat all `-d` flags. Re-pushing without `-d` detaches previous datasets.
4. **Canonical Model Slugs**: Use canonical model slugs (`gemini-3.5-flash`, `claude-haiku-4-5`) without provider prefixes.
