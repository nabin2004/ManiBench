#!/usr/bin/env python3
"""
Upload ManiBench Tasks to Kaggle Benchmarks
==========================================
Reads tasks/tasks_manifest.json and uploads each task to Kaggle Benchmarks
via the `kaggle benchmarks tasks push` CLI command.

Usage:
  # Push all tasks (async upload without waiting for model run):
  uv run python tasks/upload_tasks.py

  # Push all tasks and wait for each benchmark verification:
  uv run python tasks/upload_tasks.py --wait

  # Push a specific task:
  uv run python tasks/upload_tasks.py --slug manibench-mb-002-gradient-descent-how-neural-networks-learn
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows to avoid charmap encoding errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT_DIR / "tasks" / "tasks_manifest.json"


def main():
    parser = argparse.ArgumentParser(description="Upload ManiBench tasks to Kaggle Benchmarks.")
    parser.add_argument("--wait", action="store_true", help="Wait for each task execution to complete on Kaggle.")
    parser.add_argument("--slug", type=str, default=None, help="Specific task slug to upload.")
    parser.add_argument("--dataset", type=str, default=None, help="Optional Kaggle dataset to attach (owner/dataset-slug).")
    parser.add_argument("--skip-completed", action="store_true", help="Skip tasks that are already marked Completed on Kaggle.")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found at {MANIFEST_PATH}. Run 'uv run python tasks/generate_tasks.py' first.")
        sys.exit(1)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    if args.slug:
        tasks = [t for t in tasks if t["slug"] == args.slug]
        if not tasks:
            print(f"No task found with slug: {args.slug}")
            sys.exit(1)

    # Ensure UTF-8 encoding in environment for CLI output
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    completed_slugs = set()
    if args.skip_completed:
        try:
            r = subprocess.run(
                [sys.executable, "-m", "kaggle", "benchmarks", "tasks", "list"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", env=env
            )
            for line in r.stdout.splitlines():
                if "Completed" in line:
                    slug_candidate = line.split()[0].strip()
                    completed_slugs.add(slug_candidate)
        except Exception as e:
            print(f"Warning: Could not fetch existing task list: {e}")

    print(f"Found {len(tasks)} task(s) in manifest (Completed on Kaggle: {len(completed_slugs)}).\n")

    successes = []
    failures = []

    for idx, item in enumerate(tasks, start=1):
        slug = item["slug"]
        file_path = ROOT_DIR / item["file"]

        if not file_path.exists():
            print(f"[{idx}/{len(tasks)}] ❌ File missing: {file_path}")
            failures.append((slug, "File missing"))
            continue

        if args.skip_completed and slug in completed_slugs:
            print(f"[{idx}/{len(tasks)}] ⏭️  Skipping {slug} (already Completed on Kaggle)")
            successes.append(slug)
            continue

        cmd = [
            sys.executable, "-m", "kaggle", "benchmarks", "tasks", "push",
            slug,
            "-f", str(file_path),
        ]
        if args.dataset:
            cmd.extend(["-d", args.dataset])
        if args.wait:
            cmd.append("--wait")

        print(f"[{idx}/{len(tasks)}] Pushing {slug} ({item['file']})...")
        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(ROOT_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            output_lines = []
            for line in process.stdout:
                sys.stdout.write(f"     {line}")
                sys.stdout.flush()
                output_lines.append(line)
            process.wait()

            if process.returncode == 0:
                print(f"  ✅ Successfully uploaded {slug}")
                successes.append(slug)
            else:
                print(f"  ❌ Failed to upload {slug} (code {process.returncode})")
                failures.append((slug, "".join(output_lines[-5:])))
        except Exception as e:
            print(f"  ❌ Error executing push: {e}")
            failures.append((slug, str(e)))

        print("-" * 60)

    print(f"\nUpload Summary:")
    print(f"  Uploaded successfully: {len(successes)} / {len(tasks)}")
    if failures:
        print(f"  Failed: {len(failures)}")
        for slug, err in failures:
            print(f"    - {slug}: {err}")


if __name__ == "__main__":
    main()
