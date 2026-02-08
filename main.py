# main.py
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _run(cmd: list[str], cwd: Path, capture_run_dir: bool = False) -> Path | None:
    if not capture_run_dir:
        subprocess.run(cmd, cwd=str(cwd), check=True)
        return None

    # Stream stdout so you still see training logs, while also extracting run_dir.
    run_dir: Path | None = None
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert p.stdout is not None
    pat = re.compile(r"^\[ok\]\s+run_dir:\s+(.+)$")

    for line in p.stdout:
        print(line, end="")  # already includes newline
        m = pat.match(line.strip())
        if m:
            run_dir = Path(m.group(1)).expanduser()
            if not run_dir.is_absolute():
                run_dir = cwd / run_dir

    rc = p.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)

    if run_dir is None:
        raise RuntimeError("Could not parse run_dir from training output.")
    return run_dir


def main():
    ap = argparse.ArgumentParser(description="Run the full Cognito M2 pipeline end-to-end.")
    ap.add_argument("--config", type=str, default="experiments/config_m2.yaml", help="Path to YAML config.")
    ap.add_argument("--skip_evaluate", action="store_true", help="Skip experiments/20_evaluate.py.")
    args = ap.parse_args()

    root = _repo_root()
    os.chdir(root)  # makes relative paths behave in Spyder / IDEs

    py = sys.executable
    cfg = args.config

    # 0) extract
    _run([py, "experiments/00_extract_dataset.py", "--config", cfg], cwd=root)

    # 1) split
    _run([py, "experiments/01_make_split.py", "--config", cfg], cwd=root)

    # 2) features
    _run([py, "experiments/02_build_features.py", "--config", cfg], cwd=root)

    # 3) train (capture run_dir)
    run_dir = _run([py, "experiments/10_train_m2.py", "--config", cfg], cwd=root, capture_run_dir=True)
    assert run_dir is not None

    # 4) evaluate (plots, confusion matrices)
    if not args.skip_evaluate:
        _run([py, "experiments/20_evaluate.py", "--run_dir", str(run_dir)], cwd=root)

    print(f"[ok] pipeline complete. run_dir: {run_dir}")


if __name__ == "__main__":
    main()
