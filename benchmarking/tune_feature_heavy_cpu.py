"""Choose CPU thread counts in fresh workers using the exact wide fixtures."""

import json
import os
import subprocess
import sys
from pathlib import Path

from benchmark_feature_heavy import ALGORITHMS, SHAPES, compare

ROOT = Path(__file__).resolve().parents[1]
output = ROOT / "audit/feature_heavy/cpu_thread_tuning_fresh.json"
rows = []
for shape in SHAPES:
    for name in ALGORITHMS:
        pair = []
        for threads in [1, 4, 8]:
            command = [
                sys.executable,
                str(ROOT / "benchmarking/benchmark_feature_heavy.py"),
                "--worker",
                "--algorithm",
                name,
                "--shape",
                shape,
                "--backend",
                "cpu",
                "--seconds",
                "0.4",
            ]
            env = dict(os.environ, NUMBA_NUM_THREADS=str(threads), PYTHONPATH=str(ROOT / "src"))
            completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=True)
            row = json.loads(completed.stdout)
            pair.append(row)
            rows.append(row)
            output.write_text(json.dumps(rows, indent=2))
            print(name, shape, threads, row["median_seconds"], flush=True)
        for row in pair[1:]:
            compare(pair[0], row)
        best = min(pair, key=lambda r: r["median_seconds"])
        print("selected", name, shape, best["numba_threads"], flush=True)
