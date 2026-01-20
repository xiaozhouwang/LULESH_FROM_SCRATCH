#!/usr/bin/env python3
import argparse
import csv
import os
import re
import subprocess
import sys
from typing import Dict, List, Tuple


ELAPSED_RE = re.compile(r"Elapsed time\s*=\s*([0-9.]+)")
FOM_RE = re.compile(r"FOM\s*=\s*([0-9.eE+-]+)")


def parse_sizes(value: str) -> List[int]:
    sizes = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        sizes.append(int(token))
    return sizes


def clean_env(base: Dict[str, str]) -> Dict[str, str]:
    env = dict(base)
    for key in list(env.keys()):
        if key.startswith("LULESH_LOG_"):
            env.pop(key, None)
    return env


def run_command(cmd: List[str], env: Dict[str, str], cwd: str) -> str:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stdout}")
    return proc.stdout


def parse_metrics(output: str) -> Tuple[float, float]:
    elapsed = None
    fom = None
    for line in output.splitlines():
        elapsed_match = ELAPSED_RE.search(line)
        if elapsed_match:
            elapsed = float(elapsed_match.group(1))
        fom_match = FOM_RE.search(line)
        if fom_match:
            fom = float(fom_match.group(1))
    if elapsed is None or fom is None:
        raise RuntimeError("Failed to parse elapsed time/FOM from output.")
    return elapsed, fom


def main() -> int:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    parser = argparse.ArgumentParser(
        description="Benchmark CPU/GPU speedup across problem sizes."
    )
    parser.add_argument("--sizes", default="30,50,70,90,110")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--cpu-threads", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cpu-bin", default=os.path.join(root_dir, "lulesh2.0"))
    parser.add_argument(
        "--gpu-bin",
        default=os.path.join(root_dir, "lulesh-gpu-opt", "lulesh-cuda", "lulesh_gpu"),
    )
    parser.add_argument(
        "--out",
        default=os.path.join(root_dir, "benchmarks", "speedup.csv"),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Write a PNG plot if matplotlib is available.",
    )
    parser.add_argument(
        "--plot-path",
        default=os.path.join(root_dir, "benchmarks", "speedup.png"),
    )
    args = parser.parse_args()

    sizes = parse_sizes(args.sizes)
    if not sizes:
        print("No sizes specified.", file=sys.stderr)
        return 2

    cpu_env = clean_env(os.environ)
    gpu_env = clean_env(os.environ)
    cpu_env["OMP_NUM_THREADS"] = str(args.cpu_threads)

    rows = []
    for size in sizes:
        cpu_elapsed_vals = []
        cpu_fom_vals = []
        gpu_elapsed_vals = []
        gpu_fom_vals = []
        for _ in range(args.repeats):
            cpu_out = run_command(
                [args.cpu_bin, "-s", str(size), "-i", str(args.iterations)],
                env=cpu_env,
                cwd=root_dir,
            )
            cpu_elapsed, cpu_fom = parse_metrics(cpu_out)
            cpu_elapsed_vals.append(cpu_elapsed)
            cpu_fom_vals.append(cpu_fom)

            gpu_out = run_command(
                [args.gpu_bin, "-s", str(size), "-i", str(args.iterations)],
                env=gpu_env,
                cwd=root_dir,
            )
            gpu_elapsed, gpu_fom = parse_metrics(gpu_out)
            gpu_elapsed_vals.append(gpu_elapsed)
            gpu_fom_vals.append(gpu_fom)

        cpu_elapsed = sum(cpu_elapsed_vals) / len(cpu_elapsed_vals)
        cpu_fom = sum(cpu_fom_vals) / len(cpu_fom_vals)
        gpu_elapsed = sum(gpu_elapsed_vals) / len(gpu_elapsed_vals)
        gpu_fom = sum(gpu_fom_vals) / len(gpu_fom_vals)

        speedup_time = cpu_elapsed / gpu_elapsed
        speedup_fom = gpu_fom / cpu_fom if cpu_fom != 0 else 0.0

        rows.append(
            {
                "size": size,
                "cpu_elapsed_s": cpu_elapsed,
                "gpu_elapsed_s": gpu_elapsed,
                "cpu_fom": cpu_fom,
                "gpu_fom": gpu_fom,
                "speedup_time": speedup_time,
                "speedup_fom": speedup_fom,
            }
        )

    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote CSV: {args.out}")

    if args.plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"Plot skipped (matplotlib unavailable): {exc}")
            return 0

        sizes_x = [row["size"] for row in rows]
        speedup = [row["speedup_time"] for row in rows]

        plt.figure(figsize=(7, 4))
        plt.plot(sizes_x, speedup, marker="o", color="#1f77b4")
        plt.xlabel("Problem size (N)")
        plt.ylabel("Speedup (CPU time / GPU time)")
        plt.title("LULESH Speedup vs Size")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        plt.tight_layout()
        plt.savefig(args.plot_path, dpi=150)
        print(f"Wrote plot: {args.plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
