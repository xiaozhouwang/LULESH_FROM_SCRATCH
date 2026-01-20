#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
import sys
from typing import Dict, List, Tuple


ELAPSED_RE = re.compile(r"_cycle(\d+)")


def parse_list(value: str) -> List[str]:
    items = []
    if not value:
        return items
    for part in value.split(","):
        token = part.strip()
        if token:
            items.append(token)
    return items


def default_tolerances(precision: str) -> Tuple[float, float]:
    if precision == "float":
        return 1e-5, 1e-4
    return 1e-12, 1e-9


def read_csv(path: str) -> List[float]:
    values = []
    with open(path, "r") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            for part in raw.split(","):
                token = part.strip()
                if token:
                    values.append(float(token))
    return values


def compare_arrays(
    cpu_vals: List[float],
    gpu_vals: List[float],
    abs_tol: float,
    rel_tol: float,
) -> Dict[str, float]:
    if len(cpu_vals) != len(gpu_vals):
        return {
            "count": min(len(cpu_vals), len(gpu_vals)),
            "max_abs": float("inf"),
            "max_rel": float("inf"),
            "oob": max(len(cpu_vals), len(gpu_vals)),
            "length_mismatch": True,
        }

    max_abs = 0.0
    max_rel = 0.0
    oob = 0
    tiny = 1e-30

    for a, b in zip(cpu_vals, gpu_vals):
        if not math.isfinite(a) or not math.isfinite(b):
            oob += 1
            max_abs = float("inf")
            max_rel = float("inf")
            continue
        diff = abs(a - b)
        rel = diff / max(abs(a), abs(b), tiny)
        if diff > max_abs:
            max_abs = diff
        if rel > max_rel:
            max_rel = rel
        if diff > abs_tol and rel > rel_tol:
            oob += 1

    return {
        "count": len(cpu_vals),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "oob": oob,
        "length_mismatch": False,
    }


def find_step_dirs(root: str, step_filter: List[str]) -> List[str]:
    if not os.path.isdir(root):
        return []
    dirs = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if not name.startswith("step"):
            continue
        if step_filter and name not in step_filter:
            continue
        dirs.append(name)
    return sorted(dirs)


def parse_cycle(step_name: str) -> int:
    match = ELAPSED_RE.search(step_name)
    if match:
        return int(match.group(1))
    return 0


def main() -> int:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(
        description="Summarize CPU/GPU log differences and plot by cycle."
    )
    parser.add_argument(
        "--cpu",
        default=os.path.join(root_dir, "benchmarks", "logs-multi", "cycles10-s1"),
    )
    parser.add_argument(
        "--gpu",
        default=os.path.join(root_dir, "benchmarks", "logs-gpu-multi", "cycles10-s1"),
    )
    parser.add_argument("--precision", choices=["double", "float"], default="double")
    parser.add_argument("--abs-tol", type=float, default=None)
    parser.add_argument("--rel-tol", type=float, default=None)
    parser.add_argument("--steps", default="", help="Comma-separated step directories.")
    parser.add_argument("--fields", default="", help="Comma-separated CSV basenames.")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--out-csv",
        default=os.path.join(root_dir, "benchmarks", "correctness.csv"),
    )
    parser.add_argument(
        "--out-steps-csv",
        default=os.path.join(root_dir, "benchmarks", "correctness_steps.csv"),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Write a PNG plot if matplotlib is available.",
    )
    parser.add_argument(
        "--plot-path",
        default=os.path.join(root_dir, "benchmarks", "correctness.png"),
    )
    args = parser.parse_args()

    step_filter = parse_list(args.steps)
    field_filter = set(parse_list(args.fields))
    abs_tol, rel_tol = default_tolerances(args.precision)
    if args.abs_tol is not None:
        abs_tol = args.abs_tol
    if args.rel_tol is not None:
        rel_tol = args.rel_tol

    cpu_steps = find_step_dirs(args.cpu, step_filter)
    if not cpu_steps:
        print("No step directories found under CPU root.", file=sys.stderr)
        return 2

    cycle_summary: Dict[int, Dict[str, float]] = {}
    step_rows = []

    missing = 0
    failures = 0

    for step in cpu_steps:
        cycle = parse_cycle(step)
        cpu_matrix = os.path.join(args.cpu, step, "matrix")
        gpu_matrix = os.path.join(args.gpu, step, "matrix")

        if not os.path.isdir(cpu_matrix):
            missing += 1
            if not args.allow_missing:
                continue
        if not os.path.isdir(gpu_matrix):
            missing += 1
            if not args.allow_missing:
                continue
        if not os.path.isdir(cpu_matrix) or not os.path.isdir(gpu_matrix):
            continue

        cpu_files = [f for f in os.listdir(cpu_matrix) if f.endswith(".csv")]
        step_max_abs = 0.0
        step_max_rel = 0.0
        step_oob = 0
        step_files = 0

        for filename in sorted(cpu_files):
            base = filename[:-4]
            if field_filter and base not in field_filter:
                continue
            cpu_path = os.path.join(cpu_matrix, filename)
            gpu_path = os.path.join(gpu_matrix, filename)
            if not os.path.exists(gpu_path):
                missing += 1
                if not args.allow_missing:
                    continue
            if not os.path.exists(gpu_path):
                continue

            try:
                cpu_vals = read_csv(cpu_path)
                gpu_vals = read_csv(gpu_path)
            except (OSError, ValueError):
                failures += 1
                continue

            result = compare_arrays(cpu_vals, gpu_vals, abs_tol, rel_tol)
            if result["length_mismatch"] or result["oob"] > 0:
                failures += 1

            step_files += 1
            step_oob += int(result["oob"])
            if result["max_abs"] > step_max_abs:
                step_max_abs = result["max_abs"]
            if result["max_rel"] > step_max_rel:
                step_max_rel = result["max_rel"]

        if step_files == 0:
            continue

        step_rows.append(
            {
                "step": step,
                "cycle": cycle,
                "max_abs": step_max_abs,
                "max_rel": step_max_rel,
                "oob": step_oob,
                "files_compared": step_files,
            }
        )

        cycle_entry = cycle_summary.setdefault(
            cycle,
            {"max_abs": 0.0, "max_rel": 0.0, "oob": 0, "files_compared": 0},
        )
        if step_max_abs > cycle_entry["max_abs"]:
            cycle_entry["max_abs"] = step_max_abs
        if step_max_rel > cycle_entry["max_rel"]:
            cycle_entry["max_rel"] = step_max_rel
        cycle_entry["oob"] += step_oob
        cycle_entry["files_compared"] += step_files

    cycles = sorted(cycle_summary.keys())
    with open(args.out_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cycle", "max_abs", "max_rel", "oob", "files_compared"])
        for cycle in cycles:
            entry = cycle_summary[cycle]
            writer.writerow(
                [
                    cycle,
                    entry["max_abs"],
                    entry["max_rel"],
                    entry["oob"],
                    entry["files_compared"],
                ]
            )

    with open(args.out_steps_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "cycle", "max_abs", "max_rel", "oob", "files_compared"])
        for row in step_rows:
            writer.writerow(
                [
                    row["step"],
                    row["cycle"],
                    row["max_abs"],
                    row["max_rel"],
                    row["oob"],
                    row["files_compared"],
                ]
            )

    print(f"Wrote CSV: {args.out_csv}")
    print(f"Wrote step CSV: {args.out_steps_csv}")

    if args.plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"Plot skipped (matplotlib unavailable): {exc}")
            return 0

        if not cycles:
            print("No cycles to plot.", file=sys.stderr)
            return 0

        eps = 1e-30
        max_abs_vals = [max(cycle_summary[c]["max_abs"], eps) for c in cycles]
        max_rel_vals = [max(cycle_summary[c]["max_rel"], eps) for c in cycles]
        oob_vals = [cycle_summary[c]["oob"] for c in cycles]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        ax1.semilogy(cycles, max_abs_vals, marker="o", label="max_abs")
        ax1.semilogy(cycles, max_rel_vals, marker="o", label="max_rel")
        ax1.set_ylabel("Max diff (log)")
        ax1.set_title("CPU vs GPU Differences by Cycle")
        ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax1.legend()

        ax2.bar(cycles, oob_vals, color="#d62728", alpha=0.7)
        ax2.set_xlabel("Cycle")
        ax2.set_ylabel("Out-of-bounds count")
        ax2.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

        plt.tight_layout()
        plt.savefig(args.plot_path, dpi=150)
        print(f"Wrote plot: {args.plot_path}")

    if failures > 0 or (missing > 0 and not args.allow_missing):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
