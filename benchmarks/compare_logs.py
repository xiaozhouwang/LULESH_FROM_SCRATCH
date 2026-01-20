#!/usr/bin/env python3
import argparse
import math
import os
import sys


def parse_list(value):
    items = []
    if not value:
        return items
    for part in value.split(","):
        token = part.strip()
        if token:
            items.append(token)
    return items


def default_tolerances(precision):
    if precision == "float":
        return 1e-5, 1e-4
    return 1e-12, 1e-9


def read_csv(path):
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


def compare_arrays(cpu_vals, gpu_vals, abs_tol, rel_tol):
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


def find_step_dirs(root, step_filter):
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


def main():
    parser = argparse.ArgumentParser(
        description="Compare LULESH CPU/GPU CSV logs with tolerances."
    )
    parser.add_argument("--cpu", required=True, help="CPU log root directory.")
    parser.add_argument("--gpu", required=True, help="GPU log root directory.")
    parser.add_argument("--precision", choices=["double", "float"], default="double")
    parser.add_argument("--abs-tol", type=float, default=None)
    parser.add_argument("--rel-tol", type=float, default=None)
    parser.add_argument("--steps", default="", help="Comma-separated step directories.")
    parser.add_argument("--fields", default="", help="Comma-separated CSV basenames.")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--quiet", action="store_true")
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

    missing = 0
    failures = 0
    total_files = 0

    for step in cpu_steps:
        cpu_matrix = os.path.join(args.cpu, step, "matrix")
        gpu_matrix = os.path.join(args.gpu, step, "matrix")

        if not os.path.isdir(cpu_matrix):
            print("Missing CPU matrix dir:", cpu_matrix, file=sys.stderr)
            missing += 1
            if not args.allow_missing:
                continue
        if not os.path.isdir(gpu_matrix):
            print("Missing GPU matrix dir:", gpu_matrix, file=sys.stderr)
            missing += 1
            if not args.allow_missing:
                continue

        if not os.path.isdir(cpu_matrix) or not os.path.isdir(gpu_matrix):
            continue

        cpu_files = [
            f for f in os.listdir(cpu_matrix) if f.endswith(".csv")
        ]
        for filename in sorted(cpu_files):
            base = filename[:-4]
            if field_filter and base not in field_filter:
                continue
            cpu_path = os.path.join(cpu_matrix, filename)
            gpu_path = os.path.join(gpu_matrix, filename)
            if not os.path.exists(gpu_path):
                print("Missing GPU file:", gpu_path, file=sys.stderr)
                missing += 1
                if not args.allow_missing:
                    continue
            if not os.path.exists(gpu_path):
                continue

            total_files += 1
            try:
                cpu_vals = read_csv(cpu_path)
                gpu_vals = read_csv(gpu_path)
            except (OSError, ValueError) as exc:
                print("Error reading CSV:", filename, exc, file=sys.stderr)
                failures += 1
                continue

            result = compare_arrays(cpu_vals, gpu_vals, abs_tol, rel_tol)
            if not args.quiet:
                print(
                    f"{step}/{filename}: count={result['count']} "
                    f"max_abs={result['max_abs']:.3e} "
                    f"max_rel={result['max_rel']:.3e} "
                    f"oob={result['oob']}"
                )
            if result["length_mismatch"] or result["oob"] > 0:
                failures += 1

    if not args.quiet:
        print(
            f"Compared {total_files} files "
            f"(missing={missing}, failures={failures})."
        )

    if failures > 0 or (missing > 0 and not args.allow_missing):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
