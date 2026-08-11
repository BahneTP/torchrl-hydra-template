#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


FIELDNAMES = [
    "algorithm",
    "variant",
    "game",
    "seed",
    "exit_code",
    "final_return_mean",
    "final_return_std",
    "wandb_run_name",
    "log_file",
]

SUMMARY_FIELDNAMES = ["algorithm", "variant", "game", "mean", "std", "n"]

RETURN_MEAN_RE = re.compile(r"eval/return_mean=([-+0-9.eE]+)")
RETURN_STD_RE = re.compile(r"eval/return_std=([-+0-9.eE]+)")


def _read_final_returns(log_file: Path) -> tuple[str, str]:
    if not log_file.exists():
        return "", ""
    mean = ""
    std = ""
    for line in log_file.read_text(errors="replace").splitlines():
        mean_match = RETURN_MEAN_RE.search(line)
        std_match = RETURN_STD_RE.search(line)
        if mean_match:
            mean = mean_match.group(1)
        if std_match:
            std = std_match.group(1)
    return mean, std


def _ensure_header(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def init(args: argparse.Namespace) -> None:
    _ensure_header(Path(args.results), FIELDNAMES)
    _ensure_header(Path(args.summary), SUMMARY_FIELDNAMES)


def append(args: argparse.Namespace) -> None:
    results = Path(args.results)
    _ensure_header(results, FIELDNAMES)
    log_file = Path(args.log_file)
    mean, std = _read_final_returns(log_file)
    row = {
        "algorithm": args.algorithm,
        "variant": args.variant,
        "game": args.game,
        "seed": args.seed,
        "exit_code": args.exit_code,
        "final_return_mean": mean,
        "final_return_std": std,
        "wandb_run_name": args.run_name,
        "log_file": str(log_file),
    }
    with results.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)


def completed(args: argparse.Namespace) -> None:
    results = Path(args.results)
    if not results.exists():
        raise SystemExit(1)
    with results.open(newline="") as f:
        for row in csv.DictReader(f):
            if (
                row.get("algorithm") == args.algorithm
                and row.get("variant") == args.variant
                and row.get("game") == args.game
                and row.get("seed") == str(args.seed)
                and row.get("exit_code") == "0"
            ):
                raise SystemExit(0)
    raise SystemExit(1)


def summarize(args: argparse.Namespace) -> None:
    results = Path(args.results)
    summary = Path(args.summary)
    groups: dict[tuple[str, str, str], list[float]] = {}
    if results.exists():
        with results.open(newline="") as f:
            for row in csv.DictReader(f):
                try:
                    value = float(row.get("final_return_mean") or "")
                except ValueError:
                    continue
                key = (
                    row.get("algorithm", ""),
                    row.get("variant", ""),
                    row.get("game", ""),
                )
                groups.setdefault(key, []).append(value)

    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for (algorithm, variant, game), values in sorted(groups.items()):
            n = len(values)
            mean = sum(values) / n
            std = 0.0
            if n > 1:
                std = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))
            writer.writerow(
                {
                    "algorithm": algorithm,
                    "variant": variant,
                    "game": game,
                    "mean": mean,
                    "std": std,
                    "n": n,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--results", required=True)
    init_parser.add_argument("--summary", required=True)
    init_parser.set_defaults(func=init)

    append_parser = subparsers.add_parser("append")
    append_parser.add_argument("--results", required=True)
    append_parser.add_argument("--algorithm", required=True)
    append_parser.add_argument("--variant", required=True)
    append_parser.add_argument("--game", required=True)
    append_parser.add_argument("--seed", required=True)
    append_parser.add_argument("--exit-code", required=True)
    append_parser.add_argument("--run-name", required=True)
    append_parser.add_argument("--log-file", required=True)
    append_parser.set_defaults(func=append)

    completed_parser = subparsers.add_parser("completed")
    completed_parser.add_argument("--results", required=True)
    completed_parser.add_argument("--algorithm", required=True)
    completed_parser.add_argument("--variant", required=True)
    completed_parser.add_argument("--game", required=True)
    completed_parser.add_argument("--seed", required=True)
    completed_parser.set_defaults(func=completed)

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--results", required=True)
    summary_parser.add_argument("--summary", required=True)
    summary_parser.set_defaults(func=summarize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
