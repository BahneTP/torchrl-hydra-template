#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import wandb


DEFAULT_RESULTS = [
    Path("logs/plain_algorithms/der/atari100k_20260810_181403/results.csv"),
]


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    project: str
    entity: str
    name: str
    game: str
    seed: int
    tags: tuple[str, ...]


def read_run_sidecar(log_file: Path) -> tuple[str, str, str] | None:
    sidecar = log_file.parent / "checkpoints" / "wandb_run.json"
    if not sidecar.exists():
        return None
    data = json.loads(sidecar.read_text())
    return data["id"], data["project"], data["entity"]


def row_is_complete(row: dict[str, str]) -> bool:
    return (
        row.get("algorithm") == "der"
        and row.get("variant") == "plain"
        and row.get("exit_code") == "0"
        and bool(row.get("final_return_mean"))
        and bool(row.get("wandb_run_name"))
        and bool(row.get("log_file"))
    )


def collect_specs(paths: list[Path], include_descriptive: bool) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if not row_is_complete(row):
                    continue
                sidecar = read_run_sidecar(Path(row["log_file"]))
                if sidecar is None:
                    continue
                run_id, project, entity = sidecar
                tags = ["fig-plain-der-atari4-v1"]
                if include_descriptive:
                    tags.extend(["tl", "der", "plain", "atari4", row["game"].lower()])
                specs.append(
                    RunSpec(
                        run_id=run_id,
                        project=project,
                        entity=entity,
                        name=row["wandb_run_name"],
                        game=row["game"],
                        seed=int(row["seed"]),
                        tags=tuple(sorted(set(tags))),
                    )
                )
    return sorted(specs, key=lambda s: (s.game, s.seed))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tag finished plain DER W&B runs for figure generation."
    )
    parser.add_argument(
        "--results",
        type=Path,
        nargs="*",
        default=DEFAULT_RESULTS,
        help="results.csv files to read",
    )
    parser.add_argument("--apply", action="store_true", help="write tags to W&B")
    parser.add_argument(
        "--no-descriptive-tags",
        action="store_true",
        help="only add fig-* tags",
    )
    parser.add_argument(
        "--require-finished",
        action="store_true",
        help="skip online runs whose W&B state is not finished",
    )
    args = parser.parse_args()

    specs = collect_specs(args.results, include_descriptive=not args.no_descriptive_tags)
    print(f"found {len(specs)} finished local run rows with figure tags")

    api = wandb.Api()
    for spec in specs:
        run = api.run(f"{spec.entity}/{spec.project}/{spec.run_id}")
        if args.require_finished and run.state != "finished":
            print(f"skip {run.state:10s} {spec.name} {spec.run_id}")
            continue

        current = set(run.tags or [])
        wanted = set(spec.tags)
        merged = sorted(current | wanted)
        action = "ok " if current == set(merged) else "tag"

        print(
            f"{action} {run.state:10s} {spec.name:36s} "
            f"{spec.game:10s} seed={spec.seed} +{sorted(wanted - current)}"
        )

        if args.apply and current != set(merged):
            run.tags = merged
            run.update()

    if not args.apply:
        print("dry-run only; re-run with --apply to update W&B")


if __name__ == "__main__":
    main()
