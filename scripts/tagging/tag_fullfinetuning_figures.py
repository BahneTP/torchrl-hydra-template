#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import wandb


DEFAULT_RESULTS = [
    Path("logs/fullfinetuning/der_resnet18_layer2_20260811_025944/results.csv"),
    Path("logs/fullfinetuning/der_resnet18_layer4_20260809_001602/results.csv"),
    Path("logs/fullfinetuning/der_dinov2_block12_20260809_001637/results.csv"),
]


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    project: str
    entity: str
    name: str
    variant: str
    game: str
    seed: int
    tags: tuple[str, ...]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def read_run_sidecar(log_file: Path) -> tuple[str, str, str] | None:
    sidecar = log_file.parent / "checkpoints" / "wandb_run.json"
    if not sidecar.exists():
        return None
    data = json.loads(sidecar.read_text())
    return data["id"], data["project"], data["entity"]


def row_is_complete(row: dict[str, str]) -> bool:
    return (
        row.get("algorithm") == "der"
        and row.get("exit_code") == "0"
        and bool(row.get("final_return_mean"))
        and bool(row.get("wandb_run_name"))
        and bool(row.get("log_file"))
    )


def figure_tags_for(row: dict[str, str]) -> list[str]:
    variant = row["variant"]
    game = row["game"]
    tags: list[str] = []

    if game == "Jamesbond" and "_full_" in variant:
        tags.append(f"fig-fullfinetuning-jamesbond-v1-{slug(variant)}")

    if game == "Jamesbond" and variant.startswith("resnet18_full_layer2_"):
        tags.append("fig-fullfinetuning-jamesbond-v1-resnet18-layer2-lr-sweep")
    if game == "Jamesbond" and variant.startswith("resnet18_full_layer4_"):
        tags.append("fig-fullfinetuning-jamesbond-v1-resnet18-layer4-lr-sweep")
    if game == "Jamesbond" and variant.startswith("dinov2_full_block12_"):
        tags.append("fig-fullfinetuning-jamesbond-v1-dinov2-block12-lr-sweep")

    return tags


def descriptive_tags_for(row: dict[str, str]) -> list[str]:
    variant = row["variant"]
    tags = ["tl", "der", "fullfinetuning", slug(row["game"])]
    if variant.startswith("resnet18_"):
        tags.append("resnet18")
    if variant.startswith("dinov2_"):
        tags.append("dinov2")
    for token in variant.split("_"):
        if token.startswith("layer") or token.startswith("block"):
            tags.append(token)
    match = re.search(r"lr_(.+)$", variant)
    if match:
        tags.append("lr-" + match.group(1).replace("_", "-"))
    return tags


def collect_specs(paths: list[Path], include_descriptive: bool) -> list[RunSpec]:
    specs_by_id: dict[str, RunSpec] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if not row_is_complete(row):
                    continue

                figure_tags = figure_tags_for(row)
                if not figure_tags:
                    continue

                sidecar = read_run_sidecar(Path(row["log_file"]))
                if sidecar is None:
                    continue
                run_id, project, entity = sidecar
                tags = list(figure_tags)
                if include_descriptive:
                    tags.extend(descriptive_tags_for(row))

                existing = specs_by_id.get(run_id)
                if existing:
                    tags = sorted(set(existing.tags).union(tags))
                else:
                    tags = sorted(set(tags))

                specs_by_id[run_id] = RunSpec(
                    run_id=run_id,
                    project=project,
                    entity=entity,
                    name=row["wandb_run_name"],
                    variant=row["variant"],
                    game=row["game"],
                    seed=int(row["seed"]),
                    tags=tuple(tags),
                )
    return sorted(specs_by_id.values(), key=lambda s: (s.variant, s.game, s.seed))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tag finished full-finetuning W&B runs for figure generation."
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
            f"{action} {run.state:10s} {spec.name:58s} "
            f"{spec.game:10s} seed={spec.seed} +{sorted(wanted - current)}"
        )

        if args.apply and current != set(merged):
            run.tags = merged
            run.update()

    if not args.apply:
        print("dry-run only; re-run with --apply to update W&B")


if __name__ == "__main__":
    main()
