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
    Path("logs/resnetlayer/der_resnet18_attentive_20260808_152406/results.csv"),
    Path("logs/resnetlayer/der_resnet18_linear_20260808_152354/results.csv"),
    Path("logs/resnetlayer/der_resnet18_linear_layer2_3games_20260810_195914/results.csv"),
]

ATARI4_GAMES = {"Assault", "BankHeist", "Jamesbond", "RoadRunner"}


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
    env_id: str


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
        row.get("exit_code") == "0"
        and bool(row.get("final_return_mean"))
        and bool(row.get("wandb_run_name"))
        and bool(row.get("log_file"))
    )


def figure_tags_for(row: dict[str, str]) -> list[str]:
    variant = row["variant"]
    game = row["game"]
    tags: list[str] = []

    if game == "Jamesbond" and variant.startswith("resnet18_"):
        tags.append(f"fig-resnetlayer-jamesbond-v1-{slug(variant)}")

    if variant == "resnet18_linear_layer2" and game in ATARI4_GAMES:
        tags.append("fig-resnetlayer-atari4-v1-resnet18-linear-layer2")

    return tags


def descriptive_tags_for(row: dict[str, str]) -> list[str]:
    variant = row["variant"]
    tags = ["tl", "der", "resnetlayer", "resnet18", slug(row["game"])]
    if "linear" in variant:
        tags.append("linear")
    if "attentive" in variant:
        tags.append("attentive")
    match = re.search(r"layer(\d+)", variant)
    if match:
        tags.append(f"layer{match.group(1)}")
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
                    env_id=f"{row['game']}-v5",
                )
    return sorted(specs_by_id.values(), key=lambda s: (s.variant, s.game, s.seed))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tag finished resnet-layer W&B runs for figure generation."
    )
    parser.add_argument(
        "--results",
        type=Path,
        nargs="*",
        default=DEFAULT_RESULTS,
        help="results.csv files to read",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write tags to W&B; default is dry-run",
    )
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
    parser.add_argument(
        "--fix-env-id",
        action="store_true",
        help="set W&B config.env_id from the local results.csv game column",
    )
    args = parser.parse_args()

    specs = collect_specs(args.results, include_descriptive=not args.no_descriptive_tags)
    print(f"found {len(specs)} finished local run rows with figure tags")

    api = wandb.Api()
    for spec in specs:
        path = f"{spec.entity}/{spec.project}/{spec.run_id}"
        run = api.run(path)
        if args.require_finished and run.state != "finished":
            print(f"skip {run.state:10s} {spec.name} {spec.run_id}")
            continue

        current = set(run.tags or [])
        wanted = set(spec.tags)
        merged = sorted(current | wanted)
        current_env_id = run.config.get("env_id")
        action = "tag"
        needs_env_id = args.fix_env_id and current_env_id != spec.env_id
        if current == set(merged) and not needs_env_id:
            action = "ok "

        print(
            f"{action} {run.state:10s} {spec.name:55s} "
            f"{spec.game:10s} seed={spec.seed} +{sorted(wanted - current)}"
            + (
                f" env_id:{current_env_id!r}->{spec.env_id!r}"
                if needs_env_id
                else ""
            )
        )

        if args.apply and (current != set(merged) or needs_env_id):
            if current != set(merged):
                run.tags = merged
            if needs_env_id:
                run.config["env_id"] = spec.env_id
            run.update()

    if not args.apply:
        print("dry-run only; re-run with --apply to update W&B")


if __name__ == "__main__":
    main()
