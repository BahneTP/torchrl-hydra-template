"""Load and aggregate periodic evaluation histories from W&B runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


HISTORY_COLUMNS = [
    "global_step",
    "eval/return_mean",
    "eval/return_std",
    "eval/episodes",
]


def load_evaluation_curve(
    run_directory: str | Path,
    game: str,
    *,
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
    periodic_episodes: int = 10,
    refresh: bool = False,
    cache_directory: str | Path | None = None,
    evaluation_steps: Sequence[int] | None = None,
) -> tuple[list[int], list[float], list[float]]:
    """Return steps and across-seed mean/std of periodic evaluation returns.

    Each seed is resolved through ``<run>/<game>/seed_N/checkpoints/wandb_run.json``.
    Histories are cached as CSV files, and only rows whose ``eval/episodes``
    equals ``periodic_episodes`` are retained. This excludes the separate
    100-episode final evaluation at step 100,000.
    """
    run_directory = Path(run_directory).expanduser().resolve()
    if not run_directory.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_directory}")
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    if periodic_episodes <= 0:
        raise ValueError("periodic_episodes must be positive")
    if evaluation_steps is not None:
        evaluation_steps = tuple(int(step) for step in evaluation_steps)
        if not evaluation_steps or any(step < 0 for step in evaluation_steps):
            raise ValueError("evaluation_steps must be non-empty and non-negative")
        if len(set(evaluation_steps)) != len(evaluation_steps):
            raise ValueError("evaluation_steps must be unique")

    if cache_directory is None:
        cache_directory = (
            Path(__file__).resolve().parent
            / "cache"
            / "wandb_history"
            / run_directory.name
            / game
        )
    cache_directory = Path(cache_directory).expanduser().resolve()
    cache_directory.mkdir(parents=True, exist_ok=True)

    seed_histories: list[pd.DataFrame] = []
    for seed in seeds:
        metadata_path = (
            run_directory / game / f"seed_{seed}" / "checkpoints" / "wandb_run.json"
        )
        if not metadata_path.is_file():
            raise FileNotFoundError(f"missing W&B metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        missing_metadata = {"id", "project", "entity"}.difference(metadata)
        if missing_metadata:
            raise ValueError(
                f"missing keys in {metadata_path}: {sorted(missing_metadata)}"
            )

        run_id = str(metadata["id"])
        cache_path = cache_directory / f"seed_{seed}_{run_id}.csv"
        if cache_path.is_file() and not refresh:
            history = pd.read_csv(cache_path)
        else:
            try:
                import wandb

                api = wandb.Api()
                run_path = f"{metadata['entity']}/{metadata['project']}/{run_id}"
                run = api.run(run_path)
                history = pd.DataFrame(
                    run.scan_history(keys=HISTORY_COLUMNS, page_size=1_000)
                )
            except Exception as error:
                raise RuntimeError(
                    f"could not download W&B history for seed {seed} ({run_id}). "
                    f"Run `wandb login`, check network access, or reuse a populated "
                    f"cache at {cache_path}."
                ) from error

            missing_columns = set(HISTORY_COLUMNS).difference(history.columns)
            if missing_columns:
                raise ValueError(
                    f"W&B run {run_id} is missing history columns: "
                    f"{sorted(missing_columns)}"
                )
            history = history[HISTORY_COLUMNS].copy()
            history.to_csv(cache_path, index=False)
            # Use the serialized representation immediately so a refreshed call
            # and a later cache-only call return bit-identical aggregates.
            history = pd.read_csv(cache_path)

        missing_columns = set(HISTORY_COLUMNS).difference(history.columns)
        if missing_columns:
            raise ValueError(
                f"cached history {cache_path} is missing columns: "
                f"{sorted(missing_columns)}"
            )
        for column in HISTORY_COLUMNS:
            history[column] = pd.to_numeric(history[column], errors="coerce")
        history = history[
            history["eval/episodes"].eq(periodic_episodes)
            & history["global_step"].notna()
            & history["eval/return_mean"].notna()
        ].copy()
        history["global_step"] = history["global_step"].astype(int)
        history = (
            history.sort_values("global_step")
            .drop_duplicates("global_step", keep="last")
            .reset_index(drop=True)
        )
        if evaluation_steps is not None:
            history = history[history["global_step"].isin(evaluation_steps)].copy()
            actual_steps = history["global_step"].tolist()
            if actual_steps != list(evaluation_steps):
                raise ValueError(
                    f"seed {seed} is missing requested evaluation steps: expected "
                    f"{list(evaluation_steps)}, found {actual_steps}"
                )
        if history.empty:
            raise ValueError(
                f"no {periodic_episodes}-episode evaluations found for seed {seed}"
            )
        seed_histories.append(history)

    expected_steps = seed_histories[0]["global_step"].tolist()
    for seed, history in zip(seeds[1:], seed_histories[1:]):
        actual_steps = history["global_step"].tolist()
        if actual_steps != expected_steps:
            raise ValueError(
                f"evaluation steps differ for seed {seed}: expected "
                f"{expected_steps}, found {actual_steps}"
            )

    returns = np.asarray(
        [history["eval/return_mean"].to_numpy(dtype=float) for history in seed_histories]
    )
    means = returns.mean(axis=0)
    stds = returns.std(axis=0, ddof=1) if len(seed_histories) > 1 else np.zeros_like(means)
    return expected_steps, means.tolist(), stds.tolist()
