"""Environment factory for gymnasium- and dm_control-backed TorchRL envs.

Builds a (possibly vectorised) ``TransformedEnv`` from a small parameter
set and an explicit list of transform descriptors.

Each transform descriptor is a dict with a ``_target_`` key (a dotted path
to a ``torchrl.envs.transforms`` class) plus its constructor kwargs.
Transforms are instantiated fresh per ``make_env()`` call so each env has
independent transform state.
"""
from __future__ import annotations

import importlib
from contextlib import nullcontext
from functools import partial
from typing import Sequence


def make_env(
    name: str,
    num_envs: int = 1,
    device: str = "cpu",
    transforms: list | None = None,
    gym_kwargs: dict | None = None,
    gym_backend: str | None = None,
    backend: str = "gymnasium",
    task: str | None = None,
    **_: object,
):
    """Build a (possibly vectorised) ``TransformedEnv``.

    Args:
        name: gymnasium env id (e.g. ``"CartPole-v1"``) or, for
            ``backend="dm_control"``, a domain name (e.g. ``"cheetah"``).
        num_envs: number of parallel envs (>1 -> ``ParallelEnv``; workers
            always run on CPU because CUDA contexts cannot survive ``fork``).
        device: target device string.
        transforms: list of ``_target_``-keyed dicts to apply on top of the
            base env. ``None`` or empty -> bare base env.
        gym_kwargs: extra kwargs passed straight to ``GymEnv`` (e.g.
            ``{"frame_skip": 4, "from_pixels": True}``).
        gym_backend: optional gym backend name for ``set_gym_backend``.
        backend: ``"gymnasium"`` (default) or ``"dm_control"``.
        task: dm_control task name (e.g. ``"run"``); required for
            ``backend="dm_control"``, ignored otherwise.
    """
    worker_device = "cpu" if num_envs > 1 else device
    env_fn = _select_env_fn(
        backend, name, task, transforms, worker_device, gym_kwargs, gym_backend
    )

    if num_envs > 1:
        from torchrl.envs import ParallelEnv

        return ParallelEnv(num_envs, env_fn, mp_start_method="spawn")
    return env_fn()


def _select_env_fn(backend, name, task, transforms, device, gym_kwargs, gym_backend):
    """Return a no-arg env constructor for the requested backend."""
    if backend == "dm_control":
        return partial(
            _make_dmc_env, name=name, task=task, transforms=transforms, device=device
        )
    if backend == "gymnasium":
        return partial(
            _make_gymnasium_env,
            name=name,
            transforms=transforms,
            device=device,
            gym_kwargs=gym_kwargs,
            gym_backend=gym_backend,
        )
    raise ValueError(f"Unknown environment backend: {backend!r}")


def _instantiate_transform(cfg: dict):
    """Instantiate a transform from a ``_target_``-keyed dict (no Hydra runtime)."""
    cfg = dict(cfg)  # copy — don't mutate the caller
    target = cfg.pop("_target_")
    module_path, class_name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**cfg)


def _make_dmc_env(
    name: str,
    task: str | None,
    transforms: list | None,
    device: str,
):
    from torchrl.envs import DMControlEnv, TransformedEnv
    from torchrl.envs.transforms import Compose

    if task is None:
        raise ValueError("backend='dm_control' requires a `task` (e.g. task: run)")

    base_env = DMControlEnv(name, task, device=device)

    if not transforms:
        return base_env

    transform_objects = [_instantiate_transform(t) for t in transforms]
    return TransformedEnv(base_env, Compose(*transform_objects))


def _make_gymnasium_env(
    name: str,
    transforms: list | None,
    device: str,
    gym_kwargs: dict | None = None,
    gym_backend: str | None = None,
):
    from torchrl.envs import GymEnv, TransformedEnv
    from torchrl.envs.transforms import Compose

    backend_ctx = nullcontext()
    if gym_backend is not None:
        from torchrl.envs import set_gym_backend
        backend_ctx = set_gym_backend(gym_backend)

    with backend_ctx:
        base_env = GymEnv(name, device=device, **(gym_kwargs or {}))

    if not transforms:
        return base_env

    transform_objects = [_instantiate_transform(t) for t in transforms]
    return TransformedEnv(base_env, Compose(*transform_objects))
