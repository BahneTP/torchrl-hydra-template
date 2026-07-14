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
import os
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
    **_: object,
):
    """Build a (possibly vectorised) ``TransformedEnv``.

    Args:
        name: env name. For ``backend="gymnasium"`` a gymnasium id (e.g.
            ``"CartPole-v1"``); for ``backend="dm_control"`` a
            ``"<domain>/<task>"`` pair (e.g. ``"cheetah/run"``).
        num_envs: number of parallel envs (>1 -> ``ParallelEnv``).
        device: target device string. ``ParallelEnv`` workers always run on
            CPU because CUDA contexts cannot survive ``fork``; the collector
            moves data to ``device`` after collection.
        transforms: list of ``_target_``-keyed dicts to apply on top of the
            base env. ``None`` or empty -> bare base env.
        gym_kwargs: extra kwargs passed straight to ``GymEnv`` (e.g.
            ``{"frame_skip": 4, "from_pixels": True}``). Gymnasium only.
        gym_backend: optional gym backend name for ``set_gym_backend``
            (e.g. ``"gymnasium"``); if ``None`` torchrl picks the default.
        backend: ``"gymnasium"`` (default) or ``"dm_control"``.
    """
    worker_device = "cpu" if num_envs > 1 else device

    if backend == "dm_control":
        env_fn = partial(
            _make_dm_control_env,
            name=name,
            transforms=transforms,
            device=worker_device,
        )
    elif backend == "gymnasium":
        env_fn = partial(
            _make_gymnasium_env,
            name=name,
            transforms=transforms,
            device=worker_device,
            gym_kwargs=gym_kwargs,
            gym_backend=gym_backend,
        )
    else:
        raise ValueError(
            f"Unsupported environment backend {backend!r}; "
            "expected 'gymnasium' or 'dm_control'."
        )

    if num_envs > 1:
        from torchrl.envs import ParallelEnv

        return ParallelEnv(num_envs, env_fn, mp_start_method="spawn")
    return env_fn()


def _instantiate_transform(cfg: dict):
    """Instantiate a transform from a ``_target_``-keyed dict (no Hydra runtime)."""
    cfg = dict(cfg)  # copy — don't mutate the caller
    target = cfg.pop("_target_")
    module_path, class_name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**cfg)


def _apply_transforms(base_env, transforms: list | None):
    from torchrl.envs import TransformedEnv
    from torchrl.envs.transforms import Compose

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
    from torchrl.envs import GymEnv

    backend_ctx = nullcontext()
    if gym_backend is not None:
        from torchrl.envs import set_gym_backend
        backend_ctx = set_gym_backend(gym_backend)

    with backend_ctx:
        base_env = GymEnv(name, device=device, **(gym_kwargs or {}))

    return _apply_transforms(base_env, transforms)


def _make_dm_control_env(
    name: str,
    transforms: list | None,
    device: str,
):
    # dm_control initialises a renderer at import time; default to headless
    # (no rendering) unless the user configured a GL backend themselves.
    os.environ.setdefault("MUJOCO_GL", "disabled")
    from torchrl.envs.libs.dm_control import DMControlEnv

    domain, _, task = name.partition("/")
    if not task:
        raise ValueError(
            f"dm_control env name must be '<domain>/<task>' (e.g. 'cheetah/run'), "
            f"got {name!r}."
        )
    base_env = DMControlEnv(domain, task, device=device)
    return _apply_transforms(base_env, transforms)
