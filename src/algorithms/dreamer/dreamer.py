from __future__ import annotations

import torch
import torch.nn as nn
from typing import Callable
from omegaconf import DictConfig, open_dict

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs import EnvBase

from hydra.utils import get_class

from src.algorithms.dreamer.buffer import Buffer
from src.algorithms.base import BaseAlgorithm, CollectorConfig, TrainingState
from src.algorithms.dreamer.model import (
    DreamerV3,
)  # default; overridden by dreamer_config._target_


class DreamerPolicy(nn.Module):
    """Wraps the Dreamer model to manage the RSSM hidden states across time steps."""

    def __init__(self, model: Dreamer, explore: bool = True):
        super().__init__()
        self.model = model
        self.explore = explore

    def forward(self, td: TensorDict) -> TensorDict:
        # 1. Robust dimensional standardisation
        is_unbatched = len(td.batch_size) == 0
        if is_unbatched:
            td = td.unsqueeze(0)  # Mutates () to (1,)

        B = td.batch_size[0]

        # 2. Check if we need to initialize or reset hidden states
        is_first = td.get("is_first", default=None)

        if "stoch" not in td or (is_first is not None and is_first.any()):
            init_state = self.model.get_initial_state(B)

            if "stoch" not in td:
                td.update(init_state)
            else:
                # Selectively reset only the environments that returned is_first=True
                mask = is_first.squeeze(-1) if is_first.dim() > 1 else is_first
                td["stoch"][mask] = init_state["stoch"][mask]
                td["deter"][mask] = init_state["deter"][mask]
                td["prev_action"][mask] = init_state["prev_action"][mask]

        # 3. Package the previous state
        state = TensorDict(
            {
                "stoch": td["stoch"],
                "deter": td["deter"],
                "prev_action": td.get("action", td["prev_action"]),
            },
            batch_size=td.batch_size,
        )

        # 4. Forward pass through Dreamer's act() method
        action, new_state = self.model.act(td, state, eval=not self.explore)

        # 5. Inject the new action and updated latent states
        td.set("action", action)
        td.update(new_state)

        # 6. Strip the temporary batch dimension to maintain Collector compatibility
        if is_unbatched:
            td = td.squeeze(0)

        return td


def _patch_devices(cfg: DictConfig, device_str: str) -> None:
    """Recursively replace Hydra accelerator strings with PyTorch device strings.

    "gpu" is replaced with the exact device the trainer resolved (e.g. "cuda:2").
    """
    with open_dict(cfg):
        for k in list(cfg.keys()):
            v = cfg[k]
            if k in ("device", "storage_device") and isinstance(v, str):
                cfg[k] = device_str if v == "gpu" else v
            elif isinstance(v, DictConfig):
                _patch_devices(v, device_str)


class DreamerAlgorithm(BaseAlgorithm):
    """Stateful algorithm wrapper for DreamerV3/EfficientDreamer."""

    def __init__(
        self,
        dreamer_config,
        buffer_config,
        device: torch.device | None = None,
        train_ratio: float = 128.0,
        action_repeat: int = 4,
        agent_video_log_every: int = 50_000,
        agent_video_max_steps: int = 200,
    ) -> None:
        # Resolve model class from dreamer_config._target_
        target = dreamer_config.get("_target_", None)
        self._model_cls = get_class(target) if target else DreamerV3
        super().__init__(device)
        self.dreamer_config = dreamer_config
        self.buffer_config = buffer_config
        self.train_ratio = train_ratio
        self.action_repeat = action_repeat
        self.agent_video_log_every = agent_video_log_every
        self.agent_video_max_steps = agent_video_max_steps
        self.batch_length = buffer_config.batch_length
        self.batch_size = buffer_config.batch_size

        self._collected_frames = 0
        batch_steps = self.batch_size * self.batch_length
        self._frames_per_update = (batch_steps / self.train_ratio) * self.action_repeat
        self._next_update_target = self._frames_per_update
        self._total_updates = 0
        self._last_video_frame = 0
        self._last_agent_video_frame = 0
        self.video_log_every = 50_000  # log world model video every N frames
        self._metrics_accum: dict[str, list[float]] = {}
        self._ep_scores: list[tuple[float, int]] = []  # (score, frame_at_done)
        self._ep_lengths: list[tuple[int, int]] = []  # (length, frame_at_done)
        self._make_env: Callable[[], EnvBase] | None = None

    @property
    def log_step(self) -> int:
        """Logical frame count (collector steps × action_repeat) for the WandB x-axis."""
        return self._collected_frames

    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        self._make_env = make_env
        proof_env = make_env()
        obs_space = proof_env.observation_spec
        act_space = proof_env.action_spec

        device_str = str(self.device)
        _patch_devices(self.dreamer_config, device_str)
        _patch_devices(self.buffer_config, device_str)

        # 1. Instantiate the model (class selected via dreamer_config._target_)
        self.model = self._model_cls(self.dreamer_config, obs_space, act_space).to(
            self.device
        )

        # 2. Instantiate the Policy Wrappers
        self._explore_policy = DreamerPolicy(self.model, explore=True).to(self.device)
        self._eval_policy = DreamerPolicy(self.model, explore=False).to(self.device)

        # 3. Create a Sequence-Aware Replay Buffer
        self.replay_buffer = Buffer(self.buffer_config)

    def step(self, td: TensorDict) -> dict[str, float]:
        """Receives a single frame from the StatefulTrainer and conditionally updates."""

        # 1. Append directly to the sequence buffer.
        self.replay_buffer.add_transition(td)

        # 2. Increment the empirical data tracker
        transitions_added = int(td.batch_size[0]) if len(td.batch_size) > 0 else 1
        self._collected_frames += transitions_added * self.action_repeat

        # 3. Track episode completions — store (value, frame) so each episode
        # can be logged at its actual x-axis position rather than averaged.
        done = td.get(("next", "done"), default=None)
        if done is not None and done.bool().any():
            mask = done.bool().squeeze(-1) if done.dim() > 1 else done.bool()
            scores = td.get(("next", "episode_reward"), default=None)
            lengths = td.get(("next", "step_count"), default=None)
            for idx in mask.nonzero(as_tuple=True)[0]:
                score = scores[idx].item() if scores is not None else None
                length = lengths[idx].item() if lengths is not None else None
                if score is not None:
                    self._ep_scores.append((score, self._collected_frames))
                if length is not None:
                    self._ep_lengths.append((length, self._collected_frames))

        # 4. The Minimum Viability Constraint
        min_required_frames = (
            self.batch_length + 1
        ) * self.action_repeat  # At least enough frames to sample
        if self._collected_frames <= min_required_frames:
            self._next_update_target = self._collected_frames + self._frames_per_update
            return {}

        # 5. Proportional update count — fires exactly once per step in normal
        # operation (frames_per_batch=1), but catches up if frames_per_batch > 1.
        update_num = 0
        while self._collected_frames >= self._next_update_target:
            update_num += 1
            self._next_update_target += self._frames_per_update

        # 6. Execute Backpropagation Through Time
        for _ in range(update_num):
            data, index, initial = self.replay_buffer.sample()
            (stoch, deter), _metrics = self.model.update(data, initial)
            self.replay_buffer.update(index, stoch, deter)
            for k, v in _metrics.items():
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                if k.startswith("loss/"):
                    key = f"losses/{k[5:]}"
                elif k.startswith("opt/"):
                    key = k
                else:
                    key = f"train/{k}"
                self._metrics_accum.setdefault(key, []).append(val)

        if update_num > 0:
            self._total_updates += update_num

            import wandb

            if (
                hasattr(self.model, "decoder")
                and self._collected_frames - self._last_video_frame
                >= self.video_log_every
            ):
                self._last_video_frame = self._collected_frames
                if wandb.run is not None:
                    wandb.log(
                        {"video/world_model": self._make_video(data, initial)},
                        step=self._collected_frames,
                    )

            if (
                self._collected_frames - self._last_agent_video_frame
                >= self.agent_video_log_every
            ):
                self._last_agent_video_frame = self._collected_frames
                if wandb.run is not None:
                    video = self._record_agent_video()
                    if video is not None:
                        wandb.log(
                            {"video/agent": video},
                            step=self._collected_frames,
                        )

        return {}

    def pop_train_metrics(self) -> dict[str, float]:
        """Return mean metrics accumulated since the last call, then reset.

        Called by StepTrainer at log boundaries. Training losses are averaged
        over the window. Episodes are logged individually to W&B at the frame
        step when they actually completed, so each episode is a separate point.
        """
        out: dict[str, float] = {}
        if self._metrics_accum:
            out = {k: sum(v) / len(v) for k, v in self._metrics_accum.items()}
            self._metrics_accum.clear()
        out["opt/updates"] = self._total_updates
        if self._ep_scores:
            import wandb

            if wandb.run is not None:
                for (score, frame), (length, _) in zip(
                    self._ep_scores, self._ep_lengths
                ):
                    wandb.log(
                        {"episode/score": score, "episode/length": length}, step=frame
                    )
            self._ep_scores.clear()
            self._ep_lengths.clear()
        return out

    @torch.no_grad()
    def _make_video(self, data, initial):
        import wandb

        # (1, T, 3, H*3, W) — truth / reconstruction / open-loop stacked vertically
        frames = self.model.video_pred(data[:1], tuple(s[:1] for s in initial))
        # (T, 3, H*3, W) channel-first uint8 — wandb.Video accepts (T, C, H, W)
        frames = (frames[0].cpu().nan_to_num(0.0).clamp(0, 1) * 255).byte().numpy()
        return wandb.Video(frames, fps=10, format="mp4")

    @torch.no_grad()
    def _record_agent_video(self):
        import wandb
        from torchrl.envs.utils import step_mdp

        env = self._make_env()
        td = env.reset()
        frames = []

        for _ in range(self.agent_video_max_steps):
            img = td.get("image", default=None)
            if img is not None:
                frames.append(img.cpu())

            td = self._eval_policy(td)
            td = env.step(td)

            if td.get(("next", "done"), default=torch.zeros(1)).bool().any():
                img = td["next"].get("image", default=None)
                if img is not None:
                    frames.append(img.cpu())
                td = env.reset()
            else:
                td = step_mdp(td)

        env.close()

        if not frames:
            return None

        video = torch.stack(frames)  # (T, C, H, W)
        video = (video.nan_to_num(0.0).clamp(0, 1) * 255).byte().numpy()
        return wandb.Video(video, fps=20, format="mp4")

    def get_policy(self) -> TensorDictModule:
        return self._eval_policy

    def get_explore_policy(self) -> TensorDictModule:
        return self._explore_policy

    def _get_training_state(self) -> TrainingState:
        from src.algorithms.dreamer.tools import recursively_collect_optim_state_dict

        return TrainingState(
            step=self._collected_frames,
            policy_state_dict=self.model.state_dict(),
            optimizer_state_dict=recursively_collect_optim_state_dict(self.model),
            extra={"scheduler": self.model._scheduler.state_dict()},
        )

    def _load_training_state(self, state: TrainingState) -> None:
        from src.algorithms.dreamer.tools import recursively_load_optim_state_dict

        self.model.load_state_dict(state.policy_state_dict)
        recursively_load_optim_state_dict(self.model, state.optimizer_state_dict)
        if state.extra and "scheduler" in state.extra:
            self.model._scheduler.load_state_dict(state.extra["scheduler"])
        self._collected_frames = state.step
        self._next_update_target = self._collected_frames + self._frames_per_update

    def get_collector_config(self) -> CollectorConfig:
        """One collector call = one gradient update. Adapts to any train_ratio."""
        return CollectorConfig(
            frames_per_batch=int(self._frames_per_update / self.action_repeat),
            init_random_frames=0,
            max_frames_per_traj=-1,  # no collector-forced resets; StepCounter owns truncation
        )
