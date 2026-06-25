from __future__ import annotations

import torch
import torch.nn as nn
from collections import deque
from typing import Callable
from omegaconf import DictConfig, OmegaConf, open_dict

_ACCEL_TO_TORCH = {"gpu": "cuda", "mps": "mps"}

OmegaConf.register_new_resolver(
    "to_torch_device",
    lambda accel: _ACCEL_TO_TORCH.get(str(accel), str(accel)),
)


def _patch_devices(cfg: DictConfig, device_str: str) -> None:
    """Replace all device/storage_device strings recursively before passing to dreamer internals."""
    with open_dict(cfg):
        for k in list(cfg.keys()):
            v = cfg[k]
            if k == "device" and isinstance(v, str):
                cfg[k] = device_str
            elif isinstance(v, DictConfig):
                _patch_devices(v, device_str)


from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs import EnvBase
from src.algorithms.dreamer.buffer import Buffer

from src.algorithms.base import BaseAlgorithm, TrainingState

# Import the modified Dreamer class you provided
from src.algorithms.dreamer.dreamer_model import Dreamer


class DreamerPolicy(nn.Module):  # TODO Recheck this
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
        self._ep_scores: deque[float] = deque(maxlen=30)
        self._ep_lengths: deque[float] = deque(maxlen=30)
        self._total_updates = 0
        self._last_video_frame = 0
        self._last_agent_video_frame = 0
        self.video_log_every = 50_000  # log world model video every N frames
        self._last_train_metrics: dict[str, float] = {}
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

        # 1. Instantiate the raw mathematical engine
        self.model = Dreamer(self.dreamer_config, obs_space, act_space).to(self.device)

        # 2. Instantiate the Policy Wrappers
        self._explore_policy = DreamerPolicy(self.model, explore=True).to(self.device)
        self._eval_policy = DreamerPolicy(self.model, explore=False).to(self.device)

        # 3. Create a Sequence-Aware Replay Buffer
        self.replay_buffer = Buffer(self.buffer_config)

    def _episode_metrics(self) -> dict[str, float]:
        if not self._ep_scores:
            return {}
        out: dict[str, float] = {"episode/score": max(self._ep_scores)}
        if self._ep_lengths:
            out["episode/length"] = self._ep_lengths[-1]
        return out

    def step(self, td: TensorDict) -> dict[str, float]:
        """Receives a single frame from the StatefulTrainer and conditionally updates."""

        # 1. Append directly to the sequence buffer.
        self.replay_buffer.add_transition(td)

        # 2. Increment the empirical data tracker
        transitions_added = int(td.batch_size[0]) if len(td.batch_size) > 0 else 1
        self._collected_frames += transitions_added * self.action_repeat

        # Track episode completions in a rolling window for log-boundary reporting.
        done = td.get(("next", "done"), default=None)
        if done is not None and done.bool().any():
            mask = done.bool().reshape(-1)
            ep_reward = td.get(("next", "episode_reward"), default=None)
            if ep_reward is not None:
                for score in ep_reward.reshape(-1)[mask].float().tolist():
                    self._ep_scores.append(score)
            ep_length = td.get(("next", "step_count"), default=None)
            if ep_length is not None:
                for length in ep_length.reshape(-1)[mask].float().tolist():
                    self._ep_lengths.append(length)

        # 3. The Minimum Viability Constraint
        min_required_frames = (
            self.batch_length + 1
        ) * self.action_repeat  # At least enough frames to sample
        if self._collected_frames <= min_required_frames:
            self._next_update_target = self._collected_frames + self._frames_per_update
            return self._episode_metrics()

        # 4. Proportional update count — fires exactly once per step in normal
        # operation (frames_per_batch=1), but catches up if frames_per_batch > 1.
        update_num = 0
        while self._collected_frames >= self._next_update_target:
            update_num += 1
            self._next_update_target += self._frames_per_update

        # 5. Execute Backpropagation Through Time
        for _ in range(update_num):
            data, index, initial = self.replay_buffer.sample()
            (stoch, deter), _metrics = self.model.update(data, initial)
            self.replay_buffer.update(index, stoch, deter)
            self._last_train_metrics = {f"train/{k}": v for k, v in _metrics.items()}

        if update_num > 0:
            self._total_updates += update_num
            self._last_train_metrics["train/opt/updates"] = self._total_updates

            import wandb

            if (
                self.dreamer_config.rep_loss == "dreamer"
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

        return {**self._episode_metrics(), **self._last_train_metrics}

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
        # Save the raw Dreamer model parameters and optimizers
        # You can use the `recursively_collect_optim_state_dict` tool from your utils file here!
        return TrainingState(
            step=self._collected_frames,
            policy_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.model._optimizer.state_dict(),
            extra={"collected_frames": self._collected_frames},
        )

    def _load_training_state(self, state: TrainingState) -> None:
        self.model.load_state_dict(state.policy_state_dict)
        self.model._optimizer.load_state_dict(state.optimizer_state_dict)
        if state.extra and "collected_frames" in state.extra:
            self._collected_frames = int(state.extra["collected_frames"])
            self._last_train_frame = self._collected_frames
            self._next_update_target = self._collected_frames + self._frames_per_update

    def get_collector_config(self) -> dict[str, any]:
        """Provides configuration parameters governing the environment collection loop.

        For Dreamer, this enforces single-step (or single-microbatch) data
        ingestion to synchronize perfectly with the model's training ratio.
        """
        from types import SimpleNamespace

        return SimpleNamespace(
            # Collect exactly 1 agent step before relinquishing control to the algorithm
            frames_per_batch=1,
            # Initialization stochasticity is handled natively by the NoopResetEnv transform;
            # the collector does not need to intervene.
            init_random_frames=0,
            # A value of 0 dictates that the collector relies strictly on the StepCounter
            # transform to manage episodic truncation boundaries.
            max_frames_per_traj=0,
        )
