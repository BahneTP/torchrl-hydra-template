from __future__ import annotations

import torch
import torch.nn as nn
from typing import Callable
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import (
    ReplayBuffer,
    LazyTensorStorage,
    TensorDictReplayBuffer,
    SliceSampler,
)
from torchrl.envs import EnvBase

from src.algorithms.base import BaseAlgorithm, TrainingState

# Import the modified Dreamer class you provided
from src.algorithms.dreamer.dreamer_model import Dreamer


class DreamerPolicy(nn.Module):
    """Wraps the Dreamer model to manage the RSSM hidden states across time steps."""

    def __init__(self, model: Dreamer, explore: bool = True):
        super().__init__()
        self.model = model
        self.explore = explore

    def forward(self, td: TensorDict) -> TensorDict:
        # 1. Bridge TorchRL and Dreamer: Add a temporary batch dimension if missing
        is_unbatched = td.batch_dims == 0
        if is_unbatched:
            td = td.unsqueeze(0)  # Safely converts (4,) to (1, 4)

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

        # 4. Forward pass through Dreamer's act() method (Now safely batched!)
        action, new_state = self.model.act(td, state, eval=not self.explore)

        # 5. Inject the new action and updated latent states
        td.set("action", action)
        td.update(new_state)

        # 6. Strip the temporary batch dimension before returning to the environment
        if is_unbatched:
            td = td.squeeze(0)

        return td


class DreamerAlgorithm(BaseAlgorithm):
    """Stateful algorithm wrapper for DreamerV3/EfficientDreamer."""

    def __init__(
        self,
        config,  # Your Hydra/OmegaConf config for Dreamer hyperparameters
        device: torch.device | None = None,
        sequence_length: int = 64,
        batch_size: int = 16,
        train_ratio: int = 4,  # Train 1 time for every 4 frames collected
        prefill_frames: int = 5000,
        buffer_size: int = 1_000_000,
    ) -> None:
        super().__init__(device)
        self.config = config
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.prefill_frames = prefill_frames
        self.buffer_size = buffer_size

        self._collected_frames = 0
        self._last_train_frame = 0

    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        proof_env = make_env()
        obs_space = proof_env.observation_spec
        act_space = proof_env.action_spec

        # 1. Instantiate the raw mathematical engine
        self.model = Dreamer(self.config, obs_space, act_space).to(self.device)

        # 2. Instantiate the Policy Wrappers
        self._explore_policy = DreamerPolicy(self.model, explore=True).to(self.device)
        self._eval_policy = DreamerPolicy(self.model, explore=False).to(self.device)

        # 3. Create a Sequence-Aware Replay Buffer
        # SliceSampler forces the buffer to return contiguous trajectories of `sequence_length`
        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=self.buffer_size, device=self.device),
            sampler=SliceSampler(
                slice_len=self.sequence_length,
                strict_length=True,
                end_key="dummy_done",
                traj_key="dummy_traj",
                # Let truncated_key fallback to its default; we will hide the data instead.
            ),
            batch_size=self.batch_size * self.sequence_length,
        )

    def step(self, td: TensorDict) -> dict[str, float]:
        """Receives a single frame from the StatefulTrainer and conditionally updates."""

        td_expanded = td.unsqueeze(0).clone()

        # 1. Inject artificial boundary markers (all zeros)
        ref = td_expanded.get("done")
        td_expanded.set("dummy_done", torch.zeros_like(ref, device=self.device))
        td_expanded.set(
            "dummy_traj", torch.zeros_like(ref, dtype=torch.long, device=self.device)
        )

        # 2. THE CRITICAL FIX: Hide the native truncation keys.
        # If SliceSampler finds them, it concatenates them with dummy_done,
        # creating a shape[1]=2 tensor that crashes its own internal checks.
        if "truncated" in td_expanded.keys():
            td_expanded.rename_key_("truncated", "masked_truncated")

        if "next" in td_expanded.keys() and "truncated" in td_expanded["next"].keys():
            td_expanded["next"].rename_key_("truncated", "masked_truncated")

        # 3. Append to the sequence buffer
        self.replay_buffer.extend(td_expanded)

        frames_added = int(td.batch_size[0]) if len(td.batch_size) > 0 else 1
        self._collected_frames += frames_added

        # 4. Check if we have enough data to start training
        if self._collected_frames < self.prefill_frames:
            return {}

        metrics = {}

        # 5. Check Train Ratio: Are we due for an update?
        if (self._collected_frames - self._last_train_frame) >= self.train_ratio:
            self._last_train_frame = self._collected_frames

            # Sample a batch of shape [Batch_Size, Sequence_Length, ...]
            batch = self.replay_buffer.sample()

            # Because our policy injects the latent states into the TensorDict,
            # they are saved in the buffer! We can just grab the very first
            # hidden state of the sequence to use as the `initial` state.
            initial_stoch = batch["stoch"][:, 0].contiguous()
            initial_deter = batch["deter"][:, 0].contiguous()
            initial = (initial_stoch, initial_deter)

            # 4. Trigger the Dreamer optimization cycle
            # This calls the tweaked method we adjusted in step 1.
            metrics = self.model.update(batch, initial)

        return metrics

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

    def get_collector_config(self) -> dict[str, any]:
        """Provides configuration parameters governing the environment collection loop.

        For Dreamer, this enforces single-step (or single-microbatch) data
        ingestion to synchronize perfectly with the model's training ratio.
        """
        return {
            # Collect exactly 1 step (per parallel environment) before handing control
            # back to the algorithm's step() function.
            "frames_per_batch": 1,
            # Use the global training frame allocation managed by the trainer configuration
            "total_frames": int(self.config.get("total_frames", 500_000)),
        }
