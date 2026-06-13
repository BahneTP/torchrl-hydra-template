"""
This file is from the R2Dreamer Repository: https://github.com/NM512/r2dreamer
It is modified to integrate with our Hydra Pipeline and TorchRL
Changes are marked with Comments #!
"""

import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler


class Buffer:
    def __init__(self, config):
        self.device = torch.device(config.device)
        self.storage_device = torch.device(config.storage_device)
        self.batch_size = int(config.batch_size)
        self.batch_length = int(config.batch_length)
        self._buffer = ReplayBuffer(
            #! Flat (ndim=1) storage: every transition occupies one slot in a 1-D
            # sequence.  The previous ndim=2 design stored each transition as a
            # 1-timestep trajectory (shape [N, 1, ...]), which made SliceSampler
            storage=LazyTensorStorage(
                max_size=int(config.max_size), device=self.storage_device
            ),
            #! traj_key="episode" groups all of env-i's steps into one long stream
            sampler=SliceSampler(
                num_slices=self.batch_size,
                traj_key="episode",
                end_key=None,
                truncated_key=None,
                strict_length=True,
            ),
            prefetch=0,
            batch_size=self.batch_size * (self.batch_length + 1),  # +1 for context
        )

    def add_transition(self, data):
        #! data: TensorDict with batch_size (num_envs,) from the Collector.
        # Stamp each step with a constant env index as the "episode" key so that
        # SliceSampler(traj_key="episode") can group env-i's transitions into one
        # contiguous stream and sample freely across game-over boundaries.
        num_envs = data.batch_size[0] if data.batch_size else 1
        data.set("episode", torch.arange(num_envs, dtype=torch.int32))
        self._buffer.extend(data)

    def sample(self):
        sample_td, info = self._buffer.sample(return_info=True)
        # SliceSampler returns B*(T+1) contiguous steps in a flat TensorDict.
        # Reshape to (B, T+1) so each row is one training sequence.
        sample_td = sample_td.view(-1, self.batch_length + 1)
        src_dev = sample_td.device
        if src_dev.type == "cpu" and self.device.type == "cuda":
            sample_td = sample_td.pin_memory().to(self.device, non_blocking=True)
        elif src_dev != self.device:
            sample_td = sample_td.to(self.device, non_blocking=True)
        #! First timestep of each sequence is used only to warm-start RSSM state.
        initial = (sample_td["stoch"][:, 0], sample_td["deter"][:, 0])
        data = sample_td[:, 1:]
        data.set_("action", sample_td["action"][:, :-1])  # action is 1 step back
        #! TorchRL wraps storage indices in a tuple; R2Dreamer iterated over it.
        # With flat 1-D storage there is exactly one element — unwrap it so update()
        # receives a plain 1-D tensor of shape (B*(T+1),).
        raw_index = info["index"]
        index = raw_index[0] if isinstance(raw_index, tuple) else raw_index
        return data, index, initial

    def update(
        self, index, stoch, deter
    ):  #! R2Dreamer used 2-D storage so index was a list of tensors and wrote `self._buffer[index[1], index[0]]`
        # index: flat 1-D tensor (B*(T+1),) — includes the initial context step.  #!
        # stoch/deter: (B, T, ...) — posterior latents for the T training steps only.  #!
        B, T = stoch.shape[0], stoch.shape[1]  #!
        training_index = index.view(B, T + 1)[:, 1:].reshape(-1)  # (B*T,)  #!
        self._buffer[training_index] = TensorDict(  #!
            {  #!
                "stoch": stoch.reshape(-1, *stoch.shape[2:]),  #!
                "deter": deter.reshape(-1, *deter.shape[2:]),  #!
            },  #!
            batch_size=(B * T,),  #!
        )  #!

    def count(self):
        if self._buffer.storage.shape is None:
            return 0
        return self._buffer.storage.shape.numel()
