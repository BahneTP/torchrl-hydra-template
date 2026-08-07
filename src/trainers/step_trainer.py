"""Step-based trainer using ``SyncDataCollector``.

Each iteration:  collector yields one batch of transitions
                 ->  ``algorithm.step(batch)``
                 ->  fire callbacks if it's a logging step.

The trainer owns the loop, the collector and the callbacks; everything that
affects learning lives in the algorithm.

Per-iteration metrics emitted on logging boundaries mirror the torchrl SOTA
DQN reference (sota-implementations/dqn/dqn_cartpole.py):
  - ``train/episode_reward``, ``train/episode_length``: mean over episodes
    that completed since the previous log (accumulated across collector
    batches so small ``frames_per_batch`` still reports every episode).
  - ``train/q_values``: mean Q-value of the actions actually executed
    (current batch).
  - ``time/collect``, ``time/step``, ``time/speed``: collector wait, in-step
    optimisation time, and frames/second for the iteration.
"""
from __future__ import annotations

import time
from math import prod

from tensordict import TensorDict

from src.trainers.base import BaseTrainer, TrainerEvent, fire_callbacks


class StepTrainer(BaseTrainer):
    def setup(self) -> None:
        super().setup()
        self._create_collector()

    def _create_collector(self) -> None:
        from torchrl.collectors import Collector

        cc = self.algorithm.get_collector_config()
        self.collector = Collector(
            create_env_fn=self.train_env,
            policy=self.algorithm.get_explore_policy(),
            frames_per_batch=cc.frames_per_batch,
            total_frames=int(self.trainer_cfg.total_frames),
            init_random_frames=cc.init_random_frames,
            max_frames_per_traj=cc.max_frames_per_traj,
            device=self.device,
            storing_device=self.device,
        )

    def _training_loop(self) -> dict[str, float]:
        log_every = int(self.trainer_cfg.log_every_n_steps)
        metrics: dict[str, float] = {}
        # Episode completions can land in any collector batch. With small
        # ``frames_per_batch`` (e.g. DER's 4) almost none coincide with a
        # logging boundary, so accumulate across the window and emit means
        # at log time.
        pending_episode_rewards: list[float] = []
        pending_episode_lengths: list[float] = []

        collector_iter = iter(self.collector)
        while True:
            collect_start = time.perf_counter()
            try:
                batch = next(collector_iter)
            except StopIteration:
                break
            collect_time = time.perf_counter() - collect_start

            batch_frames = batch.numel()
            self._step += batch_frames

            step_start = time.perf_counter()
            metrics = self.algorithm.step(batch)
            step_time = time.perf_counter() - step_start

            log_step = getattr(self.algorithm, "log_step", self._step)
            pop = getattr(self.algorithm, "pop_train_metrics", None)

            if pop is None:
                ep_rewards, ep_lengths, instant = _batch_metrics(batch)
                pending_episode_rewards.extend(ep_rewards)
                pending_episode_lengths.extend(ep_lengths)
            else:
                instant = {}

            if self._should_log(log_every, batch_frames):
                log_metrics = pop() if pop is not None else dict(metrics)
                if pop is None:
                    if pending_episode_rewards:
                        log_metrics["train/episode_reward"] = (
                            sum(pending_episode_rewards)
                            / len(pending_episode_rewards)
                        )
                        pending_episode_rewards.clear()
                    if pending_episode_lengths:
                        log_metrics["train/episode_length"] = (
                            sum(pending_episode_lengths)
                            / len(pending_episode_lengths)
                        )
                        pending_episode_lengths.clear()
                    log_metrics.update(instant)
                total_time = collect_time + step_time
                log_metrics["time/collect"] = collect_time
                log_metrics["time/step"] = step_time
                log_metrics["time/speed"] = (
                    batch_frames / total_time if total_time > 0 else 0.0
                )
                fire_callbacks(
                    TrainerEvent.ON_STEP_END,
                    self.callbacks,
                    metrics=log_metrics,
                    step=log_step,
                )

        if hasattr(self.algorithm, "finalize_metrics"):
            self.algorithm.finalize_metrics()

        return metrics



def _batch_metrics(
    batch: TensorDict,
) -> tuple[list[float], list[float], dict[str, float]]:
    """Split completed-episode stats from instantaneous batch metrics.

    Episode rewards/lengths are returned as per-episode lists so the trainer
    can accumulate them across collector batches between logging boundaries.
    Instantaneous metrics (e.g. ``train/q_values``) are returned as a dict
    for the current batch only.

    Each metric is emitted only when the underlying TensorDict key is present:
    ``RewardSum`` for ``episode_reward``, ``StepCounter`` for ``step_count``,
    and a ``QValueActor``-style policy for ``action_value`` / ``action``.
    """
    flat = batch.reshape(-1)
    episode_rewards: list[float] = []
    episode_lengths: list[float] = []
    out: dict[str, float] = {}

    done = flat.get(("next", "done"), default=None)
    if done is not None and done.bool().any():
        mask = done.bool()
        rewards = flat.get(("next", "episode_reward"), default=None)
        if rewards is not None:
            episode_rewards.extend(rewards[mask].float().reshape(-1).tolist())
        lengths = flat.get(("next", "step_count"), default=None)
        if lengths is not None:
            episode_lengths.extend(lengths[mask].float().reshape(-1).tolist())

    # Q-value of the action actually executed.
    # Handles both one-hot encoding (action shape [B, A]) and categorical
    # encoding (action shape [B], integer indices).
    action_value = flat.get("action_value", default=None)
    action = flat.get("action", default=None)
    if action_value is not None and action is not None:
        if action_value.dim() > 2 and action_value.shape[-2] == 1:
            action_value = action_value.squeeze(-2)
        if action_value.dim() > 2:
            return episode_rewards, episode_lengths, out
        while action.dim() > 1 and action.shape[-1] == 1:
            action = action.squeeze(-1)
        if action.shape == action_value.shape:
            out["train/q_values"] = (
                (action_value * action).sum().item() / flat.numel()
            )
        else:
            if action.numel() != prod(action_value.shape[:-1]):
                return episode_rewards, episode_lengths, out
            action_index = action.long().reshape(*action_value.shape[:-1], 1)
            out["train/q_values"] = (
                action_value.gather(-1, action_index)
                .mean()
                .item()
            )

    return episode_rewards, episode_lengths, out
