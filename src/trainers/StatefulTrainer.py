"""Step-by-step trainer for stateful/recurrent algorithms (e.g., Dreamer).

Unlike `StepTrainer` which uses `SyncDataCollector` to yield i.i.d. batches,
this trainer manually steps the environment. This is required for algorithms
where the policy is stateful (e.g., maintains an RSSM hidden state) and where
the replay buffer must receive strictly sequential, contiguous trajectories.

The trainer owns the loop, device placement, and logging schedules.
The algorithm owns the replay buffer, RSSM state, and gradient updates.
"""

from __future__ import annotations

import time

import torch
from torchrl.envs.utils import step_mdp

from src.trainers.BaseTrainer import BaseTrainer, TrainerEvent, fire_callbacks


class StatefulTrainer(BaseTrainer):
    """Manually steps the environment and feeds sequential transitions to the algorithm."""

    def _training_loop(self) -> dict[str, float]:
        total_frames = int(self.trainer_cfg.total_frames)
        log_every = int(self.trainer_cfg.log_every_n_steps)

        explore_policy = self.algorithm.get_explore_policy()

        # Accumulators for logging completed episodes
        episode_returns: list[float] = []
        episode_lengths: list[float] = []

        # Initial environment reset and device placement
        td = self.train_env.reset()
        td = td.to(self.device)

        metrics: dict[str, float] = {}

        while self._step < total_frames:
            collect_start = time.perf_counter()

            # 1. Action selection (Algorithm's policy handles RSSM state internally)
            with torch.no_grad():
                td = explore_policy(td)

            # 2. Environment step
            td = self.train_env.step(td)
            collect_time = time.perf_counter() - collect_start

            frames = int(td.batch_size[0]) if td.batch_size else 1
            self._step += frames

            # 3. Episode tracking (Delegated to TorchRL Transforms)
            done = td.get(("next", "done"), default=None)
            terminated = td.get(("next", "terminated"), default=None)

            is_done = None
            if done is not None:
                is_done = done.clone()
            if terminated is not None:
                is_done = (
                    terminated.clone() if is_done is None else (is_done | terminated)
                )

            # Extract metrics only for environments that actually finished this step
            if is_done is not None and is_done.any():
                mask = is_done.squeeze(-1) if is_done.dim() > 1 else is_done

                ep_rewards = td.get(("next", "episode_reward"), default=None)
                if ep_rewards is not None:
                    episode_returns.extend(ep_rewards[mask].flatten().tolist())

                ep_lengths = td.get(("next", "step_count"), default=None)
                if ep_lengths is not None:
                    episode_lengths.extend(ep_lengths[mask].flatten().tolist())

            # 4. Hand transition to algorithm
            # The algorithm will push to its buffer and decide if it needs to update.
            step_start = time.perf_counter()
            metrics = self.algorithm.step(td)
            step_time = time.perf_counter() - step_start

            # 5. Advance MDP ("next" state becomes current state)
            # keep_other=True preserves stateful flags like `is_first` or policy latents
            td = step_mdp(td, keep_other=True)  #! Important

            if td.get("done", td.get("terminated")).any():
                td = self.train_env.reset()

            # 6. Logging boundary
            if self._should_log(log_every, frames):
                if episode_returns:
                    metrics["train/episode_reward"] = sum(episode_returns) / len(
                        episode_returns
                    )
                    metrics["train/episode_length"] = sum(episode_lengths) / len(
                        episode_lengths
                    )
                    episode_returns.clear()
                    episode_lengths.clear()

                total_time = collect_time + step_time
                metrics["time/collect"] = collect_time
                metrics["time/step"] = step_time
                metrics["time/speed"] = frames / total_time if total_time > 0 else 0.0

                fire_callbacks(
                    TrainerEvent.ON_STEP_END,
                    self.callbacks,
                    metrics=metrics,
                    step=self._step,
                )

        return metrics

    def evaluate(self, num_episodes: int) -> dict[str, float]:
        """Run evaluation episodes with the deterministic (greedy) policy."""
        from torchrl.envs.utils import ExplorationType, set_exploration_type

        eval_env = self.eval_environment.make_env(num_envs=1, device=str(self.device))
        policy = self.algorithm.get_policy()

        returns: list[float] = []

        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            for _ in range(num_episodes):
                td = eval_env.reset()
                td = td.to(self.device)

                done_flag = False
                while not done_flag:
                    td = policy(td)
                    td = eval_env.step(td)

                    # Safely check for done/terminated
                    done = td.get(
                        ("next", "done"), default=torch.zeros(1, dtype=torch.bool)
                    )
                    terminated = td.get(
                        ("next", "terminated"), default=torch.zeros(1, dtype=torch.bool)
                    )
                    done_flag = done.any().item() or terminated.any().item()

                    if done_flag:
                        # Grab the final calculated score directly from TorchRL
                        final_reward = td.get(
                            ("next", "episode_reward"),
                            default=td.get(("next", "reward")),
                        )
                        returns.append(final_reward.sum().item())

                    td = td["next"]

        eval_env.close()
        t = torch.tensor(returns, dtype=torch.float32)
        return {
            "eval/return_mean": t.mean().item(),
            "eval/return_std": t.std().item() if len(returns) > 1 else 0.0,
            "eval/return_min": t.min().item(),
            "eval/return_max": t.max().item(),
        }
