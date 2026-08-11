"""Discrete SAC-BBF: BBF with an explicit stochastic policy head."""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from torchrl.envs import EnvBase

from src.algorithms.bbf.bbf import BBFAlgorithm, _masked_nstep_return, _project_distribution
from src.algorithms.bbf.networks import SACBBFNetwork


class SACPolicyModule(TensorDictModuleBase):
    """Write actions sampled from, or greedy under, a SAC-BBF policy head."""

    def __init__(
        self,
        network: SACBBFNetwork,
        action_spec,
        *,
        obs_key: str,
        greedy: bool = False,
    ) -> None:
        self.in_keys = [obs_key]
        self.out_keys = ["action"]
        super().__init__()
        self.network = network
        self.action_spec = action_spec
        self.obs_key = obs_key
        self.greedy = greedy

    def forward(self, tensordict: TensorDict) -> TensorDict:
        logits = self.network.policy_logits(tensordict[self.obs_key])
        if logits.dim() == 1:
            index = logits.argmax(-1) if self.greedy else torch.distributions.Categorical(logits=logits).sample()
        else:
            index = logits.argmax(-1) if self.greedy else torch.distributions.Categorical(logits=logits).sample()
        tensordict.set("action", _index_to_action(index, self.action_spec, logits.shape[-1]))
        return tensordict


class SACBBFAlgorithm(BBFAlgorithm):
    """SAC-BBF variant using BBF's replay/reset machinery plus a policy loss."""

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        entropy_decay_period: int = 80_000,
        entropy_initial_coef: float = 1e-2,
        entropy_final_coef: float = 0.0,
        policy_lr: float = 1e-4,
        alpha_lr: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(device, **kwargs)
        self.entropy_decay_period = entropy_decay_period
        self.entropy_initial_coef = entropy_initial_coef
        self.entropy_final_coef = entropy_final_coef
        self.policy_lr = policy_lr
        self.alpha_lr = alpha_lr

    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        proof_env = make_env()
        obs_shape = tuple(proof_env.observation_spec[self.obs_key].shape)
        action_spec = proof_env.action_spec
        num_actions = int(action_spec.space.n)
        proof_env.close()
        self._action_spec = action_spec
        self._obs_shape = obs_shape

        def make_network() -> SACBBFNetwork:
            return SACBBFNetwork(
                obs_shape,
                num_actions,
                width_scale=self.width_scale,
                hidden_dim=self.hidden_dim,
                num_atoms=self.num_atoms,
                v_min=self.v_min,
                v_max=self.v_max,
                dueling=self.dueling,
                renorm=self.renormalize_latent,
            )

        self._make_network = make_network
        self.network = make_network().to(self.device)
        self.target_network = make_network().to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.requires_grad_(False)
        self.support = self.network.support.to(self.device)

        self._explore_policy = SACPolicyModule(
            self.target_network,
            action_spec,
            obs_key=self.obs_key,
            greedy=False,
        ).to(self.device)
        self._eval_policy = _SACEvalPolicy(
            self.target_network,
            action_spec,
            obs_key=self.obs_key,
        ).to(self.device)

        self.replay_buffer = self._make_replay_buffer(self.window)
        self.optimizer = self._make_optimizer()

    def _make_optimizer(self) -> torch.optim.AdamW:
        q_decay, q_no_decay = [], []
        policy_decay, policy_no_decay = [], []
        alpha = []
        for name, p in self.network.named_parameters():
            if not p.requires_grad:
                continue
            if name == "_log_alpha":
                alpha.append(p)
                continue
            is_policy = name.startswith(("policy_projection.", "predict_policy.", "policy."))
            has_decay = p.ndim > 1
            if is_policy:
                (policy_decay if has_decay else policy_no_decay).append(p)
            else:
                (q_decay if has_decay else q_no_decay).append(p)
        groups = []
        if q_decay:
            groups.append({"params": q_decay, "lr": self.lr, "weight_decay": self.weight_decay})
        if q_no_decay:
            groups.append({"params": q_no_decay, "lr": self.lr, "weight_decay": 0.0})
        if policy_decay:
            groups.append({"params": policy_decay, "lr": self.policy_lr, "weight_decay": self.weight_decay})
        if policy_no_decay:
            groups.append({"params": policy_no_decay, "lr": self.policy_lr, "weight_decay": 0.0})
        if alpha:
            groups.append({"params": alpha, "lr": self.alpha_lr, "weight_decay": 0.0})
        return torch.optim.AdamW(groups, eps=self.adam_eps)

    def step(self, batch: TensorDict) -> dict[str, float]:
        if batch.batch_dims != 1:
            raise ValueError(
                "SAC-BBF requires a temporally contiguous stream: set trainer.num_envs=1"
            )
        num_frames = batch.numel()
        self._store(batch)
        self._collected_frames += num_frames

        metrics = {
            "train/entropy_coef": self._entropy_coef(),
            "train/n_step": float(self._current_horizon()),
            "train/gamma": self._current_gamma(),
            "train/num_resets": float(self._num_resets),
        }
        if self._collected_frames < self.min_replay_history:
            return metrics

        num_updates = max(1, round(self.replay_ratio * num_frames))
        sums: dict[str, float] = {}
        for _ in range(num_updates):
            if (
                self.reset_interval > 0
                and self._steps_since_reset > self.reset_interval
                and (self.no_resets_after == 0 or self._grad_steps < self.no_resets_after)
            ):
                self._shrink_and_perturb()
            update_metrics = self._update()
            for key, value in update_metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value)
            self._grad_steps += 1
            self._steps_since_reset += 1

        metrics.update({key: value / num_updates for key, value in sums.items()})
        metrics["train/grad_steps"] = float(self._grad_steps)
        return metrics

    def _update(self) -> dict[str, float]:
        n = self._current_horizon()
        gamma = self._current_gamma()
        sample = self._sample()
        b = sample["action"].shape[0]
        arange = torch.arange(b, device=self.device)

        obs = sample["obs"]
        obs_t = self._augment(obs[:, 0])
        returns, bootstrap, alive = _masked_nstep_return(
            sample["reward"], sample["cut"], gamma, n
        )

        with torch.no_grad():
            obs_tn = self._augment(obs[:, n])
            next_policy_logits = self.network.policy_logits(obs_tn)
            next_actions = torch.distributions.Categorical(logits=next_policy_logits).sample()
            target_logits = self.target_network.q_logits(self.target_network.encode(obs_tn))
            target_probs = F.softmax(target_logits, -1)
            next_dist = target_probs[arange, next_actions]
            target_dist = _project_distribution(
                next_dist, returns, bootstrap, self.support, gamma**n
            )

            future = obs[:, 1 : self.spr_depth + 1].reshape(-1, *obs.shape[2:])
            future = self._augment(future)
            spr_targets = self.target_network.sac_project(
                self.target_network.encode(future)
            ).view(b, self.spr_depth, -1)
            spr_targets = F.normalize(spr_targets, dim=-1)

        latent_t = self.network.encode(obs_t)
        logits_t = self.network.q_logits(latent_t)
        q_values = self.network.q_values_from_logits(logits_t)
        log_p = F.log_softmax(logits_t[arange, sample["action"][:, 0]], -1)
        rl_loss_elem = -(target_dist * log_p).sum(-1)

        z_hat = latent_t
        predictions = []
        for j in range(self.spr_depth):
            z_hat = self.network.transition_model(z_hat, sample["action"][:, j])
            predictions.append(self.network.sac_predict(z_hat))
        spr_pred = F.normalize(torch.stack(predictions, 1), dim=-1)
        spr_loss_elem = ((spr_pred - spr_targets) ** 2).sum(-1)
        spr_loss_elem = (spr_loss_elem * alive[:, : self.spr_depth]).mean(dim=1) * 0.5

        policy_logits = self.network.policy_logits_from_latent(latent_t)
        log_prob = F.log_softmax(policy_logits.float(), dim=-1)
        prob = F.softmax(policy_logits.float(), dim=-1)
        policy_samples = torch.distributions.Categorical(logits=policy_logits).sample()
        sampled_q = q_values.float()[arange, policy_samples]
        expected_q = (q_values.float() * prob).sum(dim=-1)
        centered_q = sampled_q - expected_q
        entropy = -(prob * log_prob).sum(dim=-1)
        sampled_log_prob = log_prob[arange, policy_samples]
        policy_loss_elem = -(centered_q.detach() * sampled_log_prob) - self._entropy_coef() * entropy

        loss_elem = rl_loss_elem + self.spr_weight * spr_loss_elem + policy_loss_elem
        loss = (sample["weights"] * loss_elem).mean()

        self.optimizer.zero_grad()
        loss.backward()
        max_norm = 10.0 if self.max_grad_norm is None else self.max_grad_norm
        grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(), max_norm)
        self.optimizer.step()

        if self.prioritized:
            self.replay_buffer.update_priority(
                sample["start_idx"], rl_loss_elem.detach().cpu() + 1e-10
            )
        self._ema_update_target()

        return {
            "train/q_loss": float(rl_loss_elem.detach().mean()),
            "train/spr_loss": float(spr_loss_elem.detach().mean()),
            "train/policy_loss": float(policy_loss_elem.detach().mean()),
            "train/entropy": float(entropy.detach().mean()),
            "train/alpha": float(self.network.entropy_scale().detach()),
            "train/grad_norm": float(torch.as_tensor(grad_norm).detach()),
        }

    def _entropy_coef(self) -> float:
        if self.entropy_decay_period <= 0:
            return self.entropy_final_coef
        steps_left = self.entropy_decay_period - self._collected_frames
        bonus = (
            (self.entropy_initial_coef - self.entropy_final_coef)
            * steps_left
            / self.entropy_decay_period
        )
        bonus = min(
            max(bonus, 0.0),
            self.entropy_initial_coef - self.entropy_final_coef,
        )
        return self.entropy_final_coef + bonus

    @torch.no_grad()
    def _shrink_and_perturb(self) -> None:
        super()._shrink_and_perturb()
        self.target_network._log_alpha.copy_(self.network._log_alpha)


class _SACEvalPolicy(TensorDictModuleBase):
    def __init__(self, network: SACBBFNetwork, action_spec, *, obs_key: str) -> None:
        self.in_keys = [obs_key]
        self.out_keys = ["action"]
        super().__init__()
        self.greedy = SACPolicyModule(network, action_spec, obs_key=obs_key, greedy=True)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        return self.greedy(tensordict)


def _index_to_action(index: torch.Tensor, action_spec, num_actions: int) -> torch.Tensor:
    if len(action_spec.shape) > 0 and action_spec.shape[-1] == num_actions:
        return F.one_hot(index.long(), num_actions).to(dtype=action_spec.dtype, device=index.device)
    if len(action_spec.shape) > 0 and action_spec.shape[-1] == 1:
        return index.reshape(*index.shape, 1).to(dtype=action_spec.dtype)
    return index.to(dtype=getattr(action_spec, "dtype", torch.long))
