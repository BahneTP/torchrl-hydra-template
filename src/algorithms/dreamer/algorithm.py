"""DreamerV3 — Mastering Diverse Domains through World Models.

Hafner et al., 2023 · https://arxiv.org/abs/2301.04104
Adapted from NM512/r2dreamer — https://github.com/NM512/r2dreamer

Pseudocode:
    Init world model (encoder, RSSM, decoder, reward head, continue head)
    Init actor, critic, slow target critic
    loop:
        Encode obs → RSSM posterior → sample action → step env → store
        If enough data:
            Sample (B, T) sequence batch from replay buffer
            World model loss:
                KL(posterior || prior) — dynamics + representation
                -log p(obs_t | z_t, h_t) — reconstruction
                -log p(r_t  | z_t, h_t) — reward prediction
                -log p(c_t  | z_t, h_t) — continue prediction
            Imagination rollout for T_imag steps:
                ẑ_τ ~ prior(ĥ_τ),  â_τ ~ actor(feat_τ)
            Actor loss  = -E[Σ V_λ(feat_τ)] - η H[actor]
            Critic loss = E[-log V_ξ(V_λ)]   (with slow target regulariser)
            Update all parameters jointly with one Adam optimiser
"""

from __future__ import annotations

import copy
import types
from typing import Callable

import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.envs import EnvBase

from src.algorithms.base import BaseAlgorithm, CollectorConfig, TrainingState
from src.algorithms.dreamer.distributions import symlog, to_f32
from src.algorithms.dreamer.networks import ConvDecoder, ConvEncoder, MLPHead, ReturnEMA
from src.algorithms.dreamer.rssm import RSSM
from src.algorithms.dreamer.tools import weight_init_

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ns(**kwargs) -> types.SimpleNamespace:
    """Shorthand for SimpleNamespace — used to build r2dreamer-style config objects."""
    return types.SimpleNamespace(**kwargs)


def _mlp_ns(
    units: int,
    layers: int,
    act: str,
    name: str,
    device: str,
    symlog_inputs: bool = False,
) -> types.SimpleNamespace:
    return _ns(
        units=units,
        layers=layers,
        act=act,
        name=name,
        device=device,
        symlog_inputs=symlog_inputs,
    )


# ---------------------------------------------------------------------------
# Simple vector-observation encoder (bypasses r2dreamer's config-heavy MLP)
# ---------------------------------------------------------------------------


class _VecEncoder(nn.Module):
    """Symlog MLP encoder for proprioceptive / low-dim vector observations."""

    def __init__(
        self, obs_dim: int, units: int, layers: int, act: str = "SiLU"
    ) -> None:
        super().__init__()
        act_cls = getattr(torch.nn, act)
        seq: list[nn.Module] = []
        inp = obs_dim
        for _ in range(layers):
            seq += [
                nn.Linear(inp, units),
                nn.RMSNorm(units, eps=1e-4, dtype=torch.float32),
                act_cls(),
            ]
            inp = units
        self.net = nn.Sequential(*seq)
        self.out_dim = units
        self.apply(weight_init_)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, T, D) — applies symlog preprocessing as in DreamerV3 §B
        return self.net(symlog(obs))


# ---------------------------------------------------------------------------
# Episode replay buffer
# ---------------------------------------------------------------------------


class _EpisodeBuffer:
    """Replay buffer that stores complete episodes and samples fixed-length sequences.

    DreamerV3 trains on (B, T) sequence chunks that must never straddle two
    different episodes — the RSSM can't unroll across episode boundaries.

    With num_envs > 1, each environment's trajectory is tracked independently.
    An episode is considered complete when done=True or terminated=True.
    Incomplete (in-progress) episodes are not sampled.
    """

    def __init__(self, max_steps: int, num_envs: int) -> None:
        self._max_steps = max_steps
        # In-progress episodes: one list of TensorDicts per env.
        self._current: list[list[TensorDict]] = [[] for _ in range(num_envs)]
        # Completed episodes, each shape (T,).
        self._episodes: list[TensorDict] = []
        self._total_steps: int = 0

    @property
    def total_steps(self) -> int:
        return self._total_steps

    def add(self, td: TensorDict) -> None:
        """Store one environment tick.  td has batch_size (num_envs,) or ()."""
        B = td.batch_size[0] if td.batch_size else 1
        for i in range(B):
            step: TensorDict = td[i] if B > 1 else td
            self._current[i].append(step.cpu())
            self._total_steps += 1

            done = step.get(("next", "done"), torch.zeros((), dtype=torch.bool))
            term = step.get(("next", "terminated"), torch.zeros((), dtype=torch.bool))
            if (done | term).any().item():
                ep = torch.stack(self._current[i])  # (T,)
                self._episodes.append(ep)
                self._current[i] = []

        # Evict oldest episodes when over capacity.
        while self._total_steps > self._max_steps and self._episodes:
            removed = self._episodes.pop(0)
            self._total_steps -= removed.batch_size[0]

    def ready(self, batch_length: int) -> bool:
        return any(ep.batch_size[0] >= batch_length for ep in self._episodes)

    def sample(
        self, batch_size: int, batch_length: int, device: torch.device
    ) -> TensorDict:
        """Return a (B, T) TensorDict of contiguous sequences."""
        valid = [ep for ep in self._episodes if ep.batch_size[0] >= batch_length]
        seqs: list[TensorDict] = []
        for _ in range(batch_size):
            ep = valid[int(torch.randint(len(valid), (1,)))]
            start = int(torch.randint(ep.batch_size[0] - batch_length + 1, (1,)))
            seqs.append(ep[start : start + batch_length])
        return torch.stack(seqs, 0).to(device)  # (B, T)


# ---------------------------------------------------------------------------
# Stateful policy wrapper
# ---------------------------------------------------------------------------


class DreamerPolicy:
    """Wraps frozen encoder + RSSM + actor for stateful environment collection.

    Maintains (stoch, deter, prev_action) across timesteps.
    Resets those states for any environment where is_init=True / is_first=True
    appears in the input TensorDict (set by TorchRL's InitTracker transform or
    the environment on reset).

    Called as:  td = policy(td)   →  adds "action" key to td.

    Returned by DreamerV3Algorithm.get_explore_policy() / get_policy().
    """

    def __init__(
        self,
        encoder: nn.Module,
        rssm: RSSM,
        actor: MLPHead,
        device: torch.device,
        num_envs: int,
        act_dim: int,
        obs_key: str,
        pixel_obs: bool,
        eval_mode: bool = False,
    ) -> None:
        self._encoder = encoder
        self._rssm = rssm
        self._actor = actor
        self._device = device
        self._obs_key = obs_key
        self._pixel_obs = pixel_obs
        self._eval_mode = eval_mode
        stoch, deter = rssm.initial(num_envs)
        self._stoch = stoch
        self._deter = deter
        self._prev_action = torch.zeros(
            num_envs, act_dim, dtype=torch.float32, device=device
        )

    @torch.no_grad()
    def __call__(self, td: TensorDict) -> TensorDict:
        td = td.to(self._device)

        # Episode-start flags → zero out hidden state for reset envs.
        reset = td.get("is_init", td.get("is_first", None))
        if reset is None:
            reset = torch.zeros(
                self._stoch.shape[0], dtype=torch.bool, device=self._device
            )
        else:
            reset = reset.bool().flatten().to(self._device)

        # Encode obs.  The encoder expects (B, T, ...) so we unsqueeze T=1,
        # encode, then squeeze back to (B, E).
        obs = td[self._obs_key].to(self._device)
        if self._pixel_obs:
            obs = obs.float() / 255.0
            embed = self._encoder(obs.unsqueeze(1)).squeeze(1)
        else:
            embed = self._encoder(obs.unsqueeze(1)).squeeze(1)

        # Single RSSM posterior step.
        stoch, deter, _ = self._rssm.obs_step(
            self._stoch, self._deter, self._prev_action, embed, reset
        )
        feat = self._rssm.get_feat(stoch, deter)
        action_dist = self._actor(feat)
        action = action_dist.mode if self._eval_mode else action_dist.rsample()

        # Persist state.
        self._stoch = stoch
        self._deter = deter
        self._prev_action = action

        return td.set("action", action)


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------


class DreamerV3Algorithm(BaseAlgorithm):
    """DreamerV3 world-model RL algorithm.

    Use with StatefulTrainer — NOT StepTrainer (no SyncDataCollector).

    Defaults are for continuous-action proprioceptive environments (DMC walker,
    cheetah).  Switch obs_key="pixels" and set cnn_depth for image envs.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        # --- Observation ---------------------------------------------------
        obs_key: str = "observation",
        # --- RSSM ----------------------------------------------------------
        deter: int = 4096,
        stoch: int = 32,
        discrete: int = 32,  # categories per stochastic variable
        hidden: int = 1024,  # projection width inside Deter
        blocks: int = 8,  # block-GRU groups
        obs_layers: int = 1,  # MLP layers for posterior head
        img_layers: int = 1,  # MLP layers for prior head
        dyn_layers: int = 1,  # hidden layers inside Deter
        unimix_ratio: float = 0.01,
        kl_free: float = 1.0,  # free-nats threshold
        # --- Encoder (vector obs) -----------------------------------------
        enc_units: int = 1024,
        enc_layers: int = 1,
        # --- Encoder (pixel obs; ignored when obs_key != "pixels") ---------
        cnn_depth: int = 32,
        cnn_mults: tuple = (1, 2, 4, 8),
        cnn_kernel: int = 4,
        # --- Heads (reward, continue, value) --------------------------------
        head_units: int = 512,
        head_layers: int = 2,
        num_bins: int = 255,  # bins for symexp-twohot reward/value heads
        # --- Actor ---------------------------------------------------------
        act_units: int = 512,
        act_layers: int = 2,
        act_entropy: float = 3e-4,  # entropy bonus coefficient
        act_outscale: float = 1.0,
        # --- Imagination ---------------------------------------------------
        imag_horizon: int = 15,
        discount: float = 0.997,  # γ = 1 - 1/333 (per DreamerV3 paper)
        lam: float = 0.95,  # λ for λ-returns
        # --- Slow target value ---------------------------------------------
        slow_target_update: int = 1,  # update every N grad steps
        slow_target_fraction: float = 0.02,
        # --- Optimisation --------------------------------------------------
        lr: float = 1e-4,
        adam_eps: float = 1e-8,
        max_grad_norm: float = 1000.0,
        # --- Loss weights --------------------------------------------------
        scale_dyn: float = 0.5,
        scale_rep: float = 0.1,
        scale_rew: float = 1.0,
        scale_con: float = 1.0,
        scale_dec: float = 1.0,  # reconstruction (image obs only)
        scale_actor: float = 1.0,
        scale_value: float = 1.0,
        scale_repval: float = 1.0,
        # --- Replay buffer -------------------------------------------------
        buffer_size: int = 1_000_000,
        batch_size: int = 16,
        batch_length: int = 64,
        prefill_steps: int = 2500,
        train_ratio: float = 512.0,  # env steps between gradient updates
    ) -> None:
        super().__init__(device)
        # Store every HP as an attribute for setup() to read.
        self.obs_key = obs_key
        self.deter = deter
        self.stoch = stoch
        self.discrete = discrete
        self.hidden = hidden
        self.blocks = blocks
        self.obs_layers = obs_layers
        self.img_layers = img_layers
        self.dyn_layers = dyn_layers
        self.unimix_ratio = unimix_ratio
        self.kl_free = kl_free
        self.enc_units = enc_units
        self.enc_layers = enc_layers
        self.cnn_depth = cnn_depth
        self.cnn_mults = tuple(cnn_mults)
        self.cnn_kernel = cnn_kernel
        self.head_units = head_units
        self.head_layers = head_layers
        self.num_bins = num_bins
        self.act_units = act_units
        self.act_layers = act_layers
        self.act_entropy = act_entropy
        self.act_outscale = act_outscale
        self.imag_horizon = imag_horizon
        self.discount = discount
        self.lam = lam
        self.slow_target_update = slow_target_update
        self.slow_target_fraction = slow_target_fraction
        self.lr = lr
        self.adam_eps = adam_eps
        self.max_grad_norm = max_grad_norm
        self.scale_dyn = scale_dyn
        self.scale_rep = scale_rep
        self.scale_rew = scale_rew
        self.scale_con = scale_con
        self.scale_dec = scale_dec
        self.scale_actor = scale_actor
        self.scale_value = scale_value
        self.scale_repval = scale_repval
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.batch_length = batch_length
        self.prefill_steps = prefill_steps
        self.train_ratio = train_ratio
        self._collected_frames: int = 0
        self._grad_steps: int = 0
        self._slow_value_updates: int = 0
        # Fractional accumulator: fires an update when >= 1.0.
        self._update_credit: float = 0.0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        """Build all networks after reading env specs."""
        proof_env = make_env()
        obs_spec = proof_env.observation_spec[self.obs_key]
        action_spec = proof_env.action_spec
        proof_env.close()

        dev = str(self.device)
        pixel_obs = len(obs_spec.shape) == 3  # (H, W, C)
        self._pixel_obs = pixel_obs

        # --- Encoder -------------------------------------------------------
        if pixel_obs:
            h, w, c = obs_spec.shape
            enc_cfg = _ns(
                depth=self.cnn_depth,
                mults=list(self.cnn_mults),
                kernel_size=self.cnn_kernel,
                norm=True,
                act="SiLU",
            )
            self.encoder = ConvEncoder(enc_cfg, input_shape=(h, w, c)).to(self.device)
            embed_size = self.encoder.out_dim
        else:
            obs_dim = int(obs_spec.shape[-1])
            self.encoder = _VecEncoder(obs_dim, self.enc_units, self.enc_layers).to(
                self.device
            )
            embed_size = self.enc_units

        # --- Action space --------------------------------------------------
        discrete_actions = (
            hasattr(action_spec, "n")
            or hasattr(action_spec, "space")
            and hasattr(action_spec.space, "n")
        )
        if discrete_actions:
            try:
                self._act_dim = int(action_spec.space.n)
            except AttributeError:
                self._act_dim = int(action_spec.n)
        else:
            self._act_dim = int(action_spec.shape[-1])
        self._discrete_actions = discrete_actions

        # --- RSSM ----------------------------------------------------------
        rssm_cfg = _ns(
            stoch=self.stoch,
            deter=self.deter,
            hidden=self.hidden,
            discrete=self.discrete,
            blocks=self.blocks,
            obs_layers=self.obs_layers,
            img_layers=self.img_layers,
            dyn_layers=self.dyn_layers,
            unimix_ratio=self.unimix_ratio,
            initial="learned",
            act="SiLU",
            device=dev,
        )
        self.rssm = RSSM(rssm_cfg, embed_size, self._act_dim).to(self.device)
        feat_size = self.rssm.feat_size  # stoch*discrete + deter

        # --- Decoder (pixel obs only) -------------------------------------
        self._use_decoder = pixel_obs
        if pixel_obs:
            dec_cfg = _ns(
                depth=self.cnn_depth,
                mults=list(self.cnn_mults),
                kernel_size=self.cnn_kernel,
                units=self.enc_units,
                bspace=self.blocks,
                act="SiLU",
            )
            h, w, c = obs_spec.shape
            self.decoder = ConvDecoder(
                dec_cfg,
                deter=self.deter,
                flat_stoch=self.rssm.flat_stoch,
                shape=(c, h, w),
            ).to(self.device)

        # --- Prediction heads ---------------------------------------------
        _base = dict(
            units=self.head_units,
            layers=self.head_layers,
            act="SiLU",
            device=dev,
            symlog_inputs=False,
        )
        _twohot = _ns(name="symexp_twohot", bin_num=self.num_bins)
        _binary = _ns(name="binary")

        self.reward_head = MLPHead(
            _ns(
                **_base,
                name="reward",
                shape=(self.num_bins,),
                outscale=1.0,
                dist=_twohot,
            ),
            inp_dim=feat_size,
        ).to(self.device)

        self.cont_head = MLPHead(
            _ns(**_base, name="cont", shape=(1,), outscale=1.0, dist=_binary),
            inp_dim=feat_size,
        ).to(self.device)

        # --- Actor --------------------------------------------------------
        if discrete_actions:
            _actor_dist = _ns(name="onehot", unimix_ratio=self.unimix_ratio)
            actor_out_shape = (self._act_dim,)
        else:
            _actor_dist = _ns(name="bounded_normal", min_std=0.1, max_std=1.0)
            actor_out_shape = (self._act_dim,)

        self.actor = MLPHead(
            _ns(
                **{**_base, "units": self.act_units, "layers": self.act_layers},
                name="actor",
                shape=actor_out_shape,
                outscale=self.act_outscale,
                dist=_actor_dist,
            ),
            inp_dim=feat_size,
        ).to(self.device)

        # --- Critic (online + slow target) --------------------------------
        self.value = MLPHead(
            _ns(
                **_base,
                name="value",
                shape=(self.num_bins,),
                outscale=1.0,
                dist=_twohot,
            ),
            inp_dim=feat_size,
        ).to(self.device)

        self._slow_value = copy.deepcopy(self.value)
        for p in self._slow_value.parameters():
            p.requires_grad_(False)

        self._return_ema = ReturnEMA(device=self.device)

        # --- Frozen copies for actor-critic rollouts ----------------------
        # These are periodically synced via clone_and_freeze() and used in
        # _imagine() so that actor-critic gradients don't flow into the world
        # model parameters.
        self._sync_frozen_nets()

        # --- Replay buffer ------------------------------------------------
        num_envs = getattr(proof_env, "num_envs", 1)
        self._buffer = _EpisodeBuffer(
            max_steps=self.buffer_size,
            num_envs=num_envs,
        )
        self._num_envs = num_envs

        # --- Optimizer (single, covers all parameters) --------------------
        all_params = (
            list(self.encoder.parameters())
            + list(self.rssm.parameters())
            + list(self.reward_head.parameters())
            + list(self.cont_head.parameters())
            + list(self.actor.parameters())
            + list(self.value.parameters())
        )
        if self._use_decoder:
            all_params += list(self.decoder.parameters())

        self.optimizer = torch.optim.Adam(all_params, lr=self.lr, eps=self.adam_eps)

    def _sync_frozen_nets(self) -> None:
        """Create or refresh frozen copies of all networks.

        The frozen copies are used inside _imagine() so that actor-critic
        gradients do not flow back into the world model parameters.  Called
        once during setup() and automatically after each update step via
        _update_slow_target().
        """

        def _freeze_copy(src: nn.Module) -> nn.Module:
            dst = copy.deepcopy(src)
            dst.eval()
            for p in dst.parameters():
                p.requires_grad_(False)
            return dst

        self._frozen_encoder = _freeze_copy(self.encoder)
        self._frozen_rssm = _freeze_copy(self.rssm)
        self._frozen_reward = _freeze_copy(self.reward_head)
        self._frozen_cont = _freeze_copy(self.cont_head)
        self._frozen_actor = _freeze_copy(self.actor)
        self._frozen_value = _freeze_copy(self.value)
        self._frozen_slow_value = _freeze_copy(self._slow_value)

    # ------------------------------------------------------------------
    # Collector interface
    # ------------------------------------------------------------------

    def get_collector_config(self) -> CollectorConfig:
        raise NotImplementedError(
            "DreamerV3Algorithm is not compatible with StepTrainer / SyncDataCollector. "
            "Use StatefulTrainer instead."
        )

    def get_policy(self) -> DreamerPolicy:
        return DreamerPolicy(
            encoder=self._frozen_encoder,
            rssm=self._frozen_rssm,
            actor=self._frozen_actor,
            device=self.device,
            num_envs=1,
            act_dim=self._act_dim,
            obs_key=self.obs_key,
            pixel_obs=self._pixel_obs,
            eval_mode=True,
        )

    def get_explore_policy(self) -> DreamerPolicy:
        return DreamerPolicy(
            encoder=self._frozen_encoder,
            rssm=self._frozen_rssm,
            actor=self._frozen_actor,
            device=self.device,
            num_envs=self._num_envs,
            act_dim=self._act_dim,
            obs_key=self.obs_key,
            pixel_obs=self._pixel_obs,
            eval_mode=False,
        )

    # ------------------------------------------------------------------
    # step() — called by StatefulTrainer every env tick
    # ------------------------------------------------------------------

    def step(self, td: TensorDict) -> dict[str, float]:
        """Store transition and do a gradient update when the train-ratio fires.

        Algorithm owns:
          - Replay buffer storage
          - Decision of when to update (prefill + train_ratio)
          - The gradient step itself
        """
        self._buffer.add(td)
        frames = td.batch_size[0] if td.batch_size else 1
        self._collected_frames += frames

        # Prefill: collect random transitions before training starts.
        if self._collected_frames < self.prefill_steps:
            return {}

        if not self._buffer.ready(self.batch_length):
            return {}

        # train_ratio: env steps between gradient updates.
        self._update_credit += frames / self.train_ratio
        if self._update_credit < 1.0:
            return {}

        n_updates = int(self._update_credit)
        self._update_credit -= n_updates

        metrics: dict[str, float] = {}
        for _ in range(n_updates):
            metrics = self._update()

        return metrics

    # ------------------------------------------------------------------
    # _update() — one full gradient step
    # ------------------------------------------------------------------

    def _update(self) -> dict[str, float]:
        """Sample a (B, T) batch and compute world model + actor-critic loss.

        Mirrors Dreamer._cal_grad() from r2dreamer, with:
        - r2dreamer-specific rep_loss variants removed (dreamer decoder only)
        - LaProp → Adam, AGC → clip_grad_norm_
        - GradScaler removed for clarity
        """
        data = self._buffer.sample(self.batch_size, self.batch_length, self.device)
        B, T = data.batch_size

        # Extract tensors from TensorDict.
        obs = data[self.obs_key]  # (B, T, *obs)
        action = data["action"]  # (B, T, A)
        reward = data["next", "reward"].float()  # (B, T, 1) or (B, T)
        done = (
            data.get(("next", "done"), torch.zeros(B, T, 1, device=self.device)).float()
            | data.get(
                ("next", "terminated"), torch.zeros(B, T, 1, device=self.device)
            ).float()
        )
        # is_first / is_init for RSSM reset masking.
        is_first = data.get(
            "is_init",
            data.get(
                "is_first", torch.zeros(B, T, dtype=torch.bool, device=self.device)
            ),
        )
        is_first = is_first.bool().squeeze(-1)  # (B, T)

        reward = reward.squeeze(-1).unsqueeze(-1)  # ensure (B, T, 1)
        done = done.squeeze(-1).unsqueeze(-1)  # ensure (B, T, 1)

        if self._pixel_obs:
            obs = obs.float() / 255.0

        # === World model: posterior rollout ===
        embed = self.encoder(obs)  # (B, T, E)

        initial = self.rssm.initial(B)  # zeros; see note on latent caching below
        post_stoch, post_deter, post_logit = self.rssm.observe(
            embed, action, initial, is_first
        )
        # post_stoch: (B, T, S, K), post_deter: (B, T, D), post_logit: (B, T, S, K)

        _, prior_logit = self.rssm.prior(post_deter)  # (B, T, S, K)

        # KL losses (dynamics = prior learns posterior, rep = posterior learns prior).
        dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        # dyn_loss, rep_loss: (B, T)

        feat = self.rssm.get_feat(post_stoch, post_deter)  # (B, T, F)

        losses: dict[str, torch.Tensor] = {
            "dyn": dyn_loss.mean(),
            "rep": rep_loss.mean(),
        }
        loss_scales = {
            "dyn": self.scale_dyn,
            "rep": self.scale_rep,
            "rew": self.scale_rew,
            "con": self.scale_con,
            "actor": self.scale_actor,
            "value": self.scale_value,
            "repval": self.scale_repval,
        }

        # === Decoder reconstruction (pixel obs only) ===
        if self._use_decoder:
            recon_dist = self.decoder(post_stoch, post_deter)
            losses["dec"] = -recon_dist.log_prob(obs).mean()
            loss_scales["dec"] = self.scale_dec

        # === Reward and continue prediction ===
        # reward head uses symexp-twohot; log_prob returns negative loss.
        losses["rew"] = -self.reward_head(feat).log_prob(reward).mean()
        cont_target = 1.0 - done.float()
        losses["con"] = -self.cont_head(feat).log_prob(cont_target).mean()

        # === Imagination rollout for actor-critic ===
        # Detach start states so actor-critic gradients don't touch world model.
        # (B*T, S, K), (B*T, D)
        start_stoch = post_stoch.reshape(-1, *post_stoch.shape[2:]).detach()
        start_deter = post_deter.reshape(-1, *post_deter.shape[2:]).detach()

        imag_feat, imag_action = self._imagine(
            (start_stoch, start_deter), self.imag_horizon + 1
        )
        # imag_feat: (B*T, H+1, F), imag_action: (B*T, H+1, A) — both detached

        # Predict reward, cont, value along imagined trajectory using frozen nets.
        imag_reward = self._frozen_reward(imag_feat).mode()  # (B*T, H+1, 1)
        imag_cont = self._frozen_cont(imag_feat).mean  # (B*T, H+1, 1)  P(not done)
        imag_value = self._frozen_value(imag_feat).mode()  # (B*T, H+1, 1)
        imag_slow = self._frozen_slow_value(imag_feat).mode()  # (B*T, H+1, 1)

        weight = torch.cumprod(imag_cont * self.discount, dim=1)  # (B*T, H+1, 1)

        ret = self._lambda_return(
            last=torch.zeros_like(imag_cont),
            term=1.0 - imag_cont,
            reward=imag_reward,
            value=imag_value,
            boot=imag_value,
            disc=self.discount,
            lamb=self.lam,
        )  # (B*T, H, 1)

        ret_offset, ret_scale = self._return_ema(ret)
        adv = (ret - imag_value[:, :-1]) / ret_scale  # (B*T, H, 1)

        # Actor loss: policy gradient weighted by advantages + entropy bonus.
        policy = self.actor(imag_feat)
        logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)  # (B*T, H, 1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)  # (B*T, H, 1)
        losses["actor"] = (
            weight[:, :-1].detach()
            * -(logpi * adv.detach() + self.act_entropy * entropy)
        ).mean()

        # Critic loss: regress to λ-returns, regularised by slow target.
        value_dist = self.value(imag_feat)
        ret_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)  # (B*T, H+1, 1)
        losses["value"] = (
            weight[:, :-1].detach()
            * (
                -value_dist.log_prob(ret_padded.detach())
                - value_dist.log_prob(imag_slow.detach())
            )[:, :-1].unsqueeze(-1)
        ).mean()

        # Replay-based critic loss — keeps world model gradients flowing into value.
        last = done.float().reshape(B * T, 1).unsqueeze(1).expand(-1, T, -1)[:, :T]
        feat_bT = self.rssm.get_feat(post_stoch, post_deter)  # (B, T, F)
        boot = ret[:, 0].reshape(B, T, 1)
        rep_value = self._frozen_value(feat_bT).mode()
        rep_slow = self._frozen_slow_value(feat_bT).mode()
        rep_ret = self._lambda_return(
            last=done.squeeze(-1).unsqueeze(-1).expand(B, T, 1),
            term=done.squeeze(-1).unsqueeze(-1).expand(B, T, 1),
            reward=reward,
            value=rep_value,
            boot=boot,
            disc=self.discount,
            lamb=self.lam,
        )
        rep_ret_padded = torch.cat([rep_ret, 0 * rep_ret[:, -1:]], 1)
        rep_weight = 1.0 - done.squeeze(-1).unsqueeze(-1).float()
        rep_value_dist = self.value(feat_bT)
        losses["repval"] = (
            rep_weight[:, :-1]
            * (
                -rep_value_dist.log_prob(rep_ret_padded.detach())
                - rep_value_dist.log_prob(rep_slow.detach())
            )[:, :-1].unsqueeze(-1)
        ).mean()

        total = sum(loss_scales[k] * v for k, v in losses.items())

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(
            [p for p in self.optimizer.param_groups[0]["params"] if p.grad is not None],
            self.max_grad_norm,
        )
        self.optimizer.step()

        self._grad_steps += 1
        self._update_slow_target()
        self._sync_frozen_nets()

        metrics = {
            "train/loss_total": total.detach().item(),
            "train/loss_dyn": losses["dyn"].detach().item(),
            "train/loss_rep": losses["rep"].detach().item(),
            "train/loss_rew": losses["rew"].detach().item(),
            "train/loss_actor": losses["actor"].detach().item(),
            "train/loss_value": losses["value"].detach().item(),
            "train/ret_mean": ret.mean().detach().item(),
            "train/ret_p5": self._return_ema.ema_vals[0].item(),
            "train/ret_p95": self._return_ema.ema_vals[1].item(),
            "train/grad_steps": float(self._grad_steps),
        }
        if "dec" in losses:
            metrics["train/loss_dec"] = losses["dec"].detach().item()
        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_slow_target(self) -> None:
        if self._slow_value_updates % self.slow_target_update == 0:
            frac = self.slow_target_fraction
            with torch.no_grad():
                for v, s in zip(self.value.parameters(), self._slow_value.parameters()):
                    s.data.copy_(frac * v.data + (1 - frac) * s.data)
        self._slow_value_updates += 1

    @torch.no_grad()
    def _imagine(
        self,
        start: tuple[torch.Tensor, torch.Tensor],
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll out the frozen actor for `horizon` steps in latent space.

        Returns:
            imag_feat:   (B, horizon, F) — latent features at each imagined step
            imag_action: (B, horizon, A) — actions sampled from the frozen actor
        """
        stoch, deter = start
        feats, actions = [], []
        for _ in range(horizon):
            feat = self._frozen_rssm.get_feat(stoch, deter)
            action = self._frozen_actor(feat).rsample()
            feats.append(feat)
            actions.append(action)
            stoch, deter = self._frozen_rssm.img_step(stoch, deter, action)
        return torch.stack(feats, 1), torch.stack(actions, 1)

    @staticmethod
    def _lambda_return(
        last: torch.Tensor,
        term: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        boot: torch.Tensor,
        disc: float,
        lamb: float,
    ) -> torch.Tensor:
        """Compute λ-returns over a sequence.

        λ=1  → discounted Monte-Carlo return.
        λ=0  → one-step TD return.
        All inputs: (B, T, 1).  Returns: (B, T-1, 1).
        """
        live = (1 - to_f32(term))[:, 1:] * disc
        cont = (1 - to_f32(last))[:, 1:] * lamb
        interm = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]
        out = [boot[:, -1]]
        for i in reversed(range(live.shape[1])):
            out.append(interm[:, i] + live[:, i] * cont[:, i] * out[-1])
        return torch.stack(list(reversed(out))[:-1], 1)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _get_training_state(self) -> TrainingState:
        state_dict = {
            "encoder": self.encoder.state_dict(),
            "rssm": self.rssm.state_dict(),
            "reward_head": self.reward_head.state_dict(),
            "cont_head": self.cont_head.state_dict(),
            "actor": self.actor.state_dict(),
            "value": self.value.state_dict(),
            "slow_value": self._slow_value.state_dict(),
        }
        if self._use_decoder:
            state_dict["decoder"] = self.decoder.state_dict()
        return TrainingState(
            step=0,
            policy_state_dict=state_dict,
            optimizer_state_dict=self.optimizer.state_dict(),
            extra={
                "collected_frames": self._collected_frames,
                "grad_steps": self._grad_steps,
            },
        )

    def _load_training_state(self, state: TrainingState) -> None:
        sd = state.policy_state_dict
        self.encoder.load_state_dict(sd["encoder"])
        self.rssm.load_state_dict(sd["rssm"])
        self.reward_head.load_state_dict(sd["reward_head"])
        self.cont_head.load_state_dict(sd["cont_head"])
        self.actor.load_state_dict(sd["actor"])
        self.value.load_state_dict(sd["value"])
        self._slow_value.load_state_dict(sd["slow_value"])
        if self._use_decoder and "decoder" in sd:
            self.decoder.load_state_dict(sd["decoder"])
        self.optimizer.load_state_dict(state.optimizer_state_dict)
        if state.extra:
            self._collected_frames = int(state.extra.get("collected_frames", 0))
            self._grad_steps = int(state.extra.get("grad_steps", 0))
        self._sync_frozen_nets()
