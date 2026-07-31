# DreamerV3

**Paper:** Hafner et al. (2025), [*Mastering Diverse Control Tasks through World Models*](https://www.nature.com/articles/s41586-025-08744-2).

Model-based RL that learns a **world model** in a compact latent space and derives both
actor and critic entirely from **imagined** rollouts. Scales across continuous and discrete control with a single hyperparameter set.

The core implementation is based on the [NM512/r2dreamer](https://github.com/NM512/r2dreamer)
codebase, adapted to the TorchRL + Hydra structure of this template.

This implementation also supports two decoder-free variants:
Morihira et al. (2026), [*R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation*](https://arxiv.org/abs/2603.18202) (Barlow Twins auxiliary loss) and
Deng et al. (2021), [*DreamerPro: Reconstruction-Free Model-Based Reinforcement Learning with Prototypical Representations*](https://arxiv.org/abs/2110.14565) (prototypical assignment via Sinkhorn-Knopp OT).
Select the variant via `algorithm=r2dreamer` or `algorithm=dreamerpro`.

## Key ideas

- **Recurrent State Space Model (RSSM).** A GRU-based world model maintains a hybrid
  latent state $(h_t, z_t)$: $h_t$ is a deterministic recurrent belief, $z_t$ is a
  discrete stochastic categorical variable. The prior predicts $z_t$ from $h_t$; the
  posterior refines $z_t$ using the current observation.
- **KL balancing.** The KL loss is split into a dynamics term (prior ↔ sg-posterior)
  and a representation term (posterior ↔ sg-prior), each weighted separately with a
  `kl_free` free-nats floor to avoid penalising early exploration.
- **Symlog + two-hot regression.** Reward and value targets are mapped through
  $\text{symlog}(x) = \text{sign}(x)\ln(|x|+1)$ and predicted as a categorical
  distribution over a fixed bucket grid, making critics robust to large reward scales
  without any normalisation.
- **λ-returns in imagination.** Actor-critic targets blend Monte-Carlo and TD via
  $\lambda$: $\lambda=1$ recovers full returns, $\lambda=0$ recovers 1-step TD.
- **Entropy-regularised actor.** An `act_entropy` bonus discourages premature
  commitment during imagination rollouts.

**Decoder-free variants** replace pixel reconstruction (`loss_scales.recon`) with
auxiliary self-supervised losses on the encoder latents:
**R2-Dreamer** uses a **Barlow Twins** cross-correlation loss between projected RSSM
features and encoder embeddings (no decoder, no data augmentation needed).
**DreamerPro** uses **SwAV-style prototypical assignment** with Sinkhorn-Knopp OT and
an EMA target encoder trained on randomly-translated observations.

## Pseudocode

1. Initialise RSSM, encoder, decoder/Barlow/prototype head, reward head,
   continuation head, actor, critic.
2. Initialise replay buffer $\mathcal{D}$ (ring buffer, stores raw transitions).

**For each environment step:**

3. Observe $o_t$; encode → posterior $z_t$; update recurrent state $h_t$.
4. Sample action $a_t \sim \pi(a \mid h_t, z_t)$ (actor with entropy bonus).
5. Store $(o_t, a_t, r_t, \text{done}_t)$ in $\mathcal{D}$.

**Every** $\lfloor B \cdot T / \rho \rfloor$ **environment steps** (train ratio $\rho=128$):

6. Sample a sequence batch $(B=16, T=64)$ from $\mathcal{D}$.
7. **World model update:** encode sequences → RSSM forward pass →
   compute KL + representation loss (reconstruction / Barlow / prototypical) + reward +
   continuation losses → update encoder + RSSM + heads.
8. **Actor-critic update:** unroll `imag_horizon=15` imagination steps from posterior
   states → compute λ-returns → update actor (entropy-regularised policy gradient) +
   critic (two-hot symexp regression).
9. Write posterior latents $(h_t, z_t)$ back to the replay buffer for future sampling.

## Implementation in this template

| Resource | Path |
|----------|------|
| Algorithm wrapper + policy | [`dreamer.py`](dreamer.py) |
| Base DreamerV3 model | [`model/dreamerv3.py`](model/dreamerv3.py) |
| R2Dreamer extension | [`model/r2dreamer.py`](model/r2dreamer.py) |
| DreamerPro extension | [`model/dreamerpro.py`](model/dreamerpro.py) |
| RSSM | [`rssm.py`](rssm.py) |
| Network modules | [`networks.py`](networks.py) |
| Sequence replay buffer | [`buffer.py`](buffer.py) |
| Utility functions | [`tools.py`](tools.py) |
| Algorithm config (DreamerV3) | [`configs/algorithm/dreamer.yaml`](../../../configs/algorithm/dreamer.yaml) |
| Algorithm config (R2-Dreamer) | [`configs/algorithm/r2dreamer.yaml`](../../../configs/algorithm/r2dreamer.yaml) |
| Algorithm config (DreamerPro) | [`configs/algorithm/dreamerpro.yaml`](../../../configs/algorithm/dreamerpro.yaml) |
| Model size presets | [`configs/algorithm/dreamer/`](../../../configs/algorithm/dreamer/) |
| Atari environment | [`configs/environment/atari100k.yaml`](../../../configs/environment/atari100k.yaml) |
| Experiment (DreamerV3 preprocessing lives here) | [`configs/experiment/dreamer/atari100k.yaml`](../../../configs/experiment/dreamer/atari100k.yaml) |

```shell
# Standard DreamerV3 (pixel reconstruction)
python src/train.py experiment=dreamer/atari100k

# Another game
python src/train.py experiment=dreamer/atari100k environment.task=Breakout

# R2-Dreamer (Barlow Twins, decoder-free)
python src/train.py experiment=dreamer/atari100k algorithm=r2dreamer

# DreamerPro (prototypical assignment, decoder-free)
python src/train.py experiment=dreamer/atari100k algorithm=dreamerpro
```

### Model size presets

Model width is configured via a separate Hydra config group that sets `model.*`
interpolation variables consumed by the RSSM, encoder, decoder, actor, and critic.

| Preset | `deter` | `hidden` / `units` | `depth` | `discrete` | ~Params |
|--------|---------|--------------------|---------|------------|---------|
| `12m`  | 2048    | 256                | 16      | 16         | ~12M    |
| `25m`  | 3072    | 512                | 32      | 32         | ~25M    |
| `50m`  | 4096    | 640                | 48      | 48         | ~50M    |
| `100m` | 6144    | 768                | 64      | 64         | ~100M   |
| `200m` | 8192    | 1024               | 64      | 64         | ~200M   |
| `400m` | 12288   | 1536               | 64      | 64         | ~400M   |

Experiments default to `200m`. Override with e.g. `+algorithm/dreamer=50m`.

### Mapping pseudocode → code

| Pseudocode step | Where in code |
|-----------------|---------------|
| RSSM prior + posterior | [`rssm.py`](rssm.py) → `RSSM.forward()` |
| World model losses (DreamerV3) | [`model/dreamerv3.py`](model/dreamerv3.py) → `DreamerV3._cal_grad()` |
| Representation loss (R2Dreamer) | [`model/r2dreamer.py`](model/r2dreamer.py) → `R2Dreamer._compute_rep_losses()` |
| Representation loss (DreamerPro) | [`model/dreamerpro.py`](model/dreamerpro.py) → `DreamerPro._compute_rep_losses()` |
| Imagination rollout | [`model/dreamerv3.py`](model/dreamerv3.py) → `DreamerV3._imagine()` |
| λ-return computation | [`model/dreamerv3.py`](model/dreamerv3.py) → `DreamerV3._lambda_return()` |
| Latent write-back | [`buffer.py`](buffer.py) → `Buffer.update()` |
| Sequence sampling | [`buffer.py`](buffer.py) → `Buffer.sample()` |
| RSSM state across steps | [`dreamer.py`](dreamer.py) → `DreamerPolicy.forward()` |
| Proportional update cadence | [`dreamer.py`](dreamer.py) → `DreamerAlgorithm.step()` |

### TorchRL integration notes

Three design decisions required custom adaptation due to TorchRL conventions:

1. **Proportional collection cadence.** `SyncDataCollector` collects
   `frames_per_batch = ⌊B·T / ρ⌋` transitions per iteration (default: 8 with
   $B=16, T=64, \rho=128$) and feeds them to `Buffer.add_transition()`. Update
   frequency is governed by `train_ratio` inside `DreamerAlgorithm.step()`, not by
   the collector batch size — the algorithm fires one gradient update per batch
   regardless of how many frames were collected.

2. **`is_first` key.** TorchRL's `InitTracker` emits `is_init`; the RSSM needs
   `is_first` to zero its hidden state at episode boundaries. A `RenameTransform`
   in the environment pipeline bridges the gap.

3. **Cross-episode sequence sampling.** Dreamer trains on fixed-length sequences
   $(T=64)$ that intentionally cross episode boundaries. Masking TorchRL's `truncated`
   and `done` keys and injecting a per-environment `episode` key allows
   `SliceSampler(traj_key="episode", end_key=None)` to sample contiguous blocks
   without respect for episode ends, matching the original ring-buffer semantics.

4. **Online sampling** (`buffer_config.online`, default on; official DreamerV3
   `replay.online: True`, absent from R2Dreamer). Each fresh, non-overlapping
   $(T{+}1)$-step segment of experience is queued and served at the front of the
   next batch before uniform sampling fills the rest, so every collected
   transition is trained on exactly once as soon as it exists. Uniform
   `SliceSampler` slices near the write cursor cannot cover the newest steps,
   so without this queue fresh data is systematically under-sampled.


## Experimental results

**Evaluation protocol.** Following the official DreamerV3 code (`run.steps: 1.1e5`),
the Atari100k experiment configs train for 110k agent steps — 10 % past the
benchmark budget of 100k steps (400k game frames) stated in the paper. The
end-of-run summary metrics stay paper-comparable: `eval/score_last` and
`eval/score_mean_last10pct` are cut off at `algorithm.benchmark_frames`
(default 400k game frames), i.e. the last episode within the budget and the mean
over episodes in its final 10 % (360k–400k frames). Episodes past the budget
appear only on the `episode/score` curve, for debugging and run-to-run comparison. Some runs below don"t have any eval metrics yet so they are not listed, replace them with new ones in the future.

**Live W&B table (canonical):** [LatentLab/torchrl-hydra-template — Table](https://wandb.ai/LatentLab/torchrl-hydra-template/table)

| Run | Environment | Config | Seed | Frames | Eval return | Notes |
|-----|-------------|--------|------|--------|-------------|-------|
| [dreamer_hero_atari100k_200m_2026-07-09_10-35-35](https://wandb.ai/LatentLab/torchrl-hydra-template/runs/8m6v58pk) | ALE/Hero-v5 | `experiment=dreamer/atari100k` | 42 | 110,000 | 10,253.3 | — |
| [dreamer_hero_atari100k_200m_2026-07-09_12-18-11](https://wandb.ai/LatentLab/torchrl-hydra-template/runs/51h2g461) | ALE/Hero-v5 | `experiment=dreamer/atari100k` | 45 | 110,000 | 12,391.2 | — |
| [dreamer_hero_atari100k_200m_2026-07-09_12-18-11](https://wandb.ai/LatentLab/torchrl-hydra-template/runs/p9fwwjdd) | ALE/Hero-v5 | `experiment=dreamer/atari100k` | 44 | 110,000 | 6,331.2 | — |
| [dreamer_hero_atari100k_200m_2026-07-09_12-18-11](https://wandb.ai/LatentLab/torchrl-hydra-template/runs/udxi7rsc) | ALE/Hero-v5 | `experiment=dreamer/atari100k` | 43 | 110,000 | 6,920.5 | — |
| [dreamerpro_hero_atari100k_200m_2026-07-09_11-16-03](https://wandb.ai/LatentLab/torchrl-hydra-template/runs/7ydjbg18) | ALE/Hero-v5 | `experiment=dreamer/atari100k algorithm=dreamerpro` | 44 | 110,000 | 2,983.5 | DreamerPro |
| [r2dreamer_hero_atari100k_200m_2026-07-08_14-10-27](https://wandb.ai/LatentLab/torchrl-hydra-template/runs/4sz2koi1) | ALE/Hero-v5 | `experiment=dreamer/atari100k algorithm=r2dreamer` | 42 | 100,000 | 4,598.8 | R2Dreamer |
