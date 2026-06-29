# DreamerV3

**Paper:** Hafner et al. (2025), [*Mastering Diverse Control Tasks through World Models*](https://www.nature.com/articles/s41586-025-08744-2).

Model-based RL that learns a **world model** in a compact latent space and derives
both actor and critic entirely from **imagined** rollouts — never from real environment
rewards directly. Scales across continuous and discrete tasks using a single set of
hyperparameters.

This implementation is based on [R2Dreamer](https://github.com/NM512/r2dreamer), which
extends DreamerV3 with Barlow Twins self-supervised representation learning
(decoder-free mode). Standard DreamerV3 reconstruction loss is available via
`rep_loss: dreamer`.

## Key ideas

- **Recurrent State Space Model (RSSM).** A GRU-based world model maintains a hybrid
  latent state $(h_t, z_t)$ where $h_t$ is a deterministic recurrent belief and $z_t$
  is a discrete stochastic categorical variable. The prior predicts $z_t$ from $h_t$;
  the posterior refines $z_t$ using the current observation.
- **KL balancing.** The RSSM KL loss is split into a representation term (posterior ↔
  prior) and a dynamics term (prior ↔ posterior), scaled with a `kl_free` free-nats
  threshold so early exploration is not penalised.
- **Barlow Twins mode** (`rep_loss: r2dreamer`). Decoder-free variant: the encoder is
  jointly trained with a Barlow Twins loss between two augmented views, replacing
  reconstruction. Uses `loss_scales.barlow` to weight the auxiliary term.
- Add more

## Pseudocode

1. Initialise RSSM, encoder, decoder/Barlow head, reward head, continuation head,
   actor, critic.
2. Initialise replay buffer $\mathcal{D}$ (ring buffer, stores raw transitions).

**For each environment step:**

3. Observe $o_t$; encode → posterior $z_t$; update recurrent state $h_t$.
4. Sample action $a_t \sim \pi(a \mid h_t, z_t)$ (actor with entropy bonus).
5. Store $(o_t, a_t, r_t, \text{done}_t)$ in $\mathcal{D}$.

**Every** $\lfloor B \cdot T / \rho \rfloor$ **environment steps** (train ratio $\rho=128$):

6. Sample a sequence batch $(B=16, T=64)$ from $\mathcal{D}$.
7. **World model update:** encode sequences → RSSM forward pass →
   compute KL + reconstruction (or Barlow Twins) + reward + continuation losses →
   update encoder + RSSM + heads.
8. **Actor-critic update:** unroll `imag_horizon=15` imagination steps from posterior
   states → compute λ-returns → update actor (entropy-regularised policy gradient) +
   critic (two-hot symexp regression).
9. Write posterior latents $(h_t, z_t)$ back to the replay buffer for future sampling.

## Implementation in this template

| Resource | Path |
|----------|------|
| Algorithm wrapper | [`dreamer.py`](dreamer.py) |
| World model + update logic | [`dreamer_model.py`](dreamer_model.py) |
| Sequence replay buffer | [`buffer.py`](buffer.py) |
| Network modules | [`networks.py`](networks.py) |
| Algorithm HPs | [`configs/algorithm/dreamer.yaml`](../../../configs/algorithm/dreamer.yaml) |
| Model size presets | [`configs/algorithm/dreamer/`](../../../configs/algorithm/dreamer/) |
| Atari environment | [`configs/environment/atari_dreamer.yaml`](../../../configs/environment/atari_dreamer.yaml) |
| Breakout experiment | [`configs/experiment/dreamer/breakout.yaml`](../../../configs/experiment/dreamer/breakout.yaml) |

```shell
python src/train.py experiment=dreamer/breakout
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
| RSSM prior + posterior | `dreamer_model.py` → `RSSM.forward()` |
| World model losses | `dreamer_model.py` → `Dreamer._cal_grad()` |
| Imagination rollout | `dreamer_model.py` → `Dreamer._imagine_ahead()` |
| Actor-critic update | `dreamer_model.py` → `Dreamer._update_actor_critic()` |
| Latent write-back | `buffer.py` → `Buffer.update()` |
| Sequence sampling | `buffer.py` → `Buffer.sample()` |
| RSSM state across steps | `dreamer.py` → `DreamerPolicy.forward()` |
| Proportional update cadence | `dreamer.py` → `DreamerAlgorithm.step()` |

### TorchRL integration notes

Three design decisions required custom adaptation due to TorchRL conventions:

1. **Single-step collection (`frames_per_batch=1`).** `SyncDataCollector` collects one
   transition at a time, feeding it directly to `Buffer.add_transition()`. The update
   cadence is controlled by `train_ratio` inside `DreamerAlgorithm.step()` rather than
   by the collector batch size.

2. **`is_first` key.** TorchRL's `InitTracker` emits `is_init`; the RSSM needs
   `is_first` to zero its hidden state at episode boundaries. A `RenameTransform`
   in the environment pipeline bridges the gap.

3. **Cross-episode sequence sampling.** Dreamer trains on fixed-length sequences
   $(T=64)$ that intentionally cross episode boundaries. Masking TorchRL's `truncated`
   and `done` keys and injecting a per-environment `episode` key allows
   `SliceSampler(traj_key="episode", end_key=None)` to sample contiguous blocks
   without respect for episode ends, matching the original ring-buffer semantics.


## Experimental results

**Live W&B table (canonical):** [LatentLab/torchrl-hydra-template — Table](https://wandb.ai/LatentLab/torchrl-hydra-template/table)

| Run | Environment | Config | Seed | Agent frames | Eval return | Notes |
|-----|-------------|--------|------|--------------|-------------|-------|
| — | ALE/Breakout-v5 | `experiment=dreamer/breakout` | — | — | — | in progress |
