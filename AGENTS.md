# Agent instructions for torchrl-hydra-template

## Project overview

A modular reinforcement learning research template built on
[TorchRL](https://github.com/pytorch/rl) and
[Hydra](https://github.com/facebookresearch/hydra). Three composable components —
**Environment**, **Algorithm**, **Trainer** — are wired together by `src/train.py`.

Implemented experiments:

| Algorithm | Environment    | Experiment config             |
|-----------|----------------|-------------------------------|
| DQN       | CartPole-v1    | `experiment=dqn/cartpole`     |
| DQN       | ALE/Pong-v5    | `experiment=dqn/pong`         |
| DDPG      | HalfCheetah-v4 | `experiment=ddpg/halfcheetah` |
| A2C       | HalfCheetah-v4 | `experiment=a2c/halfcheetah`  |
| TD-MPC2   | dmc cheetah-run | `experiment=tdmpc2/cheetah_run` |
| DreamerV3 | ALE/Hero-v5<br>(Atari100k) | `experiment=dreamer/hero` |

Other algorithms will follow.

## Design principles

1. **Readable algorithm code.** Each algorithm file should read close to the
   pseudocode from the paper. `step()` is short and corresponds to the update
   equations. Long config-shuffling and framework glue belong elsewhere.
2. **Hard separation of responsibilities.**
   - **Algorithm** owns everything that affects the learning curve: network, replay
     buffer, loss, optimiser, exploration, target-net schedule, and the collector
     config (`frames_per_batch`, `init_random_frames`, ...). All hyperparameters live
     as keyword arguments on `__init__`.
   - **Trainer** owns the loop. It creates the collector from
     `algorithm.get_collector_config()`, calls `algorithm.step(batch)`, manages the
     device, fires callbacks, and checkpoints.  Nothing on the trainer config affects
     reward or sample efficiency.
   - **Environment** is a fixed task definition: env name + transform list. It does
     not know about the algorithm.
3. **One source of truth per concern.** HP defaults live in the algorithm's
   `__init__` (with type hints + docstrings). YAML mirrors them for overrides.
4. **Callable factories via Hydra.** Design choices that are `Callable`s (replay
   buffer, network) are configured in `configs/algorithm/*.yaml` with `_partial_`
   and nested `_target_` nodes. **`src/train.py` and `src/eval.py` build the
   algorithm with `hydra.utils.instantiate(cfg.algorithm, device=None)`** so those
   nested configs become real callables. Plain `OmegaConf.to_container` + `**kwargs`
   would pass dicts instead of partials.

   *Exception (TD-MPC2 precedent):* when subnetworks are architecturally coupled
   (shared latent dims, checkpoint-compatibility pins parameter names), factories
   add failure modes without research payoff. Then architecture knobs are plain
   scalar kwargs and the model/buffer are built inside `setup()` — document the
   deviation in the algorithm README.
5. **Reusable building blocks live in `src/components/`.** Anything not specific
   to one algorithm (e.g. two-hot discrete-regression math, SimNorm/NormedLinear/
   vmapped `Ensemble` layers, `RunningScale`) goes there so other algorithms can
   import it. Code adapted from external repos carries a source-attribution
   header comment (upstream URL + file path + license).

## Algorithm constructor pattern

```python
class DQNAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        device: torch.device | None = None,
        *,
        # Design choices: factories injected as Callables
        replay_buffer: Callable[[], ReplayBuffer] = lambda: TensorDictReplayBuffer(...),
        # Q-net factory; setup() passes (obs_shape, num_actions) — see below.
        network: Callable[[tuple[int, ...], int], nn.Module] = functools.partial(
            make_mlp_q_net, num_cells=[120, 84], activation_class=nn.ReLU
        ),
        # Observation tensordict key (e.g. "observation" for vector obs, "pixels" for image obs).
        obs_key: str = "observation",
        # Scalar HPs
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        batch_size: int = 128,
        max_grad_norm: float = 10.0,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        annealing_frames: int = 250_000,
        frames_per_batch: int = 1_000,
        init_random_frames: int = 10_000,
        max_frames_per_traj: int = -1,
        num_updates: int = 100,
        hard_update_freq: int = 50,
    ) -> None:
        super().__init__(device)
        # ... store kwargs verbatim ...
```

Rules:
- `*` makes every HP keyword-only.
- `BaseAlgorithm.__init__(device)` — **no `cfg` parameter**. Algorithms read env
  specs from `make_env()` inside `setup()`.
- `replay_buffer` is a **no-arg** factory returning a `ReplayBuffer`.
- `network` (DQN) is a factory called as **`network(obs_shape, num_actions)`** —
  positional `obs_shape` is the raw observation shape tuple (e.g. `(4,)` for
  CartPole, `(4, 84, 84)` for stacked Atari frames) and `num_actions` is the
  discrete action count. For DDPG, `actor_network` and `value_network` use the
  same call signature with the continuous action vector size. Use the helpers
  in `src/networks.py`:
    - `make_mlp_q_net(obs_shape, num_actions, *, num_cells, activation_class)` —
      flattens `obs_shape` into a torchrl `MLP`. Default for vector observations.
    - `NatureDQN(obs_shape, num_actions, *, ...)` — Mnih et al. 2015 ConvNet+MLP
      head. Default for image observations.
    - `make_mlp_ddpg_actor(obs_shape, action_dim, *, num_cells, activation_class)` —
      MLP body for a deterministic actor (DDPG); no final tanh, the algorithm
      wraps it in `TanhModule` to rescale to the action spec.
    - `make_mlp_ddpg_critic(obs_shape, action_dim, *, num_cells, activation_class)` —
      state-action critic; takes `[obs, action]` concatenated by `ValueOperator`.
    - `make_mlp_a2c_actor(obs_shape, action_dim, *, num_cells, activation_class)` —
      MLP body for an A2C stochastic actor; outputs `2 * action_dim` features
      that `NormalParamExtractor` splits into `loc` and `scale` for TanhNormal.
    - `make_mlp_a2c_value(obs_shape, action_dim, *, num_cells, activation_class)` —
      state-value (V(s)) critic; `action_dim` is unused but kept for signature
      parity with the actor factory.
  All keep everything after the two positional args **kwarg-only**, so a Hydra
  `_partial_` config can pre-bind kwargs without colliding with `setup()`'s call.
- `obs_key` selects which tensordict key the observation comes from. Vector
  envs (CartPole) use `"observation"`; pixel envs (Atari with `from_pixels=True`)
  use `"pixels"`. The key is forwarded to `QValueActor.in_keys` and used to read
  the spec for the network factory.
- **Activation class in YAML:** `torchrl.modules.MLP` expects `activation_class`
  to be a **type** (it instantiates internally). In Hydra YAML, **`_target_:
  torch.nn.ReLU` nests an instantiation** and produces a module instance, which
  breaks `MLP`. Use **`hydra.utils.get_class`** instead:

  ```yaml
  activation_class:
    _target_: hydra.utils.get_class
    path: torch.nn.ReLU
  ```

- Scalar HPs are plain kwargs and **do** appear in YAML.

## `step(batch)` shape

```python
def step(self, batch: TensorDict) -> dict[str, float]:
    # 1. Always — anneal exploration, store transitions
    batch = batch.reshape(-1)
    self.greedy_module.step(batch.numel())
    self.replay_buffer.extend(batch)
    self._collected_frames += batch.numel()

    # 2. Warm-up gate
    if self._collected_frames < self.init_random_frames:
        return {"train/epsilon": float(self.greedy_module.eps)}

    # 3. Optimisation loop — sample, loss, backward, optimiser, target update
    for j in range(self.num_updates):
        sample = self.replay_buffer.sample(self.batch_size).to(self.device)
        loss = self.loss_module(sample)["loss"]
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_actor.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.target_updater.step()

    return {"train/q_loss": ..., "train/epsilon": ...}
```

The trainer never touches the replay buffer, target network or epsilon — those are
algorithm internals. Per-batch metrics (`train/episode_reward`,
`train/episode_length`, `train/q_values`) and timing (`time/collect`,
`time/step`, `time/speed`) are computed by `StepTrainer` from the collector
batch and merged into the algorithm's metrics dict at logging boundaries.
This mirrors the torchrl SOTA DQN reference and keeps batch-level bookkeeping
out of the algorithm.

### On-policy variant (A2C)

A2C drops three of those internals entirely: no long-term replay buffer, no
target networks, no warm-up. Each `step(batch)` runs `GAE` on the rollout
under `no_grad`, refills a one-shot buffer with `SamplerWithoutReplacement`,
and does one epoch of mini-batch updates with `A2CLoss`. The buffer in
`a2c.py` is built directly in `setup()` (not exposed as a `_partial_`
factory) because its size is locked to `frames_per_batch / mini_batch_size`
— it's an implementation detail of the on-policy schedule, not a research
choice. `get_collector_config()` returns `init_random_frames=0` since the
stochastic actor explores from frame zero.

## Instantiation in `src/train.py` / `src/eval.py`

```python
from hydra.utils import instantiate, get_class
from omegaconf import OmegaConf

algorithm = instantiate(cfg.algorithm, device=None)  # recursive; resolves _partial_ factories

env_kwargs = {k: v for k, v in OmegaConf.to_container(cfg.environment, resolve=True).items()
              if k != "_target_"}
environment = Environment(**env_kwargs)

TrainerClass = get_class(cfg.trainer._target_)
trainer = TrainerClass(cfg=cfg, algorithm=algorithm, environment=environment)
```

The **environment** is still unpacked from a flat dict. The **algorithm** must use
`instantiate` whenever its YAML contains nested `_target_` / `_partial_` nodes
(e.g. `replay_buffer`, `network`).

YAML values override Python defaults where present; absent keys fall back to
constructor defaults.

## Environment

`Environment.__init__` accepts:
- `name`: gymnasium env id (e.g. `"CartPole-v1"`, `"ALE/Pong-v5"`), or the
  dm_control domain name (e.g. `"cheetah"`) when `backend: dm_control`.
- `transforms`: list of `_target_`-keyed dicts, each instantiated as a
  `torchrl.envs.transforms` object and composed on top of the base env.
  Always include `StepCounter` explicitly. Add `RewardSum` if you want
  `train/episode_reward` in the trainer metrics — it populates the
  `("next", "episode_reward")` key the trainer reads.
- `gym_kwargs`: optional dict forwarded straight to `GymEnv` (e.g.
  `{"frame_skip": 4, "from_pixels": true, "pixels_only": false,
  "categorical_action_encoding": true}` for Atari).
- `gym_backend`: optional backend name (`"gymnasium"`); if set, the GymEnv
  construction is wrapped in `set_gym_backend(...)`.
- `backend`: `"gymnasium"` (default) or `"dm_control"`.
- `task`: dm_control task name (e.g. `"run"`); required for
  `backend: dm_control`, ignored otherwise.

```yaml
# configs/environment/cartpole.yaml
name: CartPole-v1
transforms:
  - _target_: torchrl.envs.transforms.StepCounter
  - _target_: torchrl.envs.transforms.RewardSum
```

```yaml
# configs/environment/pong_train.yaml — pixel-based Atari env
name: ALE/Pong-v5
gym_backend: gymnasium
gym_kwargs:
  frame_skip: 4
  from_pixels: true
  pixels_only: false
  categorical_action_encoding: true
transforms:
  - _target_: torchrl.envs.NoopResetEnv
    noops: 30
    random: true
  # ... (see configs/environment/pong_train.yaml for the full SOTA stack)
```

```yaml
# configs/environment/dmc_cheetah_run.yaml — dm_control env
backend: dm_control
name: cheetah
task: run
transforms:
  - _target_: torchrl.envs.transforms.FrameSkipTransform  # action repeat 2
    frame_skip: 2
  - _target_: torchrl.envs.transforms.CatTensors          # flatten dict obs -> "observation"
    in_keys: [position, velocity]
    out_key: observation
  # ... (DoubleToFloat, InitTracker, StepCounter, RewardSum)
```

The factory in `src/environments/factory.py` supports **gymnasium** (`GymEnv`)
and **dm_control** (`DMControlEnv`). dm_control observations are dicts; use
`CatTensors` to flatten them into a single `observation` key (alphabetical
`in_keys` order matches the upstream TD-MPC2 concatenation). For >1 `num_envs`,
workers run on CPU (`ParallelEnv` with `mp_start_method="spawn"`).

### Separate evaluation env

`BaseTrainer` accepts an optional `eval_environment: Environment | None` arg.
When set, `evaluate()` builds its eval env from it; otherwise it falls back to
`environment`. Wire it in via Hydra package overrides:

```yaml
# configs/train.yaml (and eval.yaml)
defaults:
  - environment: ???
  - environment@eval_environment: null   # default: no separate eval env

# configs/experiment/dqn/pong.yaml
defaults:
  - override /environment: pong_train
  - override /environment@eval_environment: pong_eval
```

`src/train.py` and `src/eval.py` build the eval `Environment` via
`cfg.get("eval_environment")` and pass it to the trainer constructor.

Use this when training-time and evaluation-time observations should differ
(e.g. Atari, where `EndOfLifeTransform` and `SignTransform` are train-only and
`VecNorm` is dropped at eval because its running stats are not checkpointed).

## Trainer

`StepTrainer` is the only trainer.  It:
- creates a `torchrl.collectors.Collector` from `algorithm.get_collector_config()`
  and `cfg.trainer.total_frames`;
- iterates the collector, calls `algorithm.step(batch)`, and fires
  `ON_STEP_END` callbacks at logging boundaries;
- delegates device resolution to `src/utils/device.py`.

`BaseTrainer` owns env lifecycle, `evaluate(num_episodes)` (greedy rollout), and
checkpoint orchestration. Checkpointing is **off by default** (`checkpoint.enabled:
false` in `configs/train.yaml`); enable it with
`checkpoint.enabled=true` (and optionally tune `save_every_n_steps` /
`save_last`). `checkpoint.resume_from` still works when checkpointing is disabled.

## File map

```
src/
  train.py                  — entry point; instantiate(cfg.algorithm); environment **kwargs
  eval.py                   — evaluation entry point; same algorithm instantiation
  networks.py               — network factories: make_mlp_q_net, NatureDQN,
                              make_mlp_ddpg_actor, make_mlp_ddpg_critic,
                              make_mlp_a2c_actor, make_mlp_a2c_value
  algorithms/
    base.py                 — BaseAlgorithm ABC; TrainingState and CollectorConfig dataclasses
    dqn/
      dqn.py                — DQNAlgorithm; replay/network factories (defaults + setup contract)
      README.md             — theory, pseudocode, W&B benchmark table
    ddpg/
      ddpg.py               — DDPGAlgorithm; actor/critic/replay/noise factories
      README.md             — theory, pseudocode, W&B benchmark table
    a2c/
      a2c.py                — A2CAlgorithm; on-policy actor/critic with GAE + A2CLoss
      README.md             — theory, pseudocode, W&B benchmark table
    tdmpc2/
      tdmpc2.py             — TDMPC2Algorithm; world-model learning + slice replay buffer
      world_model.py        — WorldModel (upstream-checkpoint compatible) + api_model_conversion
      planner.py            — MPPIPlanner (latent-space planning, warm-started)
      policy.py             — TensorDictModule wrapper (reads obs + is_init)
      README.md             — theory, pseudocode, W&B benchmark table
  components/               — reusable building blocks (per-file attribution headers)
    math.py                 — symlog/symexp (canonical), two-hot discrete regression, squashed-Gaussian helpers (from nicklashansen/tdmpc2, MIT)
    layers.py               — SimNorm, NormedLinear, vmapped Ensemble, LayerNorm-Mish mlp (from nicklashansen/tdmpc2, MIT)
    scale.py                — RunningScale (trimmed-percentile value normalizer) (from nicklashansen/tdmpc2, MIT)
    distributions.py        — Dreamer distribution factories: OneHotDist, TwoHot, SymlogDist, ... (from NM512/r2dreamer)
    ema.py                  — polyak_update (in-place EMA of parameters)
    optim/                  — LaProp optimizer (Z-T-WANG/LaProp-Optimizer, MIT), adaptive gradient clipping
  environments/
    environment.py          — Environment wrapper (holds factory kwargs, exposes make_env)
    factory.py              — make_env: gymnasium/dm_control + transforms list + gym_kwargs/gym_backend
  trainers/
    BaseTrainer.py          — BaseTrainer ABC, TrainerEvent, Callback protocol, fire_callbacks
    StepTrainer.py          — StepTrainer (Collector-driven loop)
  callbacks/                — ProgressCallback, CheckpointCallback, WandBLogger, TensorBoardLogger
  utils/                    — device resolution, seeding, callback builders
configs/
  algorithm/dqn.yaml        — DQN HPs (CartPole defaults); _partial_ replay_buffer + network
  algorithm/dqn_atari.yaml  — DQN HPs (Atari/NatureDQN defaults; pixel obs)
  algorithm/ddpg.yaml       — DDPG HPs (HalfCheetah defaults); _partial_ actor/critic/noise
  algorithm/a2c.yaml        — A2C HPs (HalfCheetah/MuJoCo defaults); _partial_ actor/value
  algorithm/tdmpc2.yaml     — TD-MPC2 HPs (model_size=5 preset; scalar knobs, no _partial_)
  environment/cartpole.yaml — env kwargs (name, transforms)
  environment/pong_train.yaml — Atari Pong env (training transforms incl. EndOfLife + Sign + VecNorm)
  environment/pong_eval.yaml  — Atari Pong env (eval transforms; drops EndOfLife + Sign + VecNorm)
  environment/halfcheetah.yaml — HalfCheetah-v4 (DoubleToFloat + InitTracker)
  environment/dmc_cheetah_run.yaml — dm_control cheetah-run (FrameSkip 2 + CatTensors)
  experiment/dqn/cartpole.yaml — composed CartPole experiment
  experiment/dqn/pong.yaml     — composed Atari Pong experiment
  experiment/ddpg/halfcheetah.yaml — composed DDPG HalfCheetah experiment
  experiment/a2c/halfcheetah.yaml — composed A2C HalfCheetah experiment
  experiment/tdmpc2/cheetah_run.yaml — composed TD-MPC2 DMC cheetah-run experiment
  logger/{wandb,tensorboard}.yaml
  paths/default.yaml
  train.yaml, eval.yaml
tests/
  test_smoke.py             — smoke tests: DQN (CartPole, Pong), DDPG, A2C, TD-MPC2
```

## Documentation

Algorithm and model READMEs use **GitHub-flavoured markdown math**: inline formulas
with `$...$`, display formulas with `$$...$$`. Do **not** use `\(...\)` or
`\[...\]` — those delimiters are not rendered on GitHub.

Example: `$Q(s, a; \theta)$`, `$\theta_{\text{target}}$`.

## Adding a new algorithm

1. Create `src/algorithms/my_algo/my_algo.py` with an `__init__.py` re-export
   following the kwargs pattern above. Use `Callable` factories for design choices
   (inline lambdas, `functools.partial`, or small helpers). Document the **call
   signature** each factory must satisfy (e.g. `network(obs_shape, num_actions)`).
2. Implement `setup(make_env)`, `step(batch) -> dict`, `get_policy()`,
   `get_explore_policy()`, `get_collector_config()`,
   `_get_training_state()`, `_load_training_state()`.
3. Create `configs/algorithm/my_algo.yaml` with `_target_`, scalar HPs, and any
   `_partial_` / nested `_target_` blocks for factories. Use `instantiate`-
   compatible patterns (see DQN: replay buffer + partial `MLP`).
4. Create `configs/experiment/my_algo/<env>.yaml` composing your algo + env.
5. Add `src/algorithms/my_algo/README.md` with theory, pseudocode, implementation
   mapping, and an experimental-results table (link to
   [W&B project table](https://wandb.ai/LatentLab/torchrl-hydra-template/table)).
   Tag benchmark W&B runs with `template`; register the algorithm's target prefix
   in `ALGO_TARGET_PREFIXES` in `scripts/update_algo_results.py`, then refresh the
   table via `python scripts/update_algo_results.py`. Use `$...$` for inline math (see
   [Documentation](#documentation)).
6. **Update `README.md` and `AGENTS.md`.**
7. Add a smoke test in `tests/test_smoke.py`.

## What not to do

- Do not place learning-affecting knobs on `trainer:` or `environment:` configs.
- Do not create `XxxConfig` dataclasses.
- Do not add `cfg: DictConfig` to `BaseAlgorithm` or pass `cfg=cfg` to algorithms.
- Do not pass `cfg.environment` directly to `Environment()` — unpack as `**kwargs`.
- Do not add `OmegaConf` imports to `base.py`.

## Running

```shell
python src/train.py experiment=dqn/cartpole
python src/train.py experiment=dqn/cartpole algorithm.lr=1e-3
python src/train.py experiment=dqn/cartpole 'logger=[wandb]'  # experiments default to wandb; plain CLI defaults to tensorboard
python src/train.py experiment=dqn/pong            # Atari Pong (40M frames, GPU)
python src/train.py experiment=ddpg/halfcheetah    # DDPG continuous control (1M frames)
python src/train.py experiment=a2c/halfcheetah     # A2C on-policy continuous control (1M frames)
python src/train.py experiment=tdmpc2/cheetah_run  # TD-MPC2 model-based control (1M frames, GPU)
python scripts/update_algo_results.py              # refresh algo README benchmark tables (W&B tag: template)
pytest tests/test_smoke.py -v

# Evaluate an official TD-MPC2 checkpoint (see src/algorithms/tdmpc2/README.md):
python src/eval.py algorithm=tdmpc2 environment=dmc_cheetah_run \
  checkpoint.resume_from=$PWD/checkpoints/cheetah-run-1.pt trainer.accelerator=gpu
```
