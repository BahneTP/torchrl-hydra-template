# Pixel-Control Algorithms: DER, SPR, SR-SPR, BBF, SAC-BBF

These modules port DER, SPR, SR-SPR, BBF, and SAC-BBF style pixel-control
agents from `BBF-pytorch` into the TorchRL/Hydra framework.

Implemented algorithms:

| Algorithm | Config |
|-----------|--------|
| DER | `algorithm=der` |
| SPR | `algorithm=spr` |
| SR-SPR | `algorithm=sr_spr` |
| BBF | `algorithm=bbf` |
| SAC-BBF | `algorithm=sac_bbf` |

The implementation keeps the learning core close to `BBF-pytorch`: C51
distributional targets, n-step returns, deterministic prioritized replay,
NoisyNet layers, dueling heads, SPR rollouts, and BBF reset logic are local to
the algorithm modules. The shared framework adapter translates TorchRL
`TensorDict` batches into the NumPy replay format used by those agents.

## Atari 100K Environment

Experiments use:

- `configs/environment/atari100k_train.yaml`
- `configs/environment/atari100k_eval.yaml`

The environment emits single 84x84 grayscale frames. Frame stacking stays inside
the algorithm/replay code, matching `BBF-pytorch`.

## Experiments

```shell
python src/train.py experiment=atari100k/der/jamesbond
python src/train.py experiment=atari100k/spr/jamesbond
python src/train.py experiment=atari100k/sr_spr/jamesbond
python src/train.py experiment=atari100k/bbf/jamesbond
python src/train.py experiment=atari100k/sac_bbf/jamesbond
```

## Known Framework Differences From `BBF-pytorch`

- Collection/evaluation are driven by the framework `StepTrainer` and TorchRL
  collector instead of the standalone `Runner`.
- Policies read and write TorchRL `TensorDict`s.
- Atari preprocessing is configured through Hydra environment files. Custom
  max-and-skip/life-loss semantics are exposed through a project TorchRL
  transform in the same transform list as the standard TorchRL transforms.
- Logging and checkpointing use framework callbacks.

The algorithmic parts that most directly affect learning are kept hard-ported.
