from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from src.environments.atari_wrappers import EpisodicLifeEnv, MaxAndSkipEnv


class _FakeAle:
    def __init__(self, env: "_FakeAtariEnv") -> None:
        self.env = env

    def lives(self) -> int:
        return self.env.lives


class _FakeAtariEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 255, shape=(2, 2, 1), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, life_loss_step: int = 2) -> None:
        super().__init__()
        self.life_loss_step = life_loss_step
        self.steps = 0
        self.lives = 3
        self.ale = _FakeAle(self)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.steps = 0
        self.lives = 3
        return self._observation(), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.steps += 1
        if self.steps == self.life_loss_step:
            self.lives -= 1
        return self._observation(), 1.0, False, False, {"steps": self.steps}

    def _observation(self) -> np.ndarray:
        return np.full(self.observation_space.shape, self.steps, dtype=np.uint8)


def test_max_and_skip_stops_on_life_loss() -> None:
    base = _FakeAtariEnv(life_loss_step=2)
    env = MaxAndSkipEnv(base, skip=4, stop_on_life_loss=True)
    env.reset()

    observation, reward, terminated, truncated, _ = env.step(1)

    assert base.steps == 2
    assert reward == 2.0
    assert not terminated
    assert not truncated
    np.testing.assert_array_equal(observation, np.full((2, 2, 1), 2, dtype=np.uint8))


def test_episodic_life_soft_reset_does_not_step() -> None:
    base = _FakeAtariEnv(life_loss_step=2)
    env = EpisodicLifeEnv(
        MaxAndSkipEnv(base, skip=4, stop_on_life_loss=True),
        advance_on_life_loss=False,
    )
    env.reset()

    terminal_observation, reward, terminated, truncated, terminal_info = env.step(1)

    assert reward == 2.0
    assert terminated
    assert not truncated
    assert base.steps == 2

    reset_observation, reset_info = env.reset()

    assert base.steps == 2
    np.testing.assert_array_equal(reset_observation, terminal_observation)
    assert reset_info == terminal_info


def test_default_life_loss_behavior_is_preserved() -> None:
    base = _FakeAtariEnv(life_loss_step=2)
    env = EpisodicLifeEnv(MaxAndSkipEnv(base, skip=4))
    env.reset()

    _, reward, terminated, truncated, _ = env.step(1)

    assert base.steps == 4
    assert reward == 4.0
    assert terminated
    assert not truncated

    env.reset()

    assert base.steps == 8
