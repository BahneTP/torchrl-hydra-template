#!/usr/bin/env python
"""Render the DER Atari visual preprocessing stages.

Outputs:
  raw_01.png ... raw_04.png
  maxpooled.png
  grayscale.png
  resized_84.png
  framestack_01.png ... framestack_04.png
  preprocessing_contact_sheet.png

The pipeline mirrors configs/environment/atari100k_der.yaml:
raw ALE frames -> frame skip 4 + max-pool over the last two frames ->
grayscale -> resize 84x84 -> 4-frame stack.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CHECKPOINT = (
    "/home/bthiehl/Transfer-Learning-in-RL/logs/jepa_comparison/pca_data/"
    "dino_jamesbond_straightening_pca_20260730_123204/checkpoints/"
    "full_block3/seed_1/last.pt"
)
JAMESBOND_ACTION_MEANINGS = [
    "NOOP",
    "FIRE",
    "UP",
    "RIGHT",
    "LEFT",
    "DOWN",
    "UPRIGHT",
    "UPLEFT",
    "DOWNRIGHT",
    "DOWNLEFT",
    "UPFIRE",
    "RIGHTFIRE",
    "LEFTFIRE",
    "DOWNFIRE",
    "UPRIGHTFIRE",
    "UPLEFTFIRE",
    "DOWNRIGHTFIRE",
    "DOWNLEFTFIRE",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="Jamesbond")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--action", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument(
        "--rollout-agent-steps",
        type=int,
        default=500,
        help="Number of DER-preprocessed agent steps to roll before saving images.",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "der_preprocessing")
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--use-checkpoint-policy",
        action="store_true",
        help="Load the DER checkpoint and use its greedy policy for the shown action.",
    )
    return parser.parse_args()


def make_raw_env(game: str):
    import gymnasium as gym

    try:
        import ale_py

        gym.register_envs(ale_py)
    except ImportError:
        pass

    return gym.make(
        f"ALE/{game}-v5",
        frameskip=1,
        repeat_action_probability=0.0,
        disable_env_checker=True,
        obs_type="rgb",
    )


def gray_and_resize(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray_image = Image.fromarray(rgb.astype(np.uint8)).convert("L")
    resized_image = gray_image.resize((84, 84), Image.Resampling.BILINEAR)
    return np.asarray(gray_image), np.asarray(resized_image)


def save_rgb(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8)).save(path)


def save_gray(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8)).save(path)


def step_with_raw_frames(env, action: int, skip: int = 4):
    raw_frames = []
    total_reward = 0.0
    terminated = truncated = False
    info = {}
    obs = None
    for _ in range(skip):
        obs, reward, terminated, truncated, info = env.step(action)
        raw_frames.append(obs)
        total_reward += float(reward)
        if terminated or truncated:
            break
    if len(raw_frames) >= 2:
        maxpooled = np.maximum(raw_frames[-2], raw_frames[-1])
    else:
        maxpooled = raw_frames[-1]
    return raw_frames, maxpooled, total_reward, terminated, truncated, info


def load_checkpoint_agent(checkpoint: Path, device: str = "cpu"):
    old_repo = checkpoint
    while old_repo.name != "Transfer-Learning-in-RL" and old_repo != old_repo.parent:
        old_repo = old_repo.parent
    if old_repo.name != "Transfer-Learning-in-RL":
        old_repo = Path("/home/bthiehl/Transfer-Learning-in-RL")
    sys.path.insert(0, str(old_repo))

    from src.algorithms.atari100k.der import DERAgent, DERConfig

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    agent_state = state.policy_state_dict
    config = DERConfig(**agent_state["config"])
    config.device = device
    agent = DERAgent(config)
    agent.load_state_dict(agent_state)
    return agent


def map_checkpoint_action(action: int, env) -> int:
    if action < env.action_space.n:
        return action
    current_meanings = env.unwrapped.get_action_meanings()
    source_meaning = (
        JAMESBOND_ACTION_MEANINGS[action]
        if action < len(JAMESBOND_ACTION_MEANINGS)
        else "NOOP"
    )
    return current_meanings.index(source_meaning) if source_meaning in current_meanings else 0


def select_checkpoint_action(agent, env, stack: list[np.ndarray]) -> int:
    # Old atari100k DER stores states as NHWC frame stacks of uint8 grayscale frames.
    stacked = np.stack(stack, axis=-1)[None]
    action = agent.select_action(stacked, eval_mode=True)
    return map_checkpoint_action(int(action.reshape(-1)[0].item()), env)


def choose_action(args: argparse.Namespace, env, stack: list[np.ndarray], agent=None) -> int:
    if args.action is not None:
        return args.action
    if args.use_checkpoint_policy:
        return select_checkpoint_action(agent, env, stack)
    return int(env.action_space.sample())


def update_stack(stack: list[np.ndarray], maxpooled: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    gray, resized = gray_and_resize(maxpooled)
    return stack[1:] + [resized], gray, resized


def make_contact_sheet(out_dir: Path, names: list[tuple[str, Path]]) -> None:
    thumbs = []
    for label, path in names:
        image = Image.open(path).convert("RGB")
        image.thumbnail((180, 150), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (190, 180), "white")
        canvas.paste(image, ((190 - image.width) // 2, 8))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 158), label, fill=(0, 0, 0))
        thumbs.append(canvas)

    cols = 4
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 190, rows * 180), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 190, (idx // cols) * 180))
    sheet.save(out_dir / "preprocessing_contact_sheet.png")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = make_raw_env(args.game)
    obs, _ = env.reset(seed=args.seed)
    agent = load_checkpoint_agent(args.checkpoint) if args.use_checkpoint_policy else None

    for _ in range(args.warmup_steps):
        action = int(env.action_space.sample())
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    _, first_resized = gray_and_resize(obs)
    stack = [first_resized.copy() for _ in range(4)]

    resets = 0
    for _ in range(args.rollout_agent_steps):
        action = choose_action(args, env, stack, agent)
        _, maxpooled, _, terminated, truncated, _ = step_with_raw_frames(env, action)
        stack, _, _ = update_stack(stack, maxpooled)
        if terminated or truncated:
            resets += 1
            obs, _ = env.reset()
            _, first_resized = gray_and_resize(obs)
            stack = [first_resized.copy() for _ in range(4)]

    action = choose_action(args, env, stack, agent)

    raw_frames, maxpooled, _, terminated, truncated, _ = step_with_raw_frames(env, action)
    for idx, frame in enumerate(raw_frames, start=1):
        save_rgb(args.out_dir / f"raw_{idx:02d}.png", frame)
    gray, resized = gray_and_resize(maxpooled)
    save_rgb(args.out_dir / "maxpooled.png", maxpooled)
    save_gray(args.out_dir / "grayscale.png", gray)
    save_gray(args.out_dir / "resized_84.png", resized)

    stack = stack[1:] + [resized]
    for idx, frame in enumerate(stack, start=1):
        save_gray(args.out_dir / f"framestack_{idx:02d}.png", frame)

    labels = [(f"raw {idx}", args.out_dir / f"raw_{idx:02d}.png") for idx in range(1, len(raw_frames) + 1)]
    labels.extend(
        [
            ("maxpool", args.out_dir / "maxpooled.png"),
            ("grayscale", args.out_dir / "grayscale.png"),
            ("resize 84", args.out_dir / "resized_84.png"),
        ]
    )
    labels.extend([(f"stack {idx}", args.out_dir / f"framestack_{idx:02d}.png") for idx in range(1, 5)])
    make_contact_sheet(args.out_dir, labels)

    env.close()
    print(f"saved DER preprocessing images to {args.out_dir}")
    print(
        f"game={args.game}; rollout_agent_steps={args.rollout_agent_steps}; "
        f"shown action={action}; resets={resets}; "
        f"terminated={terminated}; truncated={truncated}"
    )


if __name__ == "__main__":
    main()
