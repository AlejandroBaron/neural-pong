"""Collect Pong transitions."""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from neural_pong.config import Config
from neural_pong.environments.pong import create_env, preprocess_frame


def _to_binary_uint8(frame: np.ndarray) -> np.ndarray:
    """Binarise a normalised frame and store it compactly as uint8."""
    return (frame >= 0.5).astype(np.uint8) * 255


def collect_transitions(
    episodes: int | None = None, output: str = "data/pong_transitions.npz"
) -> int:
    """Collect Pong dataset using random policy."""
    config = Config()

    env = create_env("PongNoFrameskip-v4")

    prev_frames = []
    frames = []
    next_frames = []
    actions = []
    rewards = []
    dones = []

    num_episodes = episodes or config.num_episodes
    output_path = output

    print(f"Collecting {num_episodes} episodes...")

    for _episode in tqdm(range(num_episodes), desc="Episodes"):
        obs, _ = env.reset()
        done = False
        step = 0
        prev_frame = _to_binary_uint8(preprocess_frame(obs, config.frame_size))

        while not done and step < config.max_steps_per_episode:
            frame = _to_binary_uint8(preprocess_frame(obs, config.frame_size))
            action = env.action_space.sample()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_frame = _to_binary_uint8(preprocess_frame(next_obs, config.frame_size))

            prev_frames.append(prev_frame)
            frames.append(frame)
            next_frames.append(next_frame)
            actions.append(action)
            rewards.append(reward)
            dones.append(float(done))

            obs = next_obs
            step += 1

    env.close()

    # Convert to arrays
    prev_frames = np.array(prev_frames)
    frames = np.array(frames)
    next_frames = np.array(next_frames)
    actions = np.array(actions)
    rewards = np.array(rewards)
    dones = np.array(dones)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Compress: binary frames shrink by ~30-50x over raw float32.
    np.savez_compressed(
        output_path,
        prev_frames=prev_frames,
        frames=frames,
        actions=actions,
        next_frames=next_frames,
        rewards=rewards,
        dones=dones,
    )

    print(f"Saved {len(frames)} transitions to {output_path}")
    print(f"Frame shape: {frames.shape}")
    print(f"Actions: {np.unique(actions)}")

    return 0
