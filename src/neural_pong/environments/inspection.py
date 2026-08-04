"""Inspect the Pong environment."""

import gymnasium as gym

from neural_pong.environments.pong import preprocess_frame


def inspect_environment() -> int:
    env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")

    print("=== Pong Environment Inspection ===")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Number of actions: {env.action_space.n}")

    obs, _ = env.reset()
    print(f"\nFrame shape: {obs.shape}")
    print(f"Frame dtype: {obs.dtype}")

    print("\n=== Step-by-step inspection ===")
    done = False
    step = 0

    while not done and step < 10:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        print(f"Step {step}: action={action}, reward={reward}, done={done}")

        frame = preprocess_frame(obs)
        print(f"  Preprocessed frame shape: {frame.shape}")

        step += 1

    env.close()
    print("\nDone")
    return 0
