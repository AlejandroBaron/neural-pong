"""Play the learned Pong environment."""

import cv2
import gymnasium as gym
import numpy as np
import torch

from neural_pong.config import Config
from neural_pong.environments.neural import NeuralPongEnv
from neural_pong.models.world_model import WorldModel


def play_neural_pong(checkpoint: str) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()

    model = WorldModel(latent_dim=config.latent_dim, num_actions=config.num_actions).to(device)

    checkpoint_data = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()

    # Create environments
    neural_env = NeuralPongEnv(model, config.frame_size)
    real_env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")

    print("Neural Pong - Press UP=w/ArrowUp, DOWN=s/ArrowDown, q=quit")

    # Initialize
    obs, _ = real_env.reset()
    _ = neural_env.reset(obs)

    action_map = {0: 0, 1: 1, 2: 2, 3: 3}  # noop, fire, right, left

    running = True
    frame_idx = 0

    while running:
        key = cv2.waitKey(1) & 0xFF

        action = 0
        if key == ord("q"):
            running = False
        elif key == ord("w") or key == 83:  # up
            action = 2
        elif key == ord("s") or key == 81:  # down
            action = 3

        real_obs, real_reward, terminated, truncated, _ = real_env.step(action_map.get(action, 0))
        real_done = terminated or truncated

        neural_frame, neural_reward, _neural_done = neural_env.step(action)

        if real_done:
            obs, _ = real_env.reset()
            _ = neural_env.reset(obs)

        real_rgb = cv2.cvtColor(real_obs, cv2.COLOR_RGB2GRAY)
        real_resized = cv2.resize(real_rgb, (config.frame_size * 3, config.frame_size * 3))
        neural_resized = cv2.resize(
            (neural_frame * 255).astype(np.uint8), (config.frame_size * 3, config.frame_size * 3)
        )

        combined = np.hstack([real_resized, neural_resized])
        cv2.imshow("Real (left) vs Neural (right)", combined)

        frame_idx += 1

        if frame_idx % 100 == 0:
            print(
                f"Frame {frame_idx}, Real reward: {real_reward}, "
                f"Neural reward: {neural_reward.item():.4f}"
            )

    real_env.close()
    cv2.destroyAllWindows()

    return 0
