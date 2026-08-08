import ale_py
import cv2
import gymnasium as gym
import numpy as np

gym.register_envs(ale_py)


def preprocess_frame(frame: np.ndarray, size: int = 84) -> np.ndarray:
    """Convert frame to grayscale, resize, and mask the score band (rows 0-7)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    resized[:8, :] = 0  # score digits: not part of the game the model must learn
    return resized.astype(np.float32) / 255.0


def create_env(env_name: str = "PongNoFrameskip-v4"):
    """Create a wrapped Atari environment."""
    return gym.make(env_name, render_mode="rgb_array")


def get_frame_shape(env) -> tuple:
    """Get the shape of a raw frame from environment."""
    env_test = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")
    _, _ = env_test.reset()
    obs, _ = env_test.step(0)
    env_test.close()
    return obs.shape
