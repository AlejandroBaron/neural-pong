import ale_py
import cv2
import gymnasium as gym
import numpy as np

gym.register_envs(ale_py)


def preprocess_frame(frame: np.ndarray, size: int = 84) -> np.ndarray:
    """Crop to the playfield (drops score and borders), grayscale, resize.

    Cropping before resizing buys ~15% vertical resolution, which is what keeps
    the ball separable from the walls during bounces.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cropped = gray[24:210, 8:152]  # walls at raw rows 24-33 and 194-209
    resized = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
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
