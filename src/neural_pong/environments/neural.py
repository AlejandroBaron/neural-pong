"""Learned Pong environment wrapper."""

import numpy as np
import torch

from neural_pong.models.world_model import WorldModel


class NeuralPongEnv:
    """Advance frames using a trained world model."""

    def __init__(self, model: WorldModel, frame_size: int = 84) -> None:
        self.model = model
        self.frame_size = frame_size
        self.state: torch.Tensor | None = None

    def reset(self, frame: np.ndarray | torch.Tensor | None = None) -> np.ndarray:
        device = next(self.model.parameters()).device
        if frame is None:
            state = torch.rand(1, 1, self.frame_size, self.frame_size, device=device)
        elif isinstance(frame, np.ndarray):
            state = torch.from_numpy(frame).float().to(device).unsqueeze(0).unsqueeze(0)
        else:
            state = frame.to(device).unsqueeze(0)

        self.state = state
        return state.cpu().numpy()[0, 0]

    def step(self, action: int | torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.state is None:
            raise ValueError("Must call reset() first")

        if isinstance(action, int):
            action = torch.tensor([action], device=self.state.device)

        with torch.no_grad():
            next_frame, reward, done = self.model.predict_next(self.state, action)

        self.state = next_frame
        return next_frame.cpu().numpy()[0, 0], reward.cpu().numpy(), done.cpu().numpy()
