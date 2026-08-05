"""Learned Pong environment wrapper."""

import numpy as np
import torch

from neural_pong.environments.pong import preprocess_frame
from neural_pong.models.world_model import WorldModel


class NeuralPongEnv:
    """Advance frames using a trained world model."""

    def __init__(self, model: WorldModel, frame_size: int = 84) -> None:
        self.model = model
        self.frame_size = frame_size
        self.state: torch.Tensor | None = None

    def _normalise_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Convert a raw or preprocessed frame to a (1, 1, H, W) float tensor."""
        if frame.ndim == 3:
            frame = preprocess_frame(frame, self.frame_size)
        elif frame.shape != (self.frame_size, self.frame_size):
            raise ValueError(f"Unexpected frame shape {frame.shape}")

        tensor = torch.from_numpy(frame).float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor.unsqueeze(0).unsqueeze(0)

    def reset(self, frame: np.ndarray | torch.Tensor | None = None) -> np.ndarray:
        device = next(self.model.parameters()).device
        if frame is None:
            current = torch.rand(1, 1, self.frame_size, self.frame_size, device=device)
        elif isinstance(frame, np.ndarray):
            current = self._normalise_frame(frame).to(device)
        else:
            current = frame.to(device)
            if current.dim() == 3:
                current = current.unsqueeze(0)

        # Duplicate the initial frame so the model still sees two channels.
        self.state = current.repeat(1, 2, 1, 1)
        return self.state[:, -1, :, :].cpu().numpy()[0]

    def step(self, action: int | torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.state is None:
            raise ValueError("Must call reset() first")

        if isinstance(action, int):
            action = torch.tensor([action], device=self.state.device)

        with torch.no_grad():
            next_frame, reward, done = self.model.predict_next(self.state, action)

        # Roll the buffer: previous becomes current, current becomes prediction.
        self.state = torch.cat([self.state[:, 1:, :, :], next_frame], dim=1)
        return next_frame.cpu().numpy()[0, 0], reward.cpu().numpy(), done.cpu().numpy()
