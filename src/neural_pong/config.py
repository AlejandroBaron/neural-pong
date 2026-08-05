from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Data paths
    data_dir: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")
    runs_dir: Path = Path("runs")

    # Data collection
    num_episodes: int = 100
    max_steps_per_episode: int = 2000

    # Model
    frame_size: int = 84
    frame_channels: int = 2
    frame_shape: tuple = (84, 84)
    num_actions: int = 6
    latent_dim: int = 256
    learning_rate: float = 1e-4
    batch_size: int = 64
    num_workers: int = 4

    # Training
    num_epochs: int = 50
    val_split: float = 0.1
    checkpoint_every: int = 5

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
