import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _load_array(data, key):
    """Load a numpy array and normalise uint8 frames to [0, 1]."""
    arr = data[key]
    tensor = torch.from_numpy(arr).float()
    if arr.dtype == np.uint8:
        tensor = tensor / 255.0
    return tensor


class PongDataset(Dataset):
    def __init__(self, data_path: str, transform=None):
        data = np.load(data_path)
        self.frames = _load_array(data, "frames")
        self.actions = torch.from_numpy(data["actions"]).long()
        self.next_frames = _load_array(data, "next_frames")
        self.rewards = torch.from_numpy(data["rewards"]).float()
        self.dones = torch.from_numpy(data["dones"]).float()

        # Previous frames give the model velocity context. Fall back to the
        # current frame for datasets collected before this field existed.
        if "prev_frames" in data:
            self.prev_frames = _load_array(data, "prev_frames")
        else:
            self.prev_frames = self.frames.clone()

        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        current_frame = self.frames[idx].unsqueeze(0)
        prev_frame = self.prev_frames[idx].unsqueeze(0)
        next_frame = self.next_frames[idx].unsqueeze(0)

        if self.transform:
            current_frame = self.transform(current_frame)
            prev_frame = self.transform(prev_frame)
            next_frame = self.transform(next_frame)

        # Binarise to a clean {0, 1} palette for both training and rollouts.
        current_frame = (current_frame >= 0.5).float()
        prev_frame = (prev_frame >= 0.5).float()
        next_frame = (next_frame >= 0.5).float()

        frame_stack = torch.cat([prev_frame, current_frame], dim=0)

        return {
            "frame": frame_stack,
            "action": self.actions[idx],
            "next_frame": next_frame,
            "reward": self.rewards[idx],
            "done": self.dones[idx],
        }


def get_dataloaders(
    data_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    val_split: float = 0.1,
):
    """"""
    dataset = PongDataset(data_path)

    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader
