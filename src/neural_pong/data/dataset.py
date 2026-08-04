import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PongDataset(Dataset):
    def __init__(self, data_path: str, transform=None):
        data = np.load(data_path)
        self.frames = torch.from_numpy(data["frames"]).float()
        self.actions = torch.from_numpy(data["actions"]).long()
        self.next_frames = torch.from_numpy(data["next_frames"]).float()
        self.rewards = torch.from_numpy(data["rewards"]).float()
        self.dones = torch.from_numpy(data["dones"]).float()
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        current_frame = self.frames[idx].unsqueeze(0)
        next_frame = self.next_frames[idx].unsqueeze(0)

        if self.transform:
            current_frame = self.transform(current_frame)
            next_frame = self.transform(next_frame)

        return {
            "frame": current_frame,
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
