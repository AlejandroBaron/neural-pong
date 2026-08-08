import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def _load_array(data, key):
    """Load a numpy array and normalise uint8 frames to [0, 1]."""
    arr = data[key]
    tensor = torch.from_numpy(arr).float()
    if arr.dtype == np.uint8:
        tensor = tensor / 255.0
    return tensor


def _history_stack(frames: torch.Tensor, idx: int, history: int) -> torch.Tensor:
    """Return frames[idx-history+1 .. idx] as a (history, H, W) binary stack.

    Pads by repeating the oldest available frame at sequence starts.
    """
    start = max(0, idx - (history - 1))
    stack = (frames[start : idx + 1] >= 0.5).float()
    if stack.size(0) < history:
        pad = stack[:1].expand(history - stack.size(0), -1, -1)
        stack = torch.cat([pad, stack], dim=0)
    return stack


class PongDataset(Dataset):
    def __init__(self, data_path: str, history: int = 4, transform=None):
        data = np.load(data_path)
        self.frames = _load_array(data, "frames")
        self.actions = torch.from_numpy(data["actions"]).long()
        self.next_frames = _load_array(data, "next_frames")
        self.rewards = torch.from_numpy(data["rewards"]).float()
        self.dones = torch.from_numpy(data["dones"]).float()
        self.history = history
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_stack = _history_stack(self.frames, idx, self.history)
        next_frame = (self.next_frames[idx].unsqueeze(0) >= 0.5).float()

        # Mirror symmetry: left/right flip keeps up/down/fire labels valid.
        if torch.rand(()) < 0.5:
            frame_stack = torch.flip(frame_stack, dims=[-1])
            next_frame = torch.flip(next_frame, dims=[-1])

        if self.transform:
            frame_stack = self.transform(frame_stack)
            next_frame = self.transform(next_frame)

        return {
            "frame": frame_stack,
            "action": self.actions[idx],
            "next_frame": next_frame,
            "reward": self.rewards[idx],
            "done": self.dones[idx],
        }


class PongSequenceDataset(Dataset):
    """Return contiguous sequences of transitions for autoregressive training."""

    def __init__(self, data_path: str, seq_len: int = 5, history: int = 4):
        data = np.load(data_path)
        self.frames = _load_array(data, "frames")
        self.actions = torch.from_numpy(data["actions"]).long()
        self.next_frames = _load_array(data, "next_frames")
        self.rewards = torch.from_numpy(data["rewards"]).float()
        self.dones = torch.from_numpy(data["dones"]).float()

        self.seq_len = seq_len
        self.history = history
        # Avoid indexing past the end of the array.
        self.valid_len = len(self.frames) - seq_len

    def __len__(self):
        return max(1, self.valid_len)

    def __getitem__(self, idx):
        idx = min(idx, self.valid_len - 1)
        end = idx + self.seq_len

        # (seq_len, history, H, W)
        frame_stacks = torch.stack(
            [_history_stack(self.frames, t, self.history) for t in range(idx, end)]
        )
        next_frames = (self.next_frames[idx:end] >= 0.5).float()

        # Mirror symmetry: one flip decision for the whole sequence.
        if torch.rand(()) < 0.5:
            frame_stacks = torch.flip(frame_stacks, dims=[-1])
            next_frames = torch.flip(next_frames, dims=[-1])

        return {
            "frame": frame_stacks,
            "action": self.actions[idx:end],
            "next_frame": next_frames.unsqueeze(1),
            "reward": self.rewards[idx:end],
            "done": self.dones[idx:end],
        }


def get_dataloaders(
    data_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    val_split: float = 0.1,
    history: int = 4,
    oversample: float = 0.0,
):
    dataset = PongDataset(data_path, history=history)

    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    sampler = None
    if oversample > 0:
        # Bounces, paddle hits and point resets are the rare high-change frames
        # the model gets wrong; draw them more often.
        diff = (dataset.frames != dataset.next_frames).float().mean(dim=(1, 2))
        weights = 1.0 + oversample * diff / diff.mean()
        sampler = WeightedRandomSampler(weights[train_set.indices], len(train_set))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


def get_sequence_dataloaders(
    data_path: str,
    seq_len: int = 5,
    batch_size: int = 32,
    num_workers: int = 0,
    val_split: float = 0.1,
    history: int = 4,
):
    dataset = PongSequenceDataset(data_path, seq_len=seq_len, history=history)

    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader
