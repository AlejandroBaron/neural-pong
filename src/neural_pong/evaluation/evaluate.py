"""Evaluate the world model."""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from neural_pong.config import Config
from neural_pong.data.dataset import get_dataloaders
from neural_pong.models.world_model import WorldModel


def evaluate_model(
    checkpoint: str,
    data: str = "data/pong_transitions.npz",
    output: str = "eval/",
) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()

    model = WorldModel(latent_dim=config.latent_dim, num_actions=config.num_actions).to(device)

    checkpoint_data = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    print(f"Loaded checkpoint from {checkpoint}")

    model.eval()

    frame_loss_fn = nn.MSELoss()
    reward_loss_fn = nn.MSELoss()
    _, val_loader = get_dataloaders(
        data,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_split=config.val_split,
    )

    frame_mse = 0
    reward_mse = 0
    done_correct = 0
    total = 0

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    saved = 0
    samples_to_save = 8

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            frame = batch["frame"].to(device)
            action = batch["action"].to(device)
            next_frame = batch["next_frame"].to(device)
            reward = batch["reward"].to(device)
            done = batch["done"].to(device)

            pred_frame, pred_reward, pred_done = model(frame, action)

            frame_prob = torch.sigmoid(pred_frame)
            frame_mse += frame_loss_fn(frame_prob, next_frame).item() * len(frame)
            reward_mse += reward_loss_fn(pred_reward, reward).item() * len(reward)
            pred_done_binary = (pred_done > 0).float()  # done head emits logits
            done_correct += (pred_done_binary == done).sum().item()
            total += len(frame)

            if saved < samples_to_save:
                for i in range(min(len(frame), samples_to_save - saved)):
                    gt = next_frame[i, 0].cpu().numpy()
                    pred = frame_prob[i, 0].cpu().numpy()
                    diff = np.abs(gt - pred)

                    comparison = np.hstack(
                        [
                            (gt * 255).astype(np.uint8),
                            (pred * 255).astype(np.uint8),
                            (diff * 255).astype(np.uint8),
                        ]
                    )
                    cv2.imwrite(str(output_path / f"sample_{saved}.png"), comparison)
                    saved += 1

    frame_mse /= total
    reward_mse /= total
    done_acc = done_correct / total

    print("\n=== Evaluation Results ===")
    print(f"Frame MSE: {frame_mse:.6f}")
    print(f"Reward MSE: {reward_mse:.6f}")
    print(f"Done Accuracy: {done_acc:.4f}")
    print(f"Saved {saved} sample visualizations to {output}")

    return 0
