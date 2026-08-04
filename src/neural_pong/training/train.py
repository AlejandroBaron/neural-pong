"""Train the world model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from neural_pong.config import Config
from neural_pong.data.dataset import get_dataloaders
from neural_pong.models.world_model import WorldModel


def train_model(data: str = "data/pong_transitions.npz", resume: str | None = None) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()

    model = WorldModel(latent_dim=config.latent_dim, num_actions=config.num_actions).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    frame_loss_fn = nn.MSELoss()
    reward_loss_fn = nn.MSELoss()
    done_loss_fn = nn.BCELoss()

    writer = SummaryWriter(log_dir=str(config.runs_dir / "world_model"))

    train_loader, val_loader = get_dataloaders(
        data,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_split=config.val_split,
    )

    start_epoch = 0
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    for epoch in tqdm(range(start_epoch, config.num_epochs), desc="Training"):
        model.train()
        train_losses = {"frame": 0, "reward": 0, "done": 0}

        for batch in train_loader:
            frame = batch["frame"].to(device)
            action = batch["action"].to(device)
            next_frame = batch["next_frame"].to(device)
            reward = batch["reward"].to(device)
            done = batch["done"].to(device)

            pred_frame, pred_reward, pred_done = model(frame, action)

            f_loss = frame_loss_fn(pred_frame, next_frame)
            r_loss = reward_loss_fn(pred_reward, reward)
            d_loss = done_loss_fn(pred_done, done)
            loss = f_loss + r_loss + d_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses["frame"] += f_loss.item()
            train_losses["reward"] += r_loss.item()
            train_losses["done"] += d_loss.item()

        model.eval()
        val_losses = {"frame": 0, "reward": 0, "done": 0}
        with torch.no_grad():
            for batch in val_loader:
                frame = batch["frame"].to(device)
                action = batch["action"].to(device)
                next_frame = batch["next_frame"].to(device)
                reward = batch["reward"].to(device)
                done = batch["done"].to(device)

                pred_frame, pred_reward, pred_done = model(frame, action)

                f_loss = frame_loss_fn(pred_frame, next_frame)
                r_loss = reward_loss_fn(pred_reward, reward)
                d_loss = done_loss_fn(pred_done, done)

                val_losses["frame"] += f_loss.item()
                val_losses["reward"] += r_loss.item()
                val_losses["done"] += d_loss.item()

        n_batches = len(train_loader)
        for key in train_losses:
            writer.add_scalar(f"Train/{key}_loss", train_losses[key] / n_batches, epoch)
            writer.add_scalar(f"Val/{key}_loss", val_losses[key] / n_batches, epoch)

        if epoch % 5 == 0:
            sample_batch = next(iter(val_loader))
            with torch.no_grad():
                pred_frame, _, _ = model(
                    sample_batch["frame"].to(device)[:4], sample_batch["action"].to(device)[:4]
                )
                writer.add_images("Sample/Predicted", pred_frame.cpu(), epoch)
                writer.add_images("Sample/GroundTruth", sample_batch["next_frame"][:4], epoch)

        if (epoch + 1) % config.checkpoint_every == 0:
            idx = (epoch + 1) // config.checkpoint_every
            checkpoint_path = config.checkpoint_dir / f"world_model_{idx}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )

    writer.close()
    print("Training complete!")
    return 0
