"""Train the world model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from neural_pong.config import Config
from neural_pong.data.dataset import get_dataloaders, get_sequence_dataloaders
from neural_pong.models.world_model import WorldModel


def train_model(
    data: str = "data/pong_transitions.npz",
    resume: str | None = None,
    autoregressive: bool = False,
    seq_len: int = 5,
    prefix: str = "world_model",
    epochs: int | None = None,
) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()

    model = WorldModel(latent_dim=config.latent_dim, num_actions=config.num_actions).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    num_epochs = epochs if epochs is not None else config.num_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Foreground pixels are rare; up-weight them so the ball/paddles matter.
    pos_weight = torch.tensor([5.0], device=device)
    frame_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    reward_loss_fn = nn.MSELoss()
    done_loss_fn = nn.BCELoss()

    writer = SummaryWriter(
        log_dir=str(config.runs_dir / ("ar_world_model" if autoregressive else "world_model"))
    )

    if autoregressive:
        train_loader, val_loader = get_sequence_dataloaders(
            data,
            seq_len=seq_len,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            val_split=config.val_split,
        )
    else:
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

    for epoch in tqdm(range(start_epoch, num_epochs), desc="Training"):
        model.train()
        train_losses = {"frame": 0, "reward": 0, "done": 0}
        train_batches = 0

        for batch in train_loader:
            frame = batch["frame"].to(device)
            action = batch["action"].to(device)
            next_frame = batch["next_frame"].to(device)
            reward = batch["reward"].to(device)
            done = batch["done"].to(device)

            if autoregressive:
                loss, step_losses = _autoregressive_step(
                    model,
                    frame_loss_fn,
                    reward_loss_fn,
                    done_loss_fn,
                    frame,
                    action,
                    next_frame,
                    reward,
                    done,
                )
            else:
                pred_frame, pred_reward, pred_done = model(frame, action)

                f_loss = frame_loss_fn(pred_frame, next_frame)
                r_loss = reward_loss_fn(pred_reward, reward)
                d_loss = done_loss_fn(pred_done, done)
                loss = f_loss + r_loss + d_loss
                step_losses = {
                    "frame": f_loss.item(),
                    "reward": r_loss.item(),
                    "done": d_loss.item(),
                }

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for key in step_losses:
                train_losses[key] += step_losses[key]
            train_batches += 1

        scheduler.step()

        model.eval()
        val_losses = {"frame": 0, "reward": 0, "done": 0}
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                frame = batch["frame"].to(device)
                action = batch["action"].to(device)
                next_frame = batch["next_frame"].to(device)
                reward = batch["reward"].to(device)
                done = batch["done"].to(device)

                if autoregressive:
                    loss, step_losses = _autoregressive_step(
                        model,
                        frame_loss_fn,
                        reward_loss_fn,
                        done_loss_fn,
                        frame,
                        action,
                        next_frame,
                        reward,
                        done,
                    )
                else:
                    pred_frame, pred_reward, pred_done = model(frame, action)

                    f_loss = frame_loss_fn(pred_frame, next_frame)
                    r_loss = reward_loss_fn(pred_reward, reward)
                    d_loss = done_loss_fn(pred_done, done)
                    step_losses = {
                        "frame": f_loss.item(),
                        "reward": r_loss.item(),
                        "done": d_loss.item(),
                    }

                for key in step_losses:
                    val_losses[key] += step_losses[key]
                val_batches += 1

        for key in train_losses:
            writer.add_scalar(f"Train/{key}_loss", train_losses[key] / train_batches, epoch)
            writer.add_scalar(f"Val/{key}_loss", val_losses[key] / val_batches, epoch)

        if epoch % 5 == 0:
            sample_batch = next(iter(val_loader))
            with torch.no_grad():
                if autoregressive:
                    sample_frames = sample_batch["frame"].to(device)
                    sample_actions = sample_batch["action"].to(device)
                    pred_frame, _, _ = model(sample_frames[:4, 0], sample_actions[:4, 0])
                else:
                    pred_frame, _, _ = model(
                        sample_batch["frame"].to(device)[:4], sample_batch["action"].to(device)[:4]
                    )
                writer.add_images("Sample/Predicted", pred_frame.cpu(), epoch)
                writer.add_images("Sample/GroundTruth", sample_batch["next_frame"][:4, 0], epoch)

        if (epoch + 1) % config.checkpoint_every == 0:
            idx = (epoch + 1) // config.checkpoint_every
            checkpoint_path = config.checkpoint_dir / f"{prefix}_{idx}.pt"
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


def _autoregressive_step(
    model: WorldModel,
    frame_loss_fn: nn.Module,
    reward_loss_fn: nn.Module,
    done_loss_fn: nn.Module,
    frame_seq: torch.Tensor,
    action_seq: torch.Tensor,
    next_frame_seq: torch.Tensor,
    reward_seq: torch.Tensor,
    done_seq: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Roll out `seq_len` steps using the model's own predictions as input.

    The input frame is detached between steps so gradients only flow through a
    single prediction step, but the network is still trained on its own noisy
    outputs (exposure-bias reduction).
    """
    seq_len = frame_seq.size(1)
    input_frame = frame_seq[:, 0]
    total_loss = 0
    losses = {"frame": 0, "reward": 0, "done": 0}

    for step in range(seq_len):
        pred_frame, pred_reward, pred_done = model(input_frame, action_seq[:, step])

        f_loss = frame_loss_fn(pred_frame, next_frame_seq[:, step])
        r_loss = reward_loss_fn(pred_reward, reward_seq[:, step])
        d_loss = done_loss_fn(pred_done, done_seq[:, step])
        total_loss += f_loss + r_loss + d_loss

        losses["frame"] += f_loss.item()
        losses["reward"] += r_loss.item()
        losses["done"] += d_loss.item()

        # Prepare next autoregressive input from the model's soft prediction.
        next_frame_soft = torch.sigmoid(pred_frame).detach()
        input_frame = torch.cat([input_frame[:, 1:, :, :], next_frame_soft], dim=1)

    return total_loss, losses
