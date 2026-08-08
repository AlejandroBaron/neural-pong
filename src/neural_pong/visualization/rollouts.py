"""Generate autoregressive rollout videos."""

from pathlib import Path

import cv2
import numpy as np
import torch

from neural_pong.config import Config
from neural_pong.models.world_model import WorldModel


def visualize_rollouts(
    checkpoint: str,
    data: str = "data/pong_transitions.npz",
    output: str = "videos/",
    lengths: list[int] | None = None,
    samples: int = 5,
) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()

    model = WorldModel(latent_dim=config.latent_dim, num_actions=config.num_actions).to(device)

    checkpoint_data = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()

    transition_data = np.load(data)
    frames = transition_data["frames"]
    actions = transition_data["actions"]
    next_frames = transition_data["next_frames"]

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    rollout_lengths = lengths or [1, 5, 20, 50, 100]
    num_samples = samples

    saved = 0

    with torch.no_grad():
        for rollout_len in rollout_lengths:
            for _i in range(min(num_samples // len(rollout_lengths), 5)):
                start_idx = np.random.randint(0, len(frames) - rollout_len)

                real_frames = [frames[start_idx].astype(np.float32) / 255.0]
                for j in range(rollout_len):
                    real_frames.append(next_frames[start_idx + j].astype(np.float32) / 255.0)

                # Seed the rollout with two identical frames.
                current = (
                    torch.from_numpy(frames[start_idx]).unsqueeze(0).unsqueeze(0).float().to(device)
                    / 255.0
                )
                current = current.repeat(1, config.frame_channels, 1, 1)

                rollout_actions = (
                    torch.from_numpy(actions[start_idx : start_idx + rollout_len]).long().to(device)
                )

                pred_frames = [current[0, -1].cpu().numpy()]
                for act in rollout_actions:
                    prev = current[:, -1:, :, :]
                    next_frame, _, _ = model.predict_next(current, act.unsqueeze(0))
                    current = torch.cat([prev, next_frame], dim=1)
                    pred_frames.append(current[0, -1].cpu().numpy())

                real_arr = np.array(real_frames)
                pred_arr = np.array(pred_frames)

                out_path = output_path / f"rollout_{rollout_len}steps_sample{saved}.mp4"
                height, width = real_arr.shape[-2:]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(str(out_path), fourcc, 30.0, (width * 2, height))

                for j in range(min(len(real_arr), len(pred_arr))):
                    real_img = cv2.cvtColor(
                        (real_arr[j] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
                    )
                    pred_img = cv2.cvtColor(
                        (pred_arr[j] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
                    )
                    combined = np.hstack([real_img, pred_img])
                    out.write(combined)

                out.release()
                print(f"Saved {out_path}")
                saved += 1

    print(f"\nGenerated {saved} rollout videos in {output}")
    return 0
