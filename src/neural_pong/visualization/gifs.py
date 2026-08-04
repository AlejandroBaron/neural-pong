"""Generate blog-ready GIFs."""

from pathlib import Path

import imageio
import numpy as np
import torch

from neural_pong.config import Config
from neural_pong.models.world_model import WorldModel


def generate_gifs(
    checkpoint: str,
    data: str = "data/pong_transitions.npz",
    output: str = "gifs/",
    lengths: list[int] | None = None,
    samples: int = 6,
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

    rollout_lengths = lengths or [1, 20]
    num_samples = samples

    saved = 0

    with torch.no_grad():
        for rollout_len in rollout_lengths:
            for _i in range(min(num_samples // len(rollout_lengths), 5)):
                start_idx = np.random.randint(0, len(frames) - rollout_len)

                real_frames = [frames[start_idx]]
                for j in range(rollout_len):
                    real_frames.append(next_frames[start_idx + j])

                current = torch.from_numpy(frames[start_idx]).unsqueeze(0).unsqueeze(0).to(device)
                rollout_actions = (
                    torch.from_numpy(actions[start_idx : start_idx + rollout_len]).long().to(device)
                )

                pred_frames = [current.cpu().numpy()[0, 0]]
                for act in rollout_actions:
                    current, _, _ = model.predict_next(current, act.unsqueeze(0))
                    pred_frames.append(current.cpu().numpy()[0, 0])

                real_arr = np.array(real_frames)
                pred_arr = np.array(pred_frames)
                diff = np.abs(real_arr - pred_arr)

                gif_data = []
                for j in range(min(len(real_arr), len(pred_arr))):
                    real_img = (real_arr[j] * 255).astype(np.uint8).repeat(3, axis=-1)
                    pred_img = (pred_arr[j] * 255).astype(np.uint8).repeat(3, axis=-1)
                    diff_img = (diff[j] * 255).astype(np.uint8).repeat(3, axis=-1)

                    combined = np.vstack([real_img, pred_img, diff_img])
                    gif_data.append(combined)

                gif_path = output_path / f"rollout_{rollout_len}steps_{saved}.gif"
                imageio.mimsave(str(gif_path), gif_data, fps=30)
                print(f"Saved {gif_path}")
                saved += 1

    print(f"\nGenerated {saved} GIF visualizations in {output}")
    return 0
