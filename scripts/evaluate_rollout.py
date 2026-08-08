"""Rollout battery for a trained world model checkpoint.

Usage: uv run python scripts/evaluate_rollout.py checkpoints/X.pt [--data data/pong_oracle_v6.npz]

Tests, from a serve-state seed and a mid-rally seed:
  - 300-step self-fed rollout: motion, ball presence, paddle survival
  - key response: 20 steps of UP / DOWN / NOOP from a static seed
  - bounce test: seed just before a wall bounce, roll 30 steps, ball must come back
Saves frame strips to /tmp for eyeballing.
"""

import argparse

import cv2
import numpy as np
import torch

from neural_pong.config import Config
from neural_pong.models.world_model import WorldModel

# v6 frame geometry: walls rows 0-4 and 78-83, play area rows 5-77
PLAY = slice(6, 77)


def paddle_y(f: np.ndarray, right: bool = True) -> float | None:
    """Mean row of the rightmost (leftmost) vertical structure in the play area."""
    area = f[PLAY]
    xs = np.where(area.mean(axis=0) > 0.02)[0]
    xs = xs[xs > 55] if right else xs[xs < 29]
    if not len(xs):
        return None
    x = xs[-1] if right else xs[0]
    ys = np.where(area[:, x] > 0)[0]
    return round(float(ys.mean()) + 6, 1) if len(ys) else None


def ball_pos(f: np.ndarray) -> np.ndarray | None:
    """(x, y) of a ball-sized blob in mid-field, else None."""
    ys, xs = np.where(f[PLAY, 12:72] > 0)
    if len(xs) < 2 or len(xs) > 40:
        return None
    return np.array([xs.mean() + 12, ys.mean() + 6])


def rollout(model, seed_frames, actions):
    """Feed `actions` through the model starting from seed_frames; yield predictions."""
    cur = torch.from_numpy(np.stack(seed_frames)).unsqueeze(0).to(DEVICE)
    out = []
    with torch.no_grad():
        for a in actions:
            nxt, _, _ = model.predict_next(cur, a.view(1))
            out.append(nxt[0, 0].cpu().numpy())
            cur = torch.cat([cur[:, 1:], nxt], dim=1)
    return out


def seed_stack(frames, idx, history=4):
    return [(frames[idx - k] >= 0.5).astype(np.float32) for k in range(history - 1, -1, -1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--data", default="data/pong_oracle_v6.npz")
    args = parser.parse_args()

    config = Config()
    model = WorldModel(
        config.latent_dim, config.num_actions, in_channels=config.frame_channels
    ).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE)["model_state_dict"])
    model.eval()

    d = np.load(args.data)
    frames, acts = d["frames"] / 255.0, d["actions"]

    # --- 300-step rollouts from a serve seed and a mid-rally seed
    for seed, tag in [(5 * 2000 + 45, "serve"), (3000, "rally")]:
        actions = torch.from_numpy(acts[seed : seed + 300]).long().to(DEVICE)
        preds = rollout(model, seed_stack(frames, seed), actions)
        motions = [
            float((preds[i] != preds[i - 1]).mean()) if i > 0 else float("nan")
            for i in range(len(preds))
        ]
        ball = [float(p[PLAY, 12:72].mean()) for p in preds]
        at = [1, 50, 100, 200, 300]
        print(f"[{tag}] motion: " + " ".join(f"@{t}={motions[t-1]:.4f}" for t in at))
        print(f"[{tag}] ball mass: " + " ".join(f"@{t}={ball[t-1]:.4f}" for t in at))
        last = preds[-1]
        print(
            f"[{tag}] @300 paddles: right y={paddle_y(last)}, left y={paddle_y(last, right=False)}"
        )
        strip = np.hstack([preds[t - 1] for t in at])
        cv2.imwrite(
            f"/tmp/eval_{tag}.png",
            cv2.resize((strip * 255).astype(np.uint8), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST),
        )

    # --- key response (20 held steps from a mid-rally seed)
    f0 = (frames[3000] >= 0.5).astype(np.float32)
    y0 = paddle_y(f0)
    for act, name in [(2, "UP"), (3, "DOWN"), (0, "NOOP")]:
        actions = torch.full((20,), act, dtype=torch.long, device=DEVICE)
        preds = rollout(model, seed_stack(frames, 3000), actions)
        print(f"[keys] {name} x20: paddle y {y0} -> {paddle_y(preds[-1])}")

    # --- bounce test: seed 6 frames before a top-wall bounce, roll 30
    prev, curr = None, None
    bounce_idx = None
    for i in range(1000, 40000):
        p = ball_pos((frames[i] >= 0.5).astype(np.float32))
        if prev is not None and curr is not None and p is not None:
            vy0, vy1 = curr[1] - prev[1], p[1] - curr[1]
            if vy0 < -0.3 and curr[1] < 16:  # moving up, near top wall
                bounce_idx = i
                break
        prev, curr = curr, p
    if bounce_idx is not None:
        actions = torch.from_numpy(acts[bounce_idx : bounce_idx + 30]).long().to(DEVICE)
        preds = rollout(model, seed_stack(frames, bounce_idx), actions)
        ys = [ball_pos(p) for p in preds]
        ys = [p[1] for p in ys if p is not None]
        if ys:
            print(
                f"[bounce] seeded pre-bounce (ball y~{curr[1]:.0f}); "
                f"model ball y over 30 steps: min={min(ys):.1f} max={max(ys):.1f} last={ys[-1]:.1f} "
                f"({'OK: came back down' if max(ys) - min(ys) > 8 else 'SUSPECT: no rebound'})"
            )
        else:
            print("[bounce] ball vanished within 30 steps of the bounce")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    main()
