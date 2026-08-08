# Neural Pong: Learning a Game Engine with a World Model

Replace the Pong game engine with a neural network that predicts game evolution.

## Architecture

```
[frame_t, action_t] → WorldModel → [frame_t+1, reward, done]
```

## Files

| File | Description |
|------|-------------|
| `src/neural_pong/config.py` | Configuration dataclass |
| `src/neural_pong/environments/` | Pong and learned-environment wrappers |
| `src/neural_pong/models/` | World-model architecture |
| `src/neural_pong/data/` | Dataset collection and loading |
| `src/neural_pong/training/` | Model training |
| `src/neural_pong/evaluation/` | Model metrics and comparisons |
| `src/neural_pong/visualization/` | Rollout videos and GIFs |
| `src/neural_pong/cli.py` | Unified command-line entrypoint |
| `tests/test_model.py` | Model forward-pass test |

## Quick Start

```bash
uv sync

# Stage 1: Inspect Pong
uv run neural-pong inspect_environment

# Stage 2: Collect dataset (~100k transitions)
uv run neural-pong collect_dataset --episodes 50

# Stage 3-5: Train
uv run neural-pong train_model --data data/pong_transitions.npz

# Stage 6: Evaluate
uv run neural-pong evaluate_model --checkpoint checkpoints/world_model_6.pt

# Stage 7: Visualize rollouts
uv run neural-pong visualize_rollouts --checkpoint checkpoints/world_model_6.pt

# Stage 8: Play neural Pong (GUI window)
uv run neural-pong play_neural_pong --checkpoint checkpoints/world_model_6.pt

# Stage 9: Play neural Pong in a terminal (works over SSH)
uv run neural-pong play_terminal --checkpoint checkpoints/world_model_6.pt

# Run tests and formatting checks
uv run --dev pytest
uv run --dev pre-commit run --all-files
```

All executable scripts now share the `neural-pong` CLI; run `uv run neural-pong --help` for every subcommand.

## Model Architecture

```
Input: (frame_t-1, frame_t, action_t)
  ↓
Frame Encoder: CNN → spatial latent maps
  ↓
Action Embedding: Index → 64-d vector, broadcast to spatial grid
  ↓
Action-conditioned Conv Transition in latent space
  ↓
U-Net Decoder with skip connections → residual logit delta
  ↓
Final frame = current_frame + predicted delta
  ↓
Three outputs:
  - Frame: raw binary-palette logits `(1, 84, 84)`
  - Reward: Linear → scalar
  - Done: Linear + sigmoid → probability
```

The two-frame input gives the model velocity, the U-Net preserves spatial
positions, and the residual skip lets the network focus on what moves.

## Loss Function

```
L = BCEWithLogits(frame) + MSE(reward) + BCE(done)

Frames are thresholded to the binary palette for both teacher forcing and
autoregressive rollout. Rollout feedback uses the hard argmax palette value,
while training receives the decoder's raw logits.
```

## CLI

```bash
uv run neural-pong --help
```

## Key Concepts

- **World Model**: Learns dynamics instead of using game engine
- **Teacher Forcing**: Train with real frames, test with predictions
- **Autoregressive Rollout**: Using predictions as next input
- **Exposure Bias**: Train on perfect history, test on imperfect predictions