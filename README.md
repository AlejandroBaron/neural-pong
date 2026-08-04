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
uv run neural-pong collect_dataset --episodes 100

# Stage 3-5: Train
uv run neural-pong train_model --data data/pong_transitions.npz

# Stage 6: Evaluate
uv run neural-pong evaluate_model --checkpoint checkpoints/world_model_epoch_XX.pt

# Stage 7: Visualize rollouts
uv run neural-pong visualize_rollouts --checkpoint checkpoints/world_model_epoch_XX.pt

# Stage 9: Play neural Pong
uv run neural-pong play_neural_pong --checkpoint checkpoints/world_model_epoch_XX.pt

# Run tests and formatting checks
uv run --dev pytest
uv run --dev pre-commit run --all-files
```

All executable scripts now share the `neural-pong` CLI; run `uv run neural-pong --help` for every subcommand.

## Model Architecture

```
Input: (frame, action)
  ↓
Frame Encoder: CNN → latent vector
  ↓
Action Embedding: Index → dense vector
  ↓
Concatenate + MLP (transition model)
  ↓
Three outputs:
  - Frame: ConvTranspose → (1, 84, 84)
  - Reward: Linear → scalar
  - Done: Linear + sigmoid → probability
```

## Loss Function

```
L = MSE(frame) + MSE(reward) + BCE(done)
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