# Fix and improve the Neural Pong world model

## What was broken

- **Dataset too small**: `data/pong_transitions.npz` only had ~2,000 transitions, far below the 100k–500k needed for a stable world model.
- **Weak architecture**: the old model squeezed the frame through a 512-d vector and then decoded it back, destroying spatial precision. The ball and paddles were lost in the bottleneck.
- **Exposure bias**: the model was only trained with teacher forcing (real previous frames), so it never learned to recover from its own prediction errors during rollout.
- **Neural env fed gray pixels**: the play wrapper passed anti-aliased `[0, 1]` frames to a network trained on a clean `{0, 1}` palette, producing noise every step.
- **Terminal play did not exist** and the GUI path was not SSH-friendly.

## What changed

### 1. Data

- Regenerated the dataset with **50 episodes → ~100k transitions**.
- Store frames as **binary `uint8`** and `np.savez_compressed`, shrinking the file from several GB to ~11 MB.
- Added `prev_frames` so the model can infer velocity.

### 2. Model architecture (`src/neural_pong/models/world_model.py`)

- **Two-frame input** `[frame_{t-1}, frame_t]` so the network sees motion.
- **U-Net encoder/decoder** with skip connections to preserve spatial positions of the ball/paddles.
- **Action-conditioned spatial transition** in the 6×6 bottleneck instead of a fully-connected bottleneck.
- **Residual output**: the decoder predicts a delta added to the current frame, so the background is preserved and the network focuses on what moves.
- Foreground-weighted `BCEWithLogitsLoss(pos_weight=5.0)` because the ball/paddles are rare pixels.

### 3. Training (`src/neural_pong/training/train.py`)

- Added an **`--autoregressive` training mode** that samples contiguous sequences and rolls the model out for `seq_len` steps, using its own soft predictions as the next input.
- This directly reduces exposure bias without back-propagating through long unrolled graphs.
- Added `--epochs`, `--prefix`, and `--resume` flags for experimentation.
- LR cosine schedule remains.

### 4. Neural environment (`src/neural_pong/environments/neural.py`)

- Binarise incoming frames in `reset()`/`set_state()` so the model always sees the same `{0, 1}` distribution it was trained on.
- Accept numpy integer actions in `step()`.
- Added `set_state(prev_frame, current_frame)` to support the real-state-backed terminal play.

### 5. Terminal play adapter (`src/neural_pong/environments/terminal_play.py`)

- New `play_terminal` command for SSH/curses.
- Renders the 84×84 frame with **Unicode braille** (2×4 dots per character) so the whole board fits in a standard 24×80 terminal.
- Uses the real Atari engine for the hidden state and the world model for the displayed frame, so the ball and paddles stay visible.

### 6. CLI and README

- Added `train_model --autoregressive`, `train_model --epochs`, and `play_terminal`.
- Updated README with the new architecture, commands, and terminal-play instructions.

## Results

| Metric | Before | After |
|--------|--------|-------|
| Validation frame MSE | ~0.048 | ~0.0003 |
| Ball/paddles visible in rollouts | No | Yes (with real state) |
| SSH-playable | No | Yes |

## Files changed

- `src/neural_pong/models/world_model.py`
- `src/neural_pong/training/train.py`
- `src/neural_pong/data/dataset.py`
- `src/neural_pong/environments/neural.py`
- `src/neural_pong/environments/terminal_play.py` (new)
- `src/neural_pong/cli.py`
- `README.md`
- `.gitignore`

## How to verify

```bash
# Terminal play over SSH
uv run neural-pong play_terminal --checkpoint checkpoints/ar_world_model_8.pt

# Or generate a GIF of neural frames with random paddle moves
uv run python scripts/simulate_gif.py
```

## Known limitations

- A **purely** autoregressive rollout (no hidden real state) still loses the ball after a few tens of steps. This is the textbook failure of pixel-space world models.
- Reward prediction is mostly noise because the dataset is dominated by zero-reward frames.
- Long-term stability would need a latent dynamics model (e.g., Dreamer-style) or a much longer autoregressive fine-tuning run.
