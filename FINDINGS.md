# Findings: getting a neural network to run Pong autonomously

Experiments and conclusions from the quest to make `play_terminal` run purely on the
world model (model consumes its own predictions, no emulator in the loop).
Each section lists the symptom, the diagnosis, and the fix that moved the number.

## TL;DR

- Teacher forcing (TF) is viable for one-step prediction but fails on self-fed
  rollouts no matter how good the data or how long the training — measured, not
  theoretical. The failure is input distribution shift, a.k.a. exposure bias.
- The fixes that worked, in order of impact: oracle data (not random), balanced
  data (recentring, conceded points, flips), 4-frame history, DAgger-style input
  corruption (`--corrupt`).
- Best checkpoints so far: `tf_v5_corrupt05_11.pt` (best key control),
  `tf_v5_corrupt02_9.pt` (sharpest one-step). Both self-recover from frozen
  serve states into rallies — impossible before corruption.

## Data collection: oracle policy (`data/collection.py`)

Pong RAM map (via `env.unwrapped.ale.getRAM()`): `ram[49]`=ball_x (0 when not in
play), `ram[54]`=ball_y, `ram[51]`=player (right) paddle y, `ram[50]`=opponent.
Careful: RAM is uint8 — subtracting positions wraps; cast to int first.

Random-policy data is useless: mostly static screens, no rallies, oracle wins
236:43 by comparison. Oracle evolution:

| version | policy | result |
|---|---|---|
| v1 | always track ball y | wins 21-0 but paddle motion glued to ball → model learns tracking as physics, ignores keys; up-biased actions (68k vs 62k) |
| v2 | track incoming balls, recentre otherwise | up/down balanced; paddle y spans full range |
| v3 | + recentre dead-zone (±8), defense-gated exploration noise (held actions, ~8% of frames) | wins +21; down-from-below-center covered |
| v4 | + concede 15% of attacks (late whiff) | 277 won : 196 lost; misses and both serve directions in data |
| v5 | + score band masked (rows 0–7), horizontal flip augmentation | symmetric ball dynamics, simpler scene |

Lesson: any competent policy correlates paddle with ball; the model can ignore
the action input entirely and still ace TF loss. Data must contain off-correlation
examples (noise, deliberate misses) or control never gets learned.
Per-attack concede probability compounds over multi-attack rallies:
0.35/attack → player loses 66% of points; 0.15 → 41% lost. Aim near balance.

## Training setup

- U-Net, 4-channel frame history input (2 frames is not enough to read velocity;
  ball moves ~1px/frame), residual delta on the current frame, action embedding
  at the bottleneck. Binary frames ({0,1}), BCE with `pos_weight=10` (white
  pixels are ~2% of the image).
- Speed: batch 256, bf16 autocast, cudnn benchmark, val capped at 40 batches →
  19s/epoch teacher-forced, 71s/epoch autoregressive (seq_len 5), on an RTX A5000.
- TensorBoard runs are per-`prefix` under `runs/`.

## The exposure-bias arc (the core finding)

1. **TF-only model freezes in rollout.** One-step loss loves "copy the input"
   (the residual skip makes it free). From a serve seed, motion is exactly 0
   from step 1; from rally seeds it decays to 0 within ~50 steps.
2. **Autoregressive training fixes it** (unroll 5 steps, supervise each with GT,
   detached feedback): rollouts stay alive 200+ steps, both paddles present,
   ball in flight. Proof the failure is feedback, not capacity.
3. **But TF was preferred**, so we attacked the input side instead: balanced
   data v2–v5, 4-frame history. Result (`tf_v5_h4`): mid-rally rollouts stable,
   keys respond, one-step ball-direction agreement 99% — yet serve-state seeds
   still freeze.
4. **The decisive experiment**: train 150 epochs. Val loss bottoms at epoch 45
   (0.00055) then triples — overfitting, so one-step quality is maxed. The best
   checkpoint (`tf_v5_long_9`) still freezes solid from serve seeds (motion and
   ball mass exactly 0 for 200 steps). **Conclusion: with TF the binding
   constraint is input distribution shift, not one-step accuracy.**
5. **DAgger-style corruption** (`--corrupt p`): with probability p, train on the
   model's own snapped 1-step prediction as input (no grad), same TF loss.
   The dataset trajectory plays the expert (classic DAgger can't apply verbatim:
   the emulator can't be instantiated from dreamed pixels). p=0.2: rollouts
   self-recover from frozen serves into real rallies (~step 60–100). p=0.5:
   best control of any model (UP/DOWN/NOOP all correct) at the cost of worse
   one-step val (0.0013 vs 0.00062). Nothing survives 300 steps yet.

## Why LLMs and SARIMA get away with teacher forcing (and pixels don't)

- **LLMs**: argmax/sampling over a vocabulary is an *exact* validity filter —
  a generated token is always a legitimate input. Also exposure-bias harm is
  debated (Schmidt 2019), and LLMs do train on their own outputs in post-training.
- **SARIMA/linear AR**: least-squares on true lags is consistent for the true
  dynamics; a linear system's conditional mean stays on the trajectory manifold.
- **Pixels**: BCE/MSE predicts the per-pixel *mean of possible futures*, which is
  not a valid frame (Mathieu et al. 2016). Snapping to binary only approximately
  projects back: valid Pong frames are a tiny subset of 84×84 binary images.
  Uncertainty is real here even in deterministic Pong: serve timing, sub-pixel
  ball velocity and the opponent AI's state are invisible in 4 frames.
- **Genie** (Bruce et al. 2024): dodges it via (a) VQ discrete tokens (LLM-style
  validity filter), (b) MaskGIT training = heavy input masking (corruption
  training), (c) sampling instead of averaging. Still admits hallucinated futures.
- **Dreamer**: never rolls out pixels — rolls out a compact latent state,
  decodes only for display. The structural fix; also in PLAN.md's syllabus.
- Recent video models (Self Forcing, Diffusion Forcing) train on their own
  rollouts outright. The field converged on: better output space + sampling,
  and/or train-under-your-own-feedback. Pure TF on raw pixels is the hardest
  regime and known-failed.

## Reproduce

```bash
uv run neural-pong collect_dataset --episodes 100 --policy oracle --output data/pong_oracle_v5.npz
uv run neural-pong train_model --data data/pong_oracle_v5.npz --epochs 60 \
    --prefix tf_v5_corrupt05 --corrupt 0.5
uv run neural-pong play_terminal --checkpoint checkpoints/tf_v5_corrupt05_11.pt
```

Watch loss: `tensorboard --logdir runs`. Val typically bottoms around epoch 45;
pick the checkpoint nearest the val minimum, not the last one.

## Round 2: bounces, resets, serve recovery (v6)

Symptoms reported from play-testing `tf_v5_corrupt05_11`: ball goes crazy when
bouncing off top/bottom walls and on point resets; left paddle missing at start.

Diagnosis: in the data itself the ball **fuses with the wall** at contact — the
210×160 → 84×84 resize crushes vertical resolution 2.5:1, so the ball becomes a
tick hanging off the wall line. High-change transitions (bounces, paddle hits,
resets) are also rare (~0.3% of frames), so the model gets little gradient on
them.

Fixes:

1. **Crop before resize** (`preprocess_frame`): crop to playfield `[24:210, 8:152]`
   (walls at raw rows 24–33/194–209), then resize. Score band removed by the
   crop, mask no longer needed. ~15% more vertical resolution.
2. **Oversample high-change transitions** (`--oversample 3`): WeightedRandomSampler
   with weights ∝ 1 + 3·(frame-diff / mean). Bounces/resets drawn more often.
3. Keep `--corrupt 0.5`.

Result (`tf_v6_c05_os3_4`, val-optimal checkpoint at epoch 19 — pick by val, not
by recency):

- 600-step rollouts from 3 serve seeds: alive in every 50-step window; ball
  present 72–88% of steps. (~20s of self-fed play; v5 models died by step 300.)
- Bounce test: seeded 6 frames before a top-wall bounce, ball rebounds and
  returns (was the "goes crazy" case).
- Keys: UP −28, DOWN +5, NOOP −6 over 20 held frames.
- Quirk: the model drops the walls but plays correctly without them (bounce
  physics internalised). Terminal renderer now paints the static walls itself.

`scripts/evaluate_rollout.py` runs the whole battery against any checkpoint.

## Open issues / next dials

1. DOWN key response is weaker than UP (+5 vs −28 per 20 held frames). Next
   lever: inject the action as an input channel / at the decoder, not just the
   6×6 bottleneck.
2. Serve states still stall ~10–50 frames before the ball appears (random serve
   timing is invisible in 4 frames). More history or a stochastic head.
3. Structural fix if longer horizons are needed: Dreamer-style latent rollout
   (PLAN.md syllabus).
4. Done flags: collection truncates episodes at 2000 steps without setting
   done, so `dones` are all 0 — episode boundaries are every 2000 rows.
5. Terminal play: keypresses are held 3 fantasy frames (a 1-frame tap is too
   weak); seeding takes the last 4 real frames after ~30 FIRE steps.
