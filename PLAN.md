# Neural Pong: Learning a Game Engine with a World Model

## Goal

The goal of this project is to replace the Pong game engine with a neural network.

Instead of using the real Atari emulator, we will train a neural network that predicts how Pong evolves over time. Once trained, the player (human or RL agent) will interact only with the neural network.

This is essentially the simplest possible **world model**.

---

# Learning Goals

By the end of this project you should understand:

- What a world model is
- Model-based vs model-free reinforcement learning
- State transitions
- Autoregressive prediction
- Teacher forcing
- Exposure bias
- Rollout error accumulation
- Why latent world models exist
- Why models like Dreamer, Genie and MuZero don't predict pixels directly

---

# What is a World Model?

Most games have a simulator.

```
state_t + action
        ↓
game engine
        ↓
state_(t+1)
```

The simulator knows the physics.

A world model learns this mapping instead.

```
state_t + action
        ↓
Neural Network
        ↓
predicted state_(t+1)
```

Instead of hand-written rules, we approximate the simulator with machine learning.

---

# Project Scope

We are NOT reproducing Dreamer.

We are NOT reproducing Genie.

We are building the smallest complete world model possible.

Inputs:

- current frame
- previous frames
- player action

Outputs:

- next frame
- reward
- terminal flag

Everything else is intentionally ignored.

---

# High-Level Pipeline

```
Real Pong
      │
      ▼
Collect Gameplay Dataset
      │
      ▼
Train World Model
      │
      ▼
Neural Pong Engine
      │
      ▼
Play Inside Learned World
```

---

# Tech Stack

Python

PyTorch

Gymnasium

ALE

TensorBoard

NumPy

OpenCV

Torchvision

---

# Folder Structure

```
project/

README.md

pyproject.toml

src/neural_pong/

  cli.py

  config.py

  environments/

    pong.py

    neural.py

    inspection.py

    play.py

  models/

    world_model.py

  data/

    collection.py

    dataset.py

  training/

    train.py

  evaluation/

    evaluate.py

  visualization/

    rollouts.py

    gifs.py

tests/

data/

checkpoints/

runs/

videos/
```

---

# Stage 1 — Learn Pong

Goal:

Understand the environment.

Tasks

- install Gymnasium Atari
- run Pong
- inspect observations
- inspect action space
- inspect rewards

Deliverable

A script that displays Pong and prints:

```
frame shape

action

reward

done
```

Learn:

What is a state?

What is an action?

What is a transition?

---

# Stage 2 — Dataset Collection

Collect transitions.

Each sample should contain

```
(frame_t,
 action_t,
 frame_t+1,
 reward,
 done)
```

Frames should be

84x84 grayscale.

Store data efficiently.

Recommended:

```
npz

or

HDF5
```

Collect

100k–500k transitions.

Use

random policy

or

simple scripted policy.

Deliverable

```
data/

frames

actions

rewards

dones
```

---

# Stage 3 — Build Dataset Loader

Implement a PyTorch Dataset.

Should return

```
current frame

action

target frame

reward

done
```

Support batching.

---

# Stage 4 — First World Model

Model inputs

current frame

action

Outputs

next frame

reward

done

Architecture

Frame encoder

↓

action embedding

↓

concatenate

↓

CNN residual blocks

↓

decoder

↓

predicted next frame

Additional linear heads

↓

reward

↓

done

Loss

Frame

MSE

+

Reward

Cross entropy

+

Done

Binary cross entropy

---

# Stage 5 — Training

Teacher forcing.

Always feed the real frame.

Train until

validation loss stabilizes.

Save checkpoints.

TensorBoard:

frame loss

reward accuracy

done accuracy

sample predictions

---

# Stage 6 — Evaluation

Compute

Frame MSE

Reward accuracy

Done accuracy

Visualize

Ground truth

Prediction

Difference image

for random samples.

---

# Stage 7 — Rollout

Now stop teacher forcing.

Feed predictions back into the model.

```
real frame

↓

predict

↓

predicted frame

↓

predict again

↓

predict again

↓

predict again
```

This is called an autoregressive rollout.

Measure

1 step

5 step

20 step

50 step

100 step

errors.

Observe drift.

This is one of the biggest limitations of pixel-space world models.

---

# Stage 8 — Replace Pong

Write an environment wrapper.

Instead of

Gym

```
step(action)
```

calls

```
world_model(frame, action)
```

Everything should now run inside the neural network.

---

# Stage 9 — Interactive Demo

Keyboard

↑

↓

controls paddle.

Display

predicted frame

instead of

real emulator.

User should not notice the difference initially.

---

# Stage 10 — Visualization

Produce GIFs

Real

↓

Prediction

Difference heatmap

Long rollouts

Error vs rollout length

Loss curves

Prediction gallery

These are the main assets for the blog.

---

# Stretch Goal 1

Predict latent features instead of pixels.

Encoder

↓

latent vector

↓

transition model

↓

decoder

This introduces latent world models.

---

# Stretch Goal 2

Train a DQN

inside

the learned world.

Evaluate

real Pong performance.

This demonstrates model-based RL.

---

# Stretch Goal 3

Train using sequences

instead of

single frames.

Input

```
frame1

frame2

frame3

frame4

action
```

Predict

frame5

This helps infer velocity.

---

# Stretch Goal 4

ConvLSTM

Replace residual blocks with ConvLSTM.

Model remembers hidden state.

Produces smoother rollouts.

---

# Stretch Goal 5

Transformer

Replace CNN transition model with

Vision Transformer

or

Video Transformer.

---

# Expected Failure Modes

Blurry predictions

Ghosting

Ball disappears

Paddle jitters

Reward prediction lags

Simulation diverges

Long rollouts collapse

These are normal.

---

# Things to Read

World Models (Ha & Schmidhuber)

Dreamer

DreamerV2

DreamerV3

PlaNet

MuZero

Genie

Genie 2

VideoGPT

PredRNN

SimWorld

---

# Success Criteria

Minimum

- Collect dataset
- Train model
- Predict next frame

Good

- Play inside learned world
- Long rollouts

Excellent

- RL agent trained inside learned world
- Interactive neural Pong
- Blog with visualizations

---

# Suggested Timeline

Day 1

Learn Gymnasium

Collect data

Day 2

Dataset loader

Baseline CNN

Day 3

Training

Evaluation

Day 4

Autoregressive rollouts

Visualization

Day 5

Neural Pong engine

Interactive demo

Day 6+

Improvements

Latent models

RL inside world model

---

# Key Concepts (Cheat Sheet)

**State**
Current observation of the environment.

**Action**
Decision taken by the player or agent.

**Transition**
How the environment changes after an action.

**Dynamics Model**
A function that predicts the next state.

**World Model**
A learned dynamics model.

**Teacher Forcing**
Training with the real previous state rather than the model's own prediction.

**Autoregressive Rollout**
Using the model's prediction as the next input repeatedly.

**Exposure Bias**
The model is trained on perfect history but tested on its own imperfect predictions.

**Model-Based RL**
An RL agent learns or uses a model of the environment.

**Model-Free RL**
An RL agent learns directly from experience without modeling the environment.

**Latent Space**
A compressed representation of the observation that preserves useful information.

---

# Final Deliverables

- Modular PyTorch implementation
- Trained world model checkpoint
- Interactive neural Pong
- Rollout visualizer
- Evaluation scripts
- GIF generation
- TensorBoard logs
- Blog-ready figures
- README explaining every component