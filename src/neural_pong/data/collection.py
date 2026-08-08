"""Collect Pong transitions."""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from neural_pong.config import Config
from neural_pong.environments.pong import create_env, preprocess_frame


def _to_binary_uint8(frame: np.ndarray) -> np.ndarray:
    """Binarise a normalised frame and store it compactly as uint8."""
    return (frame >= 0.5).astype(np.uint8) * 255


class OraclePolicy:
    """Scripted Pong player from RAM: serve when out, track incoming balls, recentre otherwise.

    RAM map: ram[49]=ball_x (0 when not in play), ram[54]=ball_y, ram[51]=player paddle y.
    Exploration noise is correlated (random action held for `hold` frames) so the
    paddle also reaches field edges; i.i.d. noise never travels far.
    A fraction of attacks is deliberately whiffed (freeze when the ball is close)
    so the data contains misses, conceded points and both serve directions;
    otherwise "paddle always reaches the ball" becomes a law of physics and the
    model ignores player input.
    """

    CENTER_Y = 109  # paddle rest position observed at episode start
    CENTER_DEADZONE = 8  # stand still near center; creeping correlates position with motion
    WHIFF_X = 90  # ball x beyond which a conceded attack freezes the paddle

    def __init__(
        self, epsilon: float = 0.01, hold: int = 8, concede_prob: float = 0.15, seed: int = 42
    ):
        self.rng = np.random.default_rng(seed)
        self.epsilon = epsilon
        self.hold = hold
        self.concede_prob = concede_prob
        self.reset()

    def reset(self) -> None:
        """Clear per-episode state (keeps the rng stream)."""
        self.prev_ball_x = 0
        self.noise_action = 0
        self.noise_left = 0
        self.in_attack = False
        self.concede = False

    def __call__(self, ram: np.ndarray) -> int:
        ball_x, ball_y, player_y = ram[49], ram[54], ram[51]
        ball_vx = int(ball_x) - int(self.prev_ball_x)  # cast: uint8 subtraction wraps
        self.prev_ball_x = ball_x

        if ball_x == 0:
            self.in_attack = False
            self.concede = False
            return 1  # FIRE to serve

        incoming = ball_vx > 0
        if incoming and not self.in_attack:
            self.concede = self.rng.random() < self.concede_prob
        self.in_attack = incoming

        # Deliberate late whiff: track normally, freeze once the ball is close.
        if self.concede and incoming and ball_x > self.WHIFF_X:
            return 0

        # No exploring when an incoming ball is close: whiffing those ends rallies.
        defend = ball_x > 100 and incoming
        if self.noise_left > 0 and not defend:
            self.noise_left -= 1
            return self.noise_action
        if not defend and self.rng.random() < self.epsilon:
            self.noise_action = int(self.rng.integers(6))
            self.noise_left = self.hold
            return self.noise_action

        # Track the ball only while it moves toward the player; recentre after,
        # but stay put when already near the centre.
        if incoming:
            target, deadzone = ball_y, 2
        else:
            target, deadzone = self.CENTER_Y, self.CENTER_DEADZONE
        if target < player_y - deadzone:
            return 2  # paddle up
        if target > player_y + deadzone:
            return 3  # paddle down
        return 0


def collect_transitions(
    episodes: int | None = None,
    output: str = "data/pong_transitions.npz",
    policy: str = "oracle",
) -> int:
    """Collect Pong dataset using an oracle (perfect) or random policy."""
    config = Config()

    env = create_env("PongNoFrameskip-v4")

    prev_frames = []
    frames = []
    next_frames = []
    actions = []
    rewards = []
    dones = []

    num_episodes = episodes or config.num_episodes
    output_path = output
    oracle = OraclePolicy() if policy == "oracle" else None

    print(f"Collecting {num_episodes} episodes with {policy} policy...")

    for _episode in tqdm(range(num_episodes), desc="Episodes"):
        obs, _ = env.reset()
        if oracle is not None:
            oracle.reset()
        done = False
        step = 0
        prev_frame = _to_binary_uint8(preprocess_frame(obs, config.frame_size))

        while not done and step < config.max_steps_per_episode:
            frame = _to_binary_uint8(preprocess_frame(obs, config.frame_size))
            if oracle is not None:
                action = oracle(env.unwrapped.ale.getRAM())
            else:
                action = env.action_space.sample()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_frame = _to_binary_uint8(preprocess_frame(next_obs, config.frame_size))

            prev_frames.append(prev_frame)
            frames.append(frame)
            next_frames.append(next_frame)
            actions.append(action)
            rewards.append(reward)
            dones.append(float(done))

            obs = next_obs
            step += 1

    env.close()

    # Convert to arrays
    prev_frames = np.array(prev_frames)
    frames = np.array(frames)
    next_frames = np.array(next_frames)
    actions = np.array(actions)
    rewards = np.array(rewards)
    dones = np.array(dones)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Compress: binary frames shrink by ~30-50x over raw float32.
    np.savez_compressed(
        output_path,
        prev_frames=prev_frames,
        frames=frames,
        actions=actions,
        next_frames=next_frames,
        rewards=rewards,
        dones=dones,
    )

    print(f"Saved {len(frames)} transitions to {output_path}")
    print(f"Frame shape: {frames.shape}")
    print(f"Actions: {np.unique(actions)}")

    return 0
