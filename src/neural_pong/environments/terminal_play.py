"""Play neural Pong in a terminal using curses."""

import curses
import time

import gymnasium as gym
import numpy as np
import torch

from neural_pong.config import Config
from neural_pong.environments.neural import NeuralPongEnv
from neural_pong.models.world_model import WorldModel

# Map a 2x4 braille cell dot to its Unicode bit value.
_BRAILLE_BITS = {
    (0, 0): 0x01,
    (0, 1): 0x02,
    (0, 2): 0x04,
    (0, 3): 0x40,
    (1, 0): 0x08,
    (1, 1): 0x10,
    (1, 2): 0x20,
    (1, 3): 0x80,
}


def _action_from_key(key: int) -> int:
    """Map terminal keys to Pong actions."""
    if key in (ord("w"), ord("W"), curses.KEY_UP):
        return 2  # paddle up
    if key in (ord("s"), ord("S"), curses.KEY_DOWN):
        return 3  # paddle down
    if key in (ord(" "), ord("f"), ord("F")):
        return 1  # fire / serve ball
    return 0  # noop


def _render_frame(stdscr, frame: np.ndarray, row: int = 1, col: int = 0) -> None:
    """Draw a binary 84x84 frame with Unicode braille (42x21 cells)."""
    height, width = stdscr.getmaxyx()
    out_h = 84 // 4
    out_w = 84 // 2

    start_row = row
    start_col = max(0, (width - out_w) // 2) if width > out_w else col

    for by in range(out_h):
        line_chars = []
        for bx in range(out_w):
            pattern = 0
            for dx, dy in _BRAILLE_BITS:
                y = by * 4 + dy
                x = bx * 2 + dx
                if y < frame.shape[0] and x < frame.shape[1] and frame[y, x] > 0.5:
                    pattern |= _BRAILLE_BITS[(dx, dy)]
            line_chars.append(chr(0x2800 + pattern) if pattern else " ")
        line = "".join(line_chars)
        try:
            stdscr.addstr(start_row + by, start_col, line, curses.color_pair(1))
        except curses.error:
            pass


def _play(stdscr, checkpoint: str, _data: str) -> int:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(50)
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()

    model = WorldModel(latent_dim=config.latent_dim, num_actions=config.num_actions).to(device)
    checkpoint_data = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()

    neural_env = NeuralPongEnv(model, config.frame_size)

    # The real Atari env provides the true state; the world model only has to
    # predict the next frame from that state. This avoids autoregressive drift
    # so the ball and paddles stay visible and react to controls.
    real_env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")
    current_obs, _ = real_env.reset()
    prev_obs = current_obs

    stdscr.clear()
    stdscr.addstr(
        0, 0, "Terminal Neural Pong — w/↑ up, s/↓ down, space/f fire, q quit"
    )
    stdscr.refresh()

    action = 0
    total_reward = 0.0
    frame_idx = 0
    running = True
    last_frame_time = time.time()

    while running:
        key = stdscr.getch()
        if key in (ord("q"), ord("Q"), 27):  # q or ESC
            running = False
            break

        action = _action_from_key(key)

        next_obs, _real_reward, terminated, truncated, _ = real_env.step(action)
        real_done = terminated or truncated

        # Predict the next frame from the *real* previous and current frames.
        neural_env.set_state(prev_obs, current_obs)
        neural_frame, neural_reward, _neural_done = neural_env.step(action)
        total_reward += float(neural_reward.item())
        frame_idx += 1

        if real_done:
            current_obs, _ = real_env.reset()
            prev_obs = current_obs
        else:
            prev_obs = current_obs
            current_obs = next_obs

        stdscr.erase()
        stdscr.addstr(
            0, 0, "Terminal Neural Pong — w/↑ up, s/↓ down, space/f fire, q quit"
        )
        _render_frame(stdscr, neural_frame, row=1, col=0)
        try:
            stdscr.addstr(
                stdscr.getmaxyx()[0] - 1,
                0,
                f"Frame {frame_idx}  Reward: {total_reward:.2f}  Action: {action}",
            )
        except curses.error:
            pass
        stdscr.refresh()

        # Cap roughly at 30 FPS
        elapsed = time.time() - last_frame_time
        if elapsed < 0.033:
            time.sleep(0.033 - elapsed)
        last_frame_time = time.time()

    real_env.close()
    return 0


def play_terminal(checkpoint: str, data: str = "data/pong_transitions.npz") -> int:
    """Entry point for terminal-based neural Pong."""
    return curses.wrapper(_play, checkpoint, data)
