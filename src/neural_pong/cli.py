import jsonargparse

from neural_pong.data.collection import collect_transitions
from neural_pong.diagnostics import run_model_self_test
from neural_pong.environments.inspection import inspect_environment
from neural_pong.environments.play import play_neural_pong
from neural_pong.environments.terminal_play import play_terminal
from neural_pong.evaluation.evaluate import evaluate_model
from neural_pong.training.train import train_model
from neural_pong.visualization.gifs import generate_gifs
from neural_pong.visualization.rollouts import visualize_rollouts


class CLI:
    def collect_dataset(
        self,
        episodes: int | None = None,
        output: str = "data/pong_transitions.npz",
        policy: str = "oracle",
    ) -> int:
        return collect_transitions(episodes, output, policy)

    def evaluate_model(
        self,
        checkpoint: str,
        data: str = "data/pong_transitions.npz",
        output: str = "eval/",
    ) -> int:
        return evaluate_model(checkpoint, data, output)

    def generate_gifs(
        self,
        checkpoint: str,
        data: str = "data/pong_transitions.npz",
        output: str = "gifs/",
        lengths: list[int] | None = None,
        samples: int = 6,
    ) -> int:
        return generate_gifs(checkpoint, data, output, lengths, samples)

    def inspect_environment(self) -> int:
        return inspect_environment()

    def play_neural_pong(self, checkpoint: str) -> int:
        return play_neural_pong(checkpoint)

    def play_terminal(self, checkpoint: str, data: str = "data/pong_transitions.npz") -> int:
        return play_terminal(checkpoint, data)

    def run_model_self_test(self) -> int:
        return run_model_self_test()

    def train_model(
        self,
        data: str = "data/pong_transitions.npz",
        resume: str | None = None,
        autoregressive: bool = True,
        seq_len: int = 5,
        prefix: str = "world_model",
        epochs: int | None = None,
    ) -> int:
        return train_model(data, resume, autoregressive, seq_len, prefix, epochs)

    def visualize_rollouts(
        self,
        checkpoint: str,
        data: str = "data/pong_transitions.npz",
        output: str = "videos/",
        lengths: list[int] | None = None,
        samples: int = 5,
    ) -> int:
        return visualize_rollouts(checkpoint, data, output, lengths, samples)


def main() -> None:
    jsonargparse.auto_cli(CLI, as_positional=False)


if __name__ == "__main__":
    main()
