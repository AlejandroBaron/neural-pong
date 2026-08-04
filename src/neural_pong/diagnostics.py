"""Run a lightweight model self-test."""


def run_model_self_test() -> int:
    try:
        import torch

        from neural_pong.models.world_model import WorldModel

        print("Creating model...")
        model = WorldModel(latent_dim=128, num_actions=4)

        print("Testing forward pass...")
        frame = torch.rand(2, 1, 84, 84)
        action = torch.tensor([0, 1])

        pred_frame, reward, done = model(frame, action)

        print(f"  Input frame shape: {frame.shape}")
        print(f"  Predicted frame shape: {pred_frame.shape}")
        print(f"  Reward shape: {reward.shape}")
        print(f"  Done shape: {done.shape}")
        print("Model forward pass OK")

        print("\nTesting autoregressive prediction...")
        current = frame
        for _ in range(5):
            current, _reward, _done = model.predict_next(current, action)
        print(f"  After 5 steps: frame shape {current.shape}")
        print("Autoregressive prediction OK")

        return 0

    except ImportError as e:
        print(f"Cannot test: {e}")
        print("Install dependencies first: uv sync")
        return 1
