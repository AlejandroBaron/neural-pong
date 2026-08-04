import torch

from neural_pong.models import WorldModel


def test_forward_shapes():
    model = WorldModel(latent_dim=128, num_actions=4)
    frame = torch.rand(2, 1, 84, 84)
    action = torch.tensor([0, 1])

    predicted_frame, reward, done = model(frame, action)

    assert predicted_frame.shape == frame.shape
    assert reward.shape == (2,)
    assert done.shape == (2,)
