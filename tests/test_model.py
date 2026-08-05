import torch

from neural_pong.models import WorldModel


def test_forward_shapes():
    model = WorldModel(latent_dim=128, num_actions=4)
    frame = torch.rand(2, 2, 84, 84)
    action = torch.tensor([0, 1])

    predicted_frame, reward, done = model(frame, action)

    assert predicted_frame.shape == (2, 1, 84, 84)
    assert reward.shape == (2,)
    assert done.shape == (2,)


def test_predict_next_returns_hard_palette_values():
    model = WorldModel(latent_dim=128, num_actions=4)
    with torch.no_grad():
        model.decoder_conv4.weight.zero_()
        model.decoder_conv4.bias.fill_(2.0)

    frame = torch.rand(1, 2, 84, 84)
    action = torch.tensor([0])

    logits, _, _ = model(frame, action)
    predicted_frame, _, _ = model.predict_next(frame, action)

    assert logits.max() > 1
    assert torch.equal(predicted_frame, torch.ones_like(predicted_frame))
