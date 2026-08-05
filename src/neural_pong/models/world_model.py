import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldModel(nn.Module):
    """U-Net world model with action-conditioned spatial transition.

    Input: two-channel frame stack [previous_frame, current_frame] and action.
    Output: predicted next frame logits, reward, done probability.
    """

    def __init__(self, latent_dim: int = 256, num_actions: int = 6):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions

        # Encoder: 84 -> 42 -> 21 -> 11 -> 6
        self.conv1 = nn.Conv2d(2, 32, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(128, latent_dim, 5, stride=2, padding=2)

        # Action embedding broadcast into the spatial bottleneck
        self.action_embed = nn.Embedding(num_actions, 64)

        # Transition model in latent space
        self.trans1 = nn.Conv2d(latent_dim + 64, latent_dim, 3, padding=1)
        self.trans2 = nn.Conv2d(latent_dim, latent_dim, 3, padding=1)

        # Decoder with skip connections
        self.up1 = nn.ConvTranspose2d(latent_dim, 128, 5, stride=2, padding=2)
        self.reduce1 = nn.Conv2d(128 + 128, 128, 3, padding=1)

        self.up2 = nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2)
        self.reduce2 = nn.Conv2d(64 + 64, 64, 3, padding=1)

        self.up3 = nn.ConvTranspose2d(64, 32, 5, stride=2, padding=2, output_padding=1)
        self.reduce3 = nn.Conv2d(32 + 32, 32, 3, padding=1)

        # Raw logit delta; current frame is added as a residual skip
        self.decoder_conv4 = nn.ConvTranspose2d(32, 1, 5, stride=2, padding=2, output_padding=1)

        # Reward and done heads from the transition bottleneck
        self.reward_head = nn.Linear(latent_dim, 1)
        self.done_head = nn.Linear(latent_dim, 1)

    def encode(self, frame: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode frame stack and return bottleneck plus skip features."""
        e1 = F.relu(self.conv1(frame))
        e2 = F.relu(self.conv2(e1))
        e3 = F.relu(self.conv3(e2))
        e4 = F.relu(self.conv4(e3))
        return e4, [e1, e2, e3]

    def transition(self, bottleneck: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Apply action-conditioned transition in latent space."""
        batch_size = bottleneck.size(0)
        action_emb = self.action_embed(action).view(batch_size, 64, 1, 1)
        action_map = action_emb.expand(-1, -1, bottleneck.size(2), bottleneck.size(3))

        x = torch.cat([bottleneck, action_map], dim=1)
        x = F.relu(self.trans1(x))
        return self.trans2(x)

    def decode(
        self, x: torch.Tensor, skips: list[torch.Tensor], current_frame: torch.Tensor
    ) -> torch.Tensor:
        """Decode bottleneck back to frame logits with residual current-frame skip."""
        e1, e2, e3 = skips

        d1 = F.relu(self.up1(x))
        d1 = torch.cat([d1, e3], dim=1)
        d1 = F.relu(self.reduce1(d1))

        d2 = F.relu(self.up2(d1))
        d2 = torch.cat([d2, e2], dim=1)
        d2 = F.relu(self.reduce2(d2))

        d3 = F.relu(self.up3(d2))
        d3 = torch.cat([d3, e1], dim=1)
        d3 = F.relu(self.reduce3(d3))

        # Residual logit: network predicts a delta on top of the current frame
        return self.decoder_conv4(d3) + current_frame

    def forward(
        self, frame: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # frame: (B, 2, H, W)
        current_frame = frame[:, -1:, :, :]

        bottleneck, skips = self.encode(frame)
        latent = self.transition(bottleneck, action)

        frame_pred = self.decode(latent, skips, current_frame)

        pooled = F.adaptive_avg_pool2d(latent, 1).view(frame.size(0), -1)
        reward = self.reward_head(pooled).squeeze(-1)
        done = torch.sigmoid(self.done_head(pooled)).squeeze(-1)

        return frame_pred, reward, done

    @staticmethod
    def hard_quantize(frame_logits: torch.Tensor) -> torch.Tensor:
        """Convert binary-palette logits to hard pixel values."""
        palette_logits = torch.cat((-frame_logits, frame_logits), dim=1)
        palette = torch.tensor((0.0, 1.0), device=frame_logits.device)
        return palette[palette_logits.argmax(dim=1)].unsqueeze(1)

    def predict_next(
        self, frame: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict the next frame using hard palette feedback."""
        frame_logits, reward, done = self.forward(frame, action)
        return self.hard_quantize(frame_logits), reward, done
