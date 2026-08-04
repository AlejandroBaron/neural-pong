import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameEncoder(nn.Module):
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.fc = nn.Linear(256 * 5 * 5, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class WorldModel(nn.Module):
    def __init__(self, latent_dim: int = 512, num_actions: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        # Frame encoder
        self.encoder = FrameEncoder(latent_dim)

        # Action embedding
        self.action_embed = nn.Embedding(num_actions, 64)

        # Transition model
        self.fc1 = nn.Linear(latent_dim + 64, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)

        # Decoder for frame prediction
        self.decoder_conv1 = nn.ConvTranspose2d(
            latent_dim, 128, 5, stride=2, padding=2, output_padding=1
        )
        self.decoder_conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1)
        self.decoder_conv3 = nn.ConvTranspose2d(64, 32, 5, stride=2, padding=2, output_padding=1)
        self.decoder_conv4 = nn.ConvTranspose2d(32, 1, 5, stride=2, padding=2, output_padding=1)

        # Heads for reward and done
        self.reward_head = nn.Linear(latent_dim, 1)
        self.done_head = nn.Linear(latent_dim, 1)

    def encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        return self.encoder(frame)

    def forward(
        self, frame: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # frame: (B, 1, H, W)
        # action: (B,)

        # Encode frame
        latent = self.encode_frame(frame)

        # Embed action
        action_emb = self.action_embed(action)

        # Concatenate and transition
        x = torch.cat([latent, action_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Decode frame
        frame_pred = self.decode(x)

        # Predict reward and done
        reward = self.reward_head(x).squeeze(-1)
        done = torch.sigmoid(self.done_head(x)).squeeze(-1)

        return frame_pred, reward, done

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        x = latent.view(-1, self.latent_dim, 1, 1)
        x = F.relu(self.decoder_conv1(x))
        x = F.relu(self.decoder_conv2(x))
        x = F.relu(self.decoder_conv3(x))
        return torch.sigmoid(self.decoder_conv4(x))

    def predict_next(
        self, frame: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict next frame given current frame and action."""
        frame_pred, reward, done = self.forward(frame, action)
        return frame_pred, reward, done
