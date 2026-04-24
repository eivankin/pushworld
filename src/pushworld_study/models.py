from __future__ import annotations

import gymnasium as gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PushWorldCNN(BaseFeaturesExtractor):
    """Paper-inspired PushWorld CNN feature extractor for SB3 policies.

    The PushWorld paper specifies kernel sizes, strides, ReLU activations, and
    fully connected layer sizes, but not convolution channel counts. This uses
    the common Atari-style `32, 64, 64` channel schedule as an explicit working
    assumption for the first reproducible baseline.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        channels = observation_space.shape[0]
        _, height, width = observation_space.shape
        if min(height, width) < 20:
            self.cnn = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=3, stride=3),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=5, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            flattened_size = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(flattened_size, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations.float()))
