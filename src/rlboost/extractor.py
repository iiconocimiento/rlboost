
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: dict[str, th.Tensor],
                 features_dim: int = 257) -> None:
        super().__init__(
            observation_space, features_dim)

        data_space = observation_space["data"]
        observation_space["target"]

        # We assume NxM datasets
        self.N = data_space.shape[0]
        self.M = data_space.shape[-1] + 1  # Adding the target

        self.linear_regs = nn.Sequential(
            nn.Linear(self.M, features_dim-1),
            nn.ReLU())

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        device = observations["data"].device

        data = observations["data"]
        target = observations["target"]
        score_base = observations["score"]

        score_array = th.ones(observations["data"].shape[0:2],
                              device=device) * score_base
        x = th.cat((data, target[:, :, None]), 2)
        x = self.linear_regs(x)
        x = th.cat((x, score_array[:, :, None]), 2)
        return x
