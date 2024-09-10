

import gym
import torch as th
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    Distribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from rlboost.network import CustomNetwork


class RLBoostActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch=None,
        activation_fn: type[nn.Module] = nn.Tanh,
        use_vf_encoding: bool = True,
        use_score_base: bool = True,
        *args,
        **kwargs,
    ) -> None:
        self.use_vf_encoding = use_vf_encoding
        self.use_score_base = use_score_base
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

        self._build(lr_schedule)
        self.action_net = nn.Sequential(
            nn.Linear(self.mlp_extractor.latent_dim_pi, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(
            self.features_dim,
            use_vf_encoding=self.use_vf_encoding,
            use_score_base=self.use_score_base)

    def _get_action_dist_from_latent(self,
                                     latent_pi: th.Tensor,
                                     latent_sde: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        mean_actions = mean_actions.permute(0, 2, 1)[:, 0]

        dist = self.action_dist
        if isinstance(dist, BernoulliDistribution):
            # Here mean_actions are the logits
            return dist.proba_distribution(action_logits=mean_actions)

        raise ValueError("Invalid action distribution")
