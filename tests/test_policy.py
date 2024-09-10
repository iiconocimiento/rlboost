import gym
import pytest
import torch as th
from stable_baselines3.common.distributions import BernoulliDistribution
from stable_baselines3.common.utils import get_schedule_fn
from torch import nn

from rlboost.network import CustomNetwork
from rlboost.policy import RLBoostActorCriticPolicy


@pytest.fixture
def policy():
    policy = RLBoostActorCriticPolicy(
        observation_space=gym.spaces.Box(low=-1, high=1, shape=(5,)),
        action_space=gym.spaces.Discrete(2),
        lr_schedule=get_schedule_fn(0.1),
        net_arch=[256, 128],
        activation_fn=nn.ReLU,
        use_vf_encoding=True,
        use_score_base=True,
    )
    policy.lr_schedule = policy._dummy_schedule  # Real schedule
    yield policy


def test_get_action_dist_from_latent(policy):
    # Set Bernoulli distribution for testing
    policy.action_dist = BernoulliDistribution(1)
    # Test with random latent codes
    latent_pi = th.randn(5, 1, policy.mlp_extractor.latent_dim_pi)
    latent_sde = th.randn(5, 1, policy.mlp_extractor.latent_dim_vf)
    action_dist = policy._get_action_dist_from_latent(
        latent_pi, latent_sde)
    assert isinstance(action_dist, BernoulliDistribution)


def test_build_mlp_extractor(policy):
    policy._build_mlp_extractor()
    assert isinstance(policy.mlp_extractor, CustomNetwork)
