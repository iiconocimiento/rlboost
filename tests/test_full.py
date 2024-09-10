
from contextlib import nullcontext as not_raise

import numpy as np
import pytest
import torch as th
from sklearn.linear_model import LogisticRegression
from stable_baselines3 import PPO

from rlboost.env import RLBoostEnv
from rlboost.extractor import CustomFeatureExtractor
from rlboost.policy import RLBoostActorCriticPolicy


@pytest.fixture
def env():
    # Establecemos algunas variables de prueba.
    X_train = np.random.rand(10, 10)
    y_train = np.random.choice([0, 1], size=(10,))
    X_valid = np.random.rand(50, 10)
    y_valid = np.random.choice([0, 1], size=(50,))
    X_test = np.random.rand(50, 10)
    y_test = np.random.choice([0, 1], size=(50,))
    estimator = LogisticRegression()

    # Creamos el entorno y lo devolvemos.
    env = RLBoostEnv(X_train, y_train,
                    X_valid, y_valid,
                    X_test, y_test, estimator)
    yield env
    env.close()


@pytest.mark.parametrize("use_vf_encoding, use_score_base, expectation", [
                         (True, True, not_raise()),
                         (True, False, not_raise()),
                         (False, True, not_raise()),
                         (False, False, pytest.raises(ValueError)),
                         ])
def test_full(env, use_vf_encoding, use_score_base, expectation):
    with expectation:
        # Create the agent
        static_params = {
            "n_epochs": 1,
            "gamma": 0.0,  # Contextual multi-armed bandits problem
            "policy_kwargs": {
                "features_extractor_class": CustomFeatureExtractor,
                "activation_fn": th.nn.ReLU,
                "use_vf_encoding": use_vf_encoding,
                "use_score_base": use_score_base,
                "net_arch": [{"pi": [100], "vf": [100]}]},
            "device": "cpu",
        }
        model = PPO(RLBoostActorCriticPolicy,
                    env, **static_params,
                    tensorboard_log=None)
        model.learn(total_timesteps=1)
