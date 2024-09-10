import torch as th
from sklearn.base import OutlierMixin
from sklearn.linear_model import LogisticRegression
from stable_baselines3 import PPO

from rlboost.env import RLBoostEnv
from rlboost.extractor import CustomFeatureExtractor
from rlboost.policy import RLBoostActorCriticPolicy


class RLBoostOutlierDetector(OutlierMixin):
    def __init__(self,
                 X_valid, y_valid,
                 agent_class=PPO,
                 steps=1e4,
                 estimator=LogisticRegression(),
                 n_epochs=10,
                 rollout_steps=256,
                 agent_batch_size=64,
                 ent_coef=1e-2,
                 vf_coef=0.5,
                 extractor_kwargs={},
                 activation_fn=th.nn.ReLU,
                 use_vf_encoding=True,
                 use_score_base=True,
                 net_arch=[{"pi": [100],
                            "vf": [100]}],
                 device="cpu"):
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.agent_class = agent_class
        self.steps = steps
        self.estimator = estimator

        self.agent_params = {
            "n_epochs": n_epochs,
            "n_steps": rollout_steps,
            "batch_size": agent_batch_size,
            "gamma": 0.0,  # Contextual multi-armed bandits problem
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "policy_kwargs": {
                "features_extractor_class": CustomFeatureExtractor,
                "features_extractor_kwargs": extractor_kwargs,
                "activation_fn": activation_fn,
                "use_vf_encoding": use_vf_encoding,
                "use_score_base": use_score_base,
                "net_arch": net_arch,
            },
            "device": device,
        }

    def fit_predict(self, X, y):
        # override for transductive outlier detectors like LocalOulierFactor
        return self.fit(X, y).predict(X, y)

    def fit(self, X, y):
        env = RLBoostEnv(X_train=X, y_train=y,
                         X_valid=self.X_valid, y_valid=self.y_valid,
                         X_test=None, y_test=None,
                         estimator=self.estimator)

        self._agent = self.agent_class(RLBoostActorCriticPolicy,
                                       env, **self.agent_params,
                                       tensorboard_log=None)
        self._agent.learn(total_timesteps=self.steps)

        return self

    def predict(self, X, y):
        data_values_dev = self.get_data_values(X, y)

        # Get data mask from data values
        activation_fn = self._agent.policy.activation_fn()
        data_mask_dev = activation_fn(data_values_dev) > 0.0
        data_mask = data_mask_dev.flatten().cpu().detach().numpy()

        data_mask = data_mask.astype(int)
        data_mask[data_mask == 0] = -1

        return data_mask

    def get_data_values(self, X, y):

        # Add fake score for observation at feature extractor

        X = th.tensor(X, device=self._agent.device)
        y = th.tensor(y, device=self._agent.device)
        fake_score = th.tensor([0.0], device=self._agent.device)

        observations = {
            "data": X[None, :, :],
            "target": y[None, :],
            "score": fake_score[None, :],
        }

        # Get data values
        latent_pi, _, _ = self._agent.policy._get_latent(observations)
        return self._agent.policy.action_net(latent_pi)
