from copy import deepcopy
from math import ceil

import gym
import numpy as np
from gym.spaces import Box, Dict, MultiBinary
from sklearn.linear_model import LogisticRegression


class RLBoostEnv(gym.Env):

    def __init__(self,
                 X_train, y_train,
                 X_valid, y_valid,
                 X_test=None, y_test=None,
                 estimator=LogisticRegression(),
                 batch_size=256):
        super().__init__()

        # Datasets specs
        self.X_train = X_train.astype(float)
        self.y_train = y_train.astype(float)
        self.X_valid = X_valid.astype(float)
        self.y_valid = y_valid.astype(float)
        self.X_test = X_test.astype(float) if X_test is not None else None
        self.y_test = y_test.astype(float) if y_test is not None else None

        self.train_samples = X_train.shape[0]
        self.features = X_train.shape[1]

        # Wrapped model
        self.estimator = estimator

        # Environment Specs
        self.batch_size = min(batch_size, self.train_samples)
        self.last_reward = 0
        self.step_count = 0
        self.index = 0

        # ACTIONS SPACE
        self.action_space = MultiBinary(self.batch_size)

        # OBSERVATION SPACE
        data_shape = (self.batch_size, self.features)
        low_data = X_train.min()
        high_data = X_train.max()

        target_shape = (self.batch_size,)
        low_tgt = y_train.min()
        high_tgt = y_train.max()

        self.observation_space = Dict({"data": Box(low=low_data,
                                                   high=high_data,
                                                   shape=data_shape,
                                                   dtype=np.float64),
                                       "target": Box(low=low_tgt,
                                                     high=high_tgt,
                                                     shape=target_shape,
                                                     dtype=np.float64),
                                       "score": Box(low=0.0,
                                                    high=1.0,
                                                    shape=(1,),
                                                    dtype=np.float64)})

        # Data Specs
        self.classes = np.unique(self.y_train)

        self.data_classes = {}
        for c in self.classes:
            indexes = np.where(self.y_train == c)[0]
            proportion = indexes.shape[0]/self.y_train.shape[0]

            self.data_classes[c] = {
                "proportion": proportion,
                "indexes": indexes}

    def __call__(self):
        # Permitir que la instancia de la clase se pueda llamar como una función
        return self

    def _build_state(self):
        indexes = []
        for c in self.classes:
            proportion = self.data_classes[c]["proportion"]
            indexes_c = self.data_classes[c]["indexes"]

            cut = ceil(self.batch_size * proportion)
            indexes_c = np.random.permutation(indexes_c)[:cut]
            indexes.append(indexes_c)
        indexes = np.concatenate(indexes)
        # Random cut for matching batch size
        indexes = np.random.permutation(indexes)[:self.batch_size]

        X = self.X_train[indexes]
        Y = self.y_train[indexes]

        # BASE
        estimator_base_ = deepcopy(self.estimator)
        estimator_base_.fit(X, Y)

        self.score_base = estimator_base_.score(self.X_valid, self.y_valid)

        state = {"data": X,
                 "target": Y,
                 "score": self.score_base}

        return state, X, Y

    def _get_reward(self, action):
        X = self.last_X
        Y = self.last_Y

        # Evaluate the retrieved action
        data_values = np.array(action).flatten()

        # Cutting mask for possible padding
        data_values = data_values[:X.shape[0]]

        x_in = X[data_values == 1]
        y_in = Y[data_values == 1]

        # PROPOSED
        estimator_ = deepcopy(self.estimator)
        try:
            estimator_.fit(x_in, y_in)
        except Exception:
            score = -1.0
        else:
            score = estimator_.score(self.X_valid, self.y_valid)

        reward = score - self.score_base
        self.last_reward = reward

        return reward

    def reset(self):
        self.index = 0
        self.last_reward = 0

        state, self.last_X, self.last_Y = self._build_state()

        return state

    def step(self, action):
        reward = self._get_reward(action)
        self.last_reward = reward

        # There is only one step
        done = True
        state = self.observation_space.sample()

        return state, reward, done, {}

    def render(self, mode='console'):
        if mode == 'console':
            print(
                f"[RLBoost] Current score on step {self.step_count}: (Reward: {self.last_reward})",
            )

    def seed(self, seed):
        np.random.seed(seed)
