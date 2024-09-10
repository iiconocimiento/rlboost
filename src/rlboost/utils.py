from copy import deepcopy

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import EvalCallback


def eval_agent(env, agent):
    scores = {}

    X_train, y_train = env.X_train, env.y_train
    X_valid, y_valid = env.X_valid, env.y_valid
    X_test, y_test = env.X_test, env.y_test

    # Add fake score for observation at feature extractor
    fake_score = np.zeros(X_train.shape[0])[:, None]
    registers = np.concatenate((X_train,
                                y_train[:, None],
                                fake_score),
                               axis=1)

    # Get data values
    registers_dev = th.tensor(registers, device=agent.device)[:, None, :]
    latent_pi, _, _ = agent.policy._get_latent(registers_dev)
    data_values_dev = agent.policy.action_net(latent_pi)
    data_values = data_values_dev.flatten().cpu().detach().numpy()
    scores["data_values"] = data_values

    # Get data mask from data values
    activation_fn = agent.policy.activation_fn()
    data_mask_dev = activation_fn(data_values_dev) > 0.0
    data_mask = data_mask_dev.flatten().cpu().detach().numpy()
    scores["data_mask"] = data_mask

    # SCORE BASELINE
    base_estimator = deepcopy(env.estimator)
    base_estimator.fit(X_train, y_train)

    scores["score_base_val"] = base_estimator.score(X_valid, y_valid)
    scores["score_base_test"] = base_estimator.score(X_test, y_test)

    # SCORES AGENT
    x_in = X_train[data_mask]
    y_in = y_train[data_mask]

    estimator = deepcopy(env.estimator)

    try:
        estimator.fit(x_in, y_in)
    except Exception:
        scores["score_val"] = -1.0
        scores["score_test"] = -1.0
    else:
        scores["score_val"] = estimator.score(X_valid, y_valid)
        scores["score_test"] = estimator.score(X_test, y_test)

    # EXTRA DATA
    total_data = X_train.shape[0]
    data_remained = data_mask.sum()
    scores["dropped_ratio"] = (total_data - data_remained)/total_data

    return scores


class ScoreCallbackWithPatience(EvalCallback):

    def __init__(self, patience_steps, **kwargs):
        super().__init__(**kwargs)
        self.patience_steps = patience_steps
        self.trigger_times = 0
        self.best_reward = -np.inf

    def _on_step(self) -> bool:
        if not super()._on_step():
            return False
        if not (self.eval_freq > 0 and self.n_calls % self.eval_freq == 0):
            return True

        env = deepcopy(self.eval_env).envs[0]

        scores = eval_agent(env, self.model)

        self.logger.record("eval/dropped_data_ratio", scores["dropped_ratio"])
        self.logger.record("eval/score_base_val", scores["score_base_val"])
        self.logger.record("eval/score_base_test", scores["score_base_test"])
        self.logger.record("eval/score_val", scores["score_val"])
        self.logger.record("eval/score_test", scores["score_test"])
        self.logger.record("eval/num_fitted_estimators",
                           env.num_fitted_estimators)

        continue_training = True
        actual_mean_reward = np.array(self.evaluations_results).mean()
        if self.patience_steps:
            # Early stopping mechanism
            if self.best_reward >= actual_mean_reward:
                self.trigger_times += 1
                if self.trigger_times >= self.patience_steps:
                    continue_training = False
            else:
                self.best_reward = actual_mean_reward
                self.trigger_times = 0

            if self.verbose > 0 and not continue_training:
                print("Early Stopping done")

        self.logger.record("eval/early_stopping", not (continue_training))

        return continue_training
