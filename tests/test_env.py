import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from stable_baselines3.common.vec_env import SubprocVecEnv

from rlboost.env import RLBoostEnv


# Creamos nuestra fixture de pytest.
@pytest.fixture
def env():
    # Establecemos algunas variables de prueba.
    X_train = np.random.rand(100, 10)
    y_train = np.random.choice([0, 1], size=(100,))
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


def test_environment_reset(env):
    state = env.reset()

    # Test that the reset function returns a valid state
    assert isinstance(state, dict)
    assert state["data"].shape == (env.batch_size, env.X_train.shape[1])
    assert state["target"].shape == (env.batch_size, )


def test_environment_step(env):
    _ = env.reset()
    action = env.action_space.sample()

    _, reward, done, info = env.step(action)
    # Test that the step function returns valid values
    assert done
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


# def test_environment_check_env(env):


def test_vectorizable(env):
    env = SubprocVecEnv([env for _ in range(5)])
