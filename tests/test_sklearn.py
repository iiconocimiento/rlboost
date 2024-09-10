import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from rlboost import RLBoostOutlierDetector


@pytest.fixture
def dataset():
    # Genera un dataset de clasificación sintético para las pruebas
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2,
                               random_state=42)
    return X, y


@pytest.fixture
def outlier_detector(dataset):
    X, y = dataset
    return RLBoostOutlierDetector(X_valid=X[:100], y_valid=y[:100],
                                  estimator=LogisticRegression(),
                                  steps=1, n_epochs=1,
                                  agent_batch_size=2,
                                  rollout_steps=2)


def test_fit(dataset, outlier_detector):
    X, y = dataset

    outlier_detector.fit(X, y)
    assert outlier_detector._agent is not None


def test_predict(dataset, outlier_detector):
    X, y = dataset
    outlier_detector.fit(X, y)
    outlier_array = outlier_detector.predict(X, y)
    assert outlier_array.shape[0] == X.shape[0]
    assert np.all(np.isin(outlier_array, [-1, 1]))
