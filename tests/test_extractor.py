import torch as th

from rlboost.extractor import CustomFeatureExtractor


def test_forward():
    # Test the forward method of CustomFeatureExtractor
    observation_space = {
        "data": th.Tensor(1, 5, 3),
        "target": th.Tensor(1, 5),
        "score": th.Tensor([1]),
    }
    features_dim = 7

    feature_extractor = CustomFeatureExtractor(
        observation_space, features_dim=features_dim)
    observations = {
        "data": th.randn(1, 5, 3),
        "target": th.randn(1, 5),
        "score": th.Tensor([0.8]),
    }
    features = feature_extractor.forward(observations)
    assert features.shape == (1, 5, features_dim)
