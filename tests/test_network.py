from contextlib import nullcontext as not_raise

import pytest
import torch as th

from rlboost.network import CustomNetwork


@pytest.mark.parametrize("use_vf_encoding, use_score_base, expectation", [
                         (True, True, not_raise()),
                         (True, False, not_raise()),
                         (False, True, not_raise()),
                         (False, False, pytest.raises(ValueError)),
                         ])
def test_network_shapes(use_vf_encoding, use_score_base, expectation):
    batch_size, seq_len, feature_dim = 32, 10, 257

    obs_space = th.ones(batch_size, seq_len, feature_dim)
    with expectation:
        network = CustomNetwork(feature_dim,
                                use_vf_encoding=use_vf_encoding,
                                use_score_base=use_score_base)
        policy_output, value_output = network.forward(obs_space)

        # Test policy output shape
        assert policy_output.shape == th.Size(
            [batch_size, seq_len, network.latent_dim_pi])

        # Test value output shape
        assert value_output.shape == th.Size(
            [batch_size, 1, network.latent_dim_vf])
