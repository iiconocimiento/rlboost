
import torch as th
from torch import nn


class CustomNetwork(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 512,
        last_layer_dim_vf: int = 512,
        use_vf_encoding: bool = True,
        use_score_base: bool = True,
    ) -> None:
        super().__init__()

        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.use_vf_encoding = use_vf_encoding
        self.use_score_base = use_score_base

        register_size = feature_dim-1  # The last one is the score_base

        # Actor
        self.policy_net = nn.Sequential(
            nn.Linear(register_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, last_layer_dim_pi),
            nn.ReLU(),
        )

        # Critic
        # Do not use TransformerEncoder
        if not self.use_vf_encoding:
            if not self.use_score_base:
                raise ValueError()
            self.value_net = nn.Sequential(
                nn.Linear(1, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, last_layer_dim_vf),
                nn.ReLU())
            return

        # Use TransformerEncoder
        self.cls = nn.Parameter(th.ones(register_size), requires_grad=True)

        self.vf_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=register_size, nhead=2),
            num_layers=4,
            norm=nn.LayerNorm(register_size))

        dim_in = register_size
        if self.use_score_base:
            dim_in += 1

        self.value_net = nn.Sequential(
            nn.Linear(dim_in, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, last_layer_dim_vf),
            nn.ReLU(),
        )

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        features_val = features[:, :, :-1]
        features_pol = features[:, :, :-1]
        score_base = features[:, :, -1:]
        score_base = th.mean(score_base, 1, True)
        num_regs = features.shape[0]

        # Policy
        policy_values = self.policy_net(features_pol)

        if not self.use_vf_encoding:
            # Values without encoding
            value_vector = self.value_net(score_base)
            return policy_values, value_vector

        # Values with encoding
        cls = self.cls[None, None, :]
        cls = cls.repeat(num_regs, 1, 1)

        x = th.cat((cls, features_val), axis=1)
        x = self.vf_encoder(x)
        cls_out = x[:, 0:1, :]

        if self.use_score_base:
            value_vector = th.cat((cls_out, score_base), 2)
        else:
            value_vector = cls_out

        value_vector = self.value_net(value_vector)

        return policy_values, value_vector
