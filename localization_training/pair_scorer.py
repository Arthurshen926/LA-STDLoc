import math

import torch
from torch import nn


class SparsePairScorer(nn.Module):
    def __init__(
        self,
        input_dim=6,
        hidden_dim=16,
        cosine_bias=0.65,
        cosine_scale=10.0,
        architecture="cosine_residual_v1",
        descriptor_dim=0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.architecture = str(architecture)
        self.descriptor_dim = int(descriptor_dim)
        if self.architecture not in {
            "cosine_residual_v1",
            "descriptor_set_residual_v2",
        }:
            raise ValueError(f"Unsupported pair scorer architecture: {self.architecture}")
        if self.architecture == "descriptor_set_residual_v2" and self.descriptor_dim <= 0:
            raise ValueError("descriptor_set_residual_v2 requires descriptor_dim > 0")
        self.cosine_bias = nn.Parameter(torch.tensor(float(cosine_bias)))
        self.log_cosine_scale = nn.Parameter(
            torch.tensor(math.log(max(float(cosine_scale), 1e-6)))
        )
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.descriptor_network = None
        if self.architecture == "descriptor_set_residual_v2":
            self.descriptor_network = nn.Sequential(
                nn.Linear(3 * self.descriptor_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 1),
            )
            nn.init.zeros_(self.descriptor_network[-1].weight)
            nn.init.zeros_(self.descriptor_network[-1].bias)

    def forward(
        self,
        pair_features,
        query_descriptors=None,
        landmark_descriptors=None,
        global_query_descriptor=None,
    ):
        if pair_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"pair feature dimension must be {self.input_dim}, got {pair_features.shape[-1]}"
            )
        scale = self.log_cosine_scale.exp().clamp(1e-3, 1e3)
        cosine_logit = scale * (pair_features[:, 0] - self.cosine_bias)
        residual = self.network(pair_features).squeeze(-1)
        if self.descriptor_network is not None:
            if query_descriptors is None or landmark_descriptors is None:
                raise ValueError(
                    "descriptor_set_residual_v2 requires pair query/landmark descriptors"
                )
            if query_descriptors.shape != landmark_descriptors.shape:
                raise ValueError("pair query and landmark descriptors must have identical shapes")
            if query_descriptors.shape[-1] != self.descriptor_dim:
                raise ValueError(
                    f"descriptor dimension must be {self.descriptor_dim}, "
                    f"got {query_descriptors.shape[-1]}"
                )
            if global_query_descriptor is None:
                global_query_descriptor = query_descriptors.mean(dim=0)
            global_query_descriptor = global_query_descriptor.reshape(1, -1)
            if global_query_descriptor.shape[-1] != self.descriptor_dim:
                raise ValueError("global query descriptor has the wrong dimension")
            descriptor_features = torch.cat(
                [
                    query_descriptors * landmark_descriptors,
                    torch.abs(query_descriptors - landmark_descriptors),
                    global_query_descriptor * landmark_descriptors,
                ],
                dim=1,
            )
            residual = residual + self.descriptor_network(
                descriptor_features
            ).squeeze(-1)
        return cosine_logit + residual

    def export_config(self):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "architecture": self.architecture,
            "descriptor_dim": self.descriptor_dim,
        }
