import torch
import torch.nn as nn

from localization_training.sparse_frontend import simple_nms

class KpDetector(torch.nn.Module):
    def __init__(self, in_dim, matchability_head=False, offset_head=False, max_offset=2.0):
        super(KpDetector, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_dim, 128, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(32, 1, 3, 1, 1)
        )
        self.matchability_head = (
            nn.Conv2d(32, 1, 3, 1, 1) if bool(matchability_head) else None
        )
        self.offset_head = nn.Conv2d(32, 2, 3, 1, 1) if bool(offset_head) else None
        self.max_offset = float(max_offset)
        if self.matchability_head is not None:
            self.initialize_matchability_from_keypoint()
        if self.offset_head is not None:
            self.initialize_offset_to_zero()
        self.sigmoid = nn.Sigmoid()

    @property
    def has_matchability_head(self):
        return self.matchability_head is not None

    @property
    def has_offset_head(self):
        return self.offset_head is not None

    def initialize_matchability_from_keypoint(self):
        if self.matchability_head is None:
            return
        with torch.no_grad():
            self.matchability_head.weight.copy_(self.cnn[-1].weight)
            self.matchability_head.bias.copy_(self.cnn[-1].bias)

    def initialize_offset_to_zero(self):
        if self.offset_head is None:
            return
        with torch.no_grad():
            self.offset_head.weight.zero_()
            self.offset_head.bias.zero_()

    def forward_all(self, feat_map):
        hidden = self.cnn[:-1](feat_map)
        keypoint = self.sigmoid(self.cnn[-1](hidden))
        matchability = (
            self.sigmoid(self.matchability_head(hidden))
            if self.matchability_head is not None
            else keypoint
        )
        offset = (
            torch.tanh(self.offset_head(hidden)) * self.max_offset
            if self.offset_head is not None
            else None
        )
        return keypoint, matchability, offset

    def forward_heads(self, feat_map):
        keypoint, matchability, _ = self.forward_all(feat_map)
        return keypoint, matchability

    def forward(self, feat_map):
        keypoint, _ = self.forward_heads(feat_map)
        return keypoint

    def forward_combined(self, feat_map):
        keypoint, matchability = self.forward_heads(feat_map)
        return torch.sqrt((keypoint * matchability).clamp_min(0.0))
