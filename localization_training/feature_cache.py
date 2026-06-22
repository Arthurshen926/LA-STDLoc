import torch
import torch.nn.functional as F


class FeatureCache:
    def __init__(self, feature_extractor, longest_edge=None, norm=True):
        self.feature_extractor = feature_extractor
        self.longest_edge = longest_edge
        self.norm = norm
        self._cache = {}

    def get(self, camera):
        key = camera.image_name
        if key in self._cache:
            return self._cache[key]
        image = camera.original_image.cuda()
        with torch.no_grad():
            feature = self.feature_extractor(image[None])["feature_map"][0]
            if self.longest_edge is not None:
                h, w = image.shape[-2:]
                scale = float(self.longest_edge) / max(h, w)
                target = (max(1, int(round(h * scale))), max(1, int(round(w * scale))))
                feature = F.interpolate(feature[None], size=target, mode="bilinear", align_corners=False)[0]
            if self.norm:
                feature = F.normalize(feature, p=2, dim=0)
        self._cache[key] = feature
        return feature
