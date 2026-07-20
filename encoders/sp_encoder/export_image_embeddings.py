# from segment_anything import sam_model_registry, SamPredictor

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
from torchvision import transforms

import argparse
import os


def batched_nms(scores, nms_radius: int):
    """Apply SuperPoint's two-pass local non-maximum suppression."""
    if nms_radius < 0:
        raise ValueError("nms_radius must be non-negative")

    def max_pool(value):
        return F.max_pool2d(
            value,
            kernel_size=nms_radius * 2 + 1,
            stride=1,
            padding=nms_radius,
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        suppression_mask = max_pool(max_mask.float()) > 0
        suppressed_scores = torch.where(suppression_mask, zeros, scores)
        new_max_mask = suppressed_scores == max_pool(suppressed_scores)
        max_mask = max_mask | (new_max_mask & (~suppression_mask))
    return torch.where(max_mask, scores, zeros)


def select_top_k_keypoints(keypoints, scores, k):
    if k is None or int(k) >= int(keypoints.shape[0]):
        return keypoints, scores
    scores, indices = torch.topk(scores, int(k), dim=0, sorted=True)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, stride: int = 8):
    """Bilinearly sample a SuperPoint descriptor grid at image-grid indices.

    Sparse SuperPoint keypoints come from the full-resolution score grid, not
    from the stride-8 descriptor grid.  Their physical pixel centers are
    ``index + 0.5``; this is the convention used by the sparse PnP path and
    by ULF-Loc's original sparse frontend.
    """
    batch, channels, height, width = descriptors.shape
    if keypoints.shape[0] != batch:
        raise ValueError("keypoints and descriptors must have the same batch size")
    keypoints = (keypoints + 0.5) / (
        keypoints.new_tensor([width, height]) * float(stride)
    )
    grid = keypoints.mul(2.0).sub(1.0).view(batch, 1, -1, 2)
    sampled = F.grid_sample(
        descriptors,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return F.normalize(sampled.reshape(batch, channels, -1), p=2, dim=1)


class SuperPoint(nn.Module):
    def __init__(self):
        super().__init__()
        out_channels = 256

        # Kept here rather than in FeatureExtractor so the direct sparse API
        # has the same defaults as ULF-Loc's SuperPoint frontend.
        self.nms_radius = 4
        self.max_num_keypoints = None
        self.detection_threshold = 0.0
        self.remove_borders = 4

        self.transform = transforms.Grayscale(num_output_channels=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, out_channels,
            kernel_size=1, stride=1, padding=0)

        path = Path(__file__).parent / 'weights/superpoint_v1.pth'
        self.load_state_dict(torch.load(str(path)), strict=False)

        print('Loaded SuperPoint model')

    def _dense_outputs(self, x):
        """Return the native stride-8 descriptor map and full-resolution scores."""
        x = self.transform(x)
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        score_logits = self.convPb(self.relu(self.convPa(x)))
        scores = F.softmax(score_logits, dim=1)[:, :-1]
        batch, _, height, width = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(batch, height, width, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(
            batch, height * 8, width * 8
        )

        descriptors = self.convDb(self.relu(self.convDa(x)))
        descriptors = F.normalize(descriptors, p=2, dim=1)
        return descriptors, scores

    @torch.inference_mode()
    def detectAndCompute(self, x, top_k=None, detection_threshold=None):
        """Return native sparse SuperPoint descriptors for every input image.

        This intentionally does not sample the resized deployment feature
        pyramid.  It is the API used by the ULF-compatible initializer and by
        the sparse frontend parity audit.
        """
        device = next(self.parameters()).device
        descriptors_dense, scores = self._dense_outputs(x.to(device))
        threshold = (
            self.detection_threshold
            if detection_threshold is None
            else float(detection_threshold)
        )
        suppressed = batched_nms(scores, self.nms_radius)
        if self.remove_borders:
            pad = int(self.remove_borders)
            suppressed[:, :pad] = -1
            suppressed[:, :, :pad] = -1
            suppressed[:, -pad:] = -1
            suppressed[:, :, -pad:] = -1

        result = []
        for batch_index in range(suppressed.shape[0]):
            y, x_coord = torch.where(suppressed[batch_index] > threshold)
            keypoints = torch.stack([x_coord, y], dim=-1).float()
            keypoint_scores = suppressed[batch_index, y, x_coord]
            keypoints, keypoint_scores = select_top_k_keypoints(
                keypoints,
                keypoint_scores,
                self.max_num_keypoints if top_k is None else top_k,
            )
            descriptors = sample_descriptors(
                keypoints[None], descriptors_dense[batch_index : batch_index + 1]
            )[0].transpose(0, 1)
            result.append(
                {
                    "keypoints": keypoints,
                    "keypoint_scores": keypoint_scores,
                    "descriptors": descriptors,
                }
            )
        return result

    @torch.inference_mode()
    def detectAndComputeDense(self, x):
        """Return the native stride-8 descriptor map and score map."""
        device = next(self.parameters()).device
        descriptors, scores = self._dense_outputs(x.to(device))
        return descriptors, scores.unsqueeze(1)

    def forward(self, x):
        """ Compute keypoints, scores, descriptors for image """
        return self._dense_outputs(x)
    

parser = argparse.ArgumentParser(
    description=(
        "Get image embeddings of an input image or directory of images."
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where embeddings will be saved. Output will be either a folder "
        "of .pt per image or a single .pt representing image embeddings."
    ),
)


parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")


def main(args: argparse.Namespace) -> None:
    print("Loading model...")

    model = SuperPoint().cuda()

    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        print(os.listdir(args.input))
        seqs = [f for f in os.listdir(args.input) if "seq" in f and "zip" not in f]
    
    print(seqs)
    os.makedirs(args.output, exist_ok=True)


    for seq in seqs:
        targets = [
            f"{seq}/{f}" for f in os.listdir(os.path.join(args.input, seq)) if "color" in f
        ]
        targets = [os.path.join(args.input, f) for f in targets]

        output_dir = os.path.join(args.output, seq)
        os.makedirs(output_dir, exist_ok=True)

        for t in targets:
            print(f"Processing '{t}'...")
            img_name = t.split(os.sep)[-1]
            image = cv2.imread(t)
            if image is None:
                print(f"Could not load '{t}' as an image, skipping...")
                continue
            
            tensor_image = torch.from_numpy(np.array(image))
            input_image = tensor_image.to(
                device="cuda", dtype=torch.float32, non_blocking=True
            )
            input_image = input_image / 255.0
            img_features, scores = model(input_image.permute(2, 0, 1)[None])
            # print(scores.shape)

            img_features = img_features.squeeze(0) # (256, 60, 80)
            img_scores = scores[0] # (60, 80)

            torch.save(img_features, os.path.join(output_dir, f"{img_name}_fmap_CxHxW.pt"))
            torch.save(img_scores, os.path.join(output_dir, f"{img_name}_smap_CxHxW.pt"))
        

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
