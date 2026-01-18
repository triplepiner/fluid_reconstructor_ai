"""
Feature extraction using SuperPoint and classical methods.

Extracts keypoints and descriptors for camera calibration and matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
import cv2
from tqdm import tqdm


@dataclass
class Features:
    """Container for detected features."""
    keypoints: torch.Tensor  # (N, 2) - (x, y) pixel coordinates
    descriptors: torch.Tensor  # (N, D) - descriptor vectors
    scores: Optional[torch.Tensor] = None  # (N,) - confidence scores

    def __len__(self):
        return self.keypoints.shape[0]

    def to(self, device: str) -> "Features":
        """Move features to device."""
        return Features(
            keypoints=self.keypoints.to(device),
            descriptors=self.descriptors.to(device),
            scores=self.scores.to(device) if self.scores is not None else None
        )


class FeatureExtractor:
    """
    Extract features from images using SuperPoint or classical methods.
    """

    def __init__(
        self,
        method: str = "superpoint",
        device: str = "cuda",
        max_keypoints: int = 2048,
        nms_radius: int = 4,
        keypoint_threshold: float = 0.005
    ):
        """
        Initialize the feature extractor.

        Args:
            method: Feature extraction method ('superpoint', 'sift', 'orb')
            device: Device for SuperPoint inference
            max_keypoints: Maximum number of keypoints to extract
            nms_radius: Non-maximum suppression radius for SuperPoint
            keypoint_threshold: Detection threshold for SuperPoint
        """
        self.method = method
        self.device = device
        self.max_keypoints = max_keypoints
        self.nms_radius = nms_radius
        self.keypoint_threshold = keypoint_threshold
        self.model = None

        if method == "superpoint":
            self._init_superpoint()
        elif method == "sift":
            self._init_sift()
        elif method == "orb":
            self._init_orb()
        else:
            raise ValueError(f"Unknown method: {method}")

    def _init_superpoint(self):
        """Initialize SuperPoint model."""
        # Try to use kornia's SuperPoint implementation
        try:
            from kornia.feature import SuperPoint as KorniaSuperPoint
            self.model = KorniaSuperPoint().to(self.device)
            self._superpoint_backend = "kornia"
        except ImportError:
            # Fallback to our own implementation
            self.model = SuperPointNet(self.device)
            self._superpoint_backend = "custom"

    def _init_sift(self):
        """Initialize SIFT detector."""
        self.sift = cv2.SIFT_create(nfeatures=self.max_keypoints)

    def _init_orb(self):
        """Initialize ORB detector."""
        self.orb = cv2.ORB_create(nfeatures=self.max_keypoints)

    def extract(self, image: Union[torch.Tensor, np.ndarray]) -> Features:
        """
        Extract features from an image.

        Args:
            image: Input image, shape (H, W, 3) or (H, W), values in [0, 1] or [0, 255]

        Returns:
            Features object containing keypoints and descriptors
        """
        if self.method == "superpoint":
            return self._extract_superpoint(image)
        elif self.method == "sift":
            return self._extract_sift(image)
        elif self.method == "orb":
            return self._extract_orb(image)

    def extract_sequence(
        self,
        frames: torch.Tensor,
        step: int = 1
    ) -> List[Features]:
        """
        Extract features from a sequence of frames.

        Args:
            frames: Sequence of frames, shape (N, H, W, 3)
            step: Extract features every `step` frames

        Returns:
            List of Features objects
        """
        features_list = []
        indices = range(0, frames.shape[0], step)

        for i in tqdm(indices, desc="Extracting features"):
            features = self.extract(frames[i])
            features_list.append(features)

        return features_list

    def _extract_superpoint(self, image: Union[torch.Tensor, np.ndarray]) -> Features:
        """Extract features using SuperPoint."""
        # Prepare image
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()

        if image.dim() == 3 and image.shape[-1] == 3:
            # Convert RGB to grayscale
            gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        else:
            gray = image

        if gray.max() > 1.0:
            gray = gray / 255.0

        # Add batch dimension
        gray = gray.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, H, W)

        with torch.no_grad():
            if self._superpoint_backend == "kornia":
                # Kornia SuperPoint returns dict
                output = self.model(gray)
                keypoints = output.keypoints[0]  # (N, 2)
                descriptors = output.descriptors[0]  # (N, 256)
                scores = output.scores[0] if hasattr(output, 'scores') else None
            else:
                keypoints, descriptors, scores = self.model(gray)

        return Features(
            keypoints=keypoints.cpu(),
            descriptors=descriptors.cpu(),
            scores=scores.cpu() if scores is not None else None
        )

    def _extract_sift(self, image: Union[torch.Tensor, np.ndarray]) -> Features:
        """Extract features using SIFT."""
        # Convert to numpy uint8
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        keypoints, descriptors = self.sift.detectAndCompute(gray, None)

        if len(keypoints) == 0:
            return Features(
                keypoints=torch.zeros(0, 2),
                descriptors=torch.zeros(0, 128),
                scores=torch.zeros(0)
            )

        # Convert to tensors
        kpts = torch.tensor([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        descs = torch.from_numpy(descriptors).float()
        scores = torch.tensor([kp.response for kp in keypoints])

        return Features(keypoints=kpts, descriptors=descs, scores=scores)

    def _extract_orb(self, image: Union[torch.Tensor, np.ndarray]) -> Features:
        """Extract features using ORB."""
        # Convert to numpy uint8
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if len(keypoints) == 0 or descriptors is None:
            return Features(
                keypoints=torch.zeros(0, 2),
                descriptors=torch.zeros(0, 32),
                scores=torch.zeros(0)
            )

        # Convert to tensors
        kpts = torch.tensor([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        descs = torch.from_numpy(descriptors).float()
        scores = torch.tensor([kp.response for kp in keypoints])

        return Features(keypoints=kpts, descriptors=descs, scores=scores)


class SuperPointNet(nn.Module):
    """
    SuperPoint feature detector and descriptor.

    Simplified implementation based on the original paper:
    "SuperPoint: Self-Supervised Interest Point Detection and Description"
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device

        # Shared encoder
        self.encoder = nn.Sequential(
            # conv1
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # conv2
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # conv3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # conv4
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Detector head
        self.detector = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 65, 1)  # 64 + 1 dustbin
        )

        # Descriptor head
        self.descriptor = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1)
        )

        self.to(device)

    def forward(
        self,
        image: torch.Tensor,
        nms_radius: int = 4,
        threshold: float = 0.005,
        max_keypoints: int = 2048
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            image: Input image, shape (B, 1, H, W)
            nms_radius: Non-maximum suppression radius
            threshold: Detection threshold
            max_keypoints: Maximum number of keypoints

        Returns:
            Tuple of (keypoints, descriptors, scores)
        """
        B, _, H, W = image.shape

        # Encoder
        features = self.encoder(image)

        # Detector
        det = self.detector(features)
        det = F.softmax(det, dim=1)
        det = det[:, :-1]  # Remove dustbin

        # Reshape to full resolution
        det = det.view(B, 64, H // 8, W // 8)
        det = F.pixel_shuffle(det, 8)  # (B, 1, H, W)
        det = det.squeeze(1)  # (B, H, W)

        # Descriptor
        desc = self.descriptor(features)
        desc = F.normalize(desc, p=2, dim=1)
        # Interpolate to full resolution
        desc = F.interpolate(desc, size=(H, W), mode='bilinear', align_corners=True)

        # Extract keypoints
        keypoints_list = []
        descriptors_list = []
        scores_list = []

        for b in range(B):
            # NMS
            kpts, scores = self._nms(det[b], nms_radius, threshold, max_keypoints)

            if len(kpts) == 0:
                kpts = torch.zeros(0, 2, device=self.device)
                descs = torch.zeros(0, 256, device=self.device)
                scores = torch.zeros(0, device=self.device)
            else:
                # Sample descriptors at keypoint locations
                kpts_norm = kpts.clone()
                kpts_norm[:, 0] = 2.0 * kpts_norm[:, 0] / (W - 1) - 1.0
                kpts_norm[:, 1] = 2.0 * kpts_norm[:, 1] / (H - 1) - 1.0
                kpts_norm = kpts_norm.view(1, 1, -1, 2)

                descs = F.grid_sample(
                    desc[b:b+1], kpts_norm,
                    mode='bilinear', align_corners=True
                )
                descs = descs.view(256, -1).T  # (N, 256)

            keypoints_list.append(kpts)
            descriptors_list.append(descs)
            scores_list.append(scores)

        # Return first batch item for now
        return keypoints_list[0], descriptors_list[0], scores_list[0]

    def _nms(
        self,
        scores: torch.Tensor,
        radius: int,
        threshold: float,
        max_keypoints: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply non-maximum suppression.

        Args:
            scores: Detection scores, shape (H, W)
            radius: NMS radius
            threshold: Detection threshold
            max_keypoints: Maximum number of keypoints

        Returns:
            Tuple of (keypoints, scores)
        """
        H, W = scores.shape

        # Threshold
        mask = scores > threshold

        # Max pooling for NMS
        pool = F.max_pool2d(
            scores.unsqueeze(0).unsqueeze(0),
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius
        )
        pool = pool.squeeze()

        # Keep only local maxima
        mask = mask & (scores == pool)

        # Get coordinates and scores
        coords = torch.nonzero(mask, as_tuple=False)  # (N, 2) - (y, x)
        kpt_scores = scores[mask]

        # Sort by score
        sorted_idx = torch.argsort(kpt_scores, descending=True)
        coords = coords[sorted_idx]
        kpt_scores = kpt_scores[sorted_idx]

        # Limit number of keypoints
        if len(coords) > max_keypoints:
            coords = coords[:max_keypoints]
            kpt_scores = kpt_scores[:max_keypoints]

        # Convert to (x, y) format
        keypoints = coords.flip(1).float()  # (y, x) -> (x, y)

        return keypoints, kpt_scores


def visualize_features(
    image: Union[torch.Tensor, np.ndarray],
    features: Features,
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 3
) -> np.ndarray:
    """
    Visualize detected features on an image.

    Args:
        image: Input image
        features: Detected features
        color: Circle color (BGR)
        radius: Circle radius

    Returns:
        Image with features drawn
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = image.copy()

    keypoints = features.keypoints.cpu().numpy()
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(image, (x, y), radius, color, -1)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
