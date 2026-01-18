"""
Feature matching between views using SuperGlue and classical methods.

Establishes correspondences between features detected in different camera views
for camera calibration and 3D reconstruction.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import cv2

from ..preprocessing.feature_extraction import Features


@dataclass
class Matches:
    """Container for feature matches between two views."""
    matches_idx: torch.Tensor  # (M, 2) - indices into (features1, features2)
    confidence: torch.Tensor  # (M,) - match confidence scores
    keypoints1: torch.Tensor  # (N1, 2) - keypoints from first view
    keypoints2: torch.Tensor  # (N2, 2) - keypoints from second view

    def __len__(self):
        return self.matches_idx.shape[0]

    @property
    def matched_keypoints1(self) -> torch.Tensor:
        """Get matched keypoints from first view."""
        return self.keypoints1[self.matches_idx[:, 0]]

    @property
    def matched_keypoints2(self) -> torch.Tensor:
        """Get matched keypoints from second view."""
        return self.keypoints2[self.matches_idx[:, 1]]


class FeatureMatcher:
    """
    Match features between views using SuperGlue or classical methods.
    """

    def __init__(
        self,
        method: str = "superglue",
        device: str = "cuda",
        match_threshold: float = 0.7,
        max_matches: int = 2048
    ):
        """
        Initialize the feature matcher.

        Args:
            method: Matching method ('superglue', 'bf', 'flann')
            device: Device for SuperGlue inference
            match_threshold: Confidence threshold for matches
            max_matches: Maximum number of matches to return
        """
        self.method = method
        self.device = device
        self.match_threshold = match_threshold
        self.max_matches = max_matches
        self.model = None

        if method == "superglue":
            self._init_superglue()
        elif method == "bf":
            self._init_bf_matcher()
        elif method == "flann":
            self._init_flann_matcher()

    def _init_superglue(self):
        """Initialize SuperGlue matcher."""
        try:
            from kornia.feature import LightGlueMatcher
            self.model = LightGlueMatcher("superpoint").to(self.device)
            self._matcher_backend = "kornia_lightglue"
        except ImportError:
            # Fallback to classical matching
            print("SuperGlue/LightGlue not available, falling back to BF matcher")
            self.method = "bf"
            self._init_bf_matcher()

    def _init_bf_matcher(self):
        """Initialize brute-force matcher."""
        # For SuperPoint (float descriptors): L2 norm
        # For ORB (binary descriptors): Hamming
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def _init_flann_matcher(self):
        """Initialize FLANN matcher."""
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=50)
        self.flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def match(
        self,
        features1: Features,
        features2: Features,
        image1: Optional[torch.Tensor] = None,
        image2: Optional[torch.Tensor] = None
    ) -> Matches:
        """
        Match features between two views.

        Args:
            features1: Features from first view
            features2: Features from second view
            image1: Optional image for SuperGlue context
            image2: Optional image for SuperGlue context

        Returns:
            Matches object containing correspondences
        """
        if len(features1) == 0 or len(features2) == 0:
            return Matches(
                matches_idx=torch.zeros(0, 2, dtype=torch.long),
                confidence=torch.zeros(0),
                keypoints1=features1.keypoints,
                keypoints2=features2.keypoints
            )

        if self.method == "superglue" and self._matcher_backend == "kornia_lightglue":
            return self._match_superglue(features1, features2, image1, image2)
        elif self.method == "bf":
            return self._match_bf(features1, features2)
        elif self.method == "flann":
            return self._match_flann(features1, features2)

        return self._match_bf(features1, features2)

    def _match_superglue(
        self,
        features1: Features,
        features2: Features,
        image1: Optional[torch.Tensor],
        image2: Optional[torch.Tensor]
    ) -> Matches:
        """Match using SuperGlue/LightGlue."""
        # Prepare inputs
        kpts1 = features1.keypoints.unsqueeze(0).to(self.device)  # (1, N, 2)
        kpts2 = features2.keypoints.unsqueeze(0).to(self.device)
        desc1 = features1.descriptors.unsqueeze(0).to(self.device)  # (1, N, D)
        desc2 = features2.descriptors.unsqueeze(0).to(self.device)

        with torch.no_grad():
            # LightGlue expects dict input
            input_dict = {
                'keypoints0': kpts1,
                'keypoints1': kpts2,
                'descriptors0': desc1.permute(0, 2, 1),  # (1, D, N)
                'descriptors1': desc2.permute(0, 2, 1),
            }

            # Add images if available
            if image1 is not None and image2 is not None:
                input_dict['image0'] = image1.unsqueeze(0).to(self.device)
                input_dict['image1'] = image2.unsqueeze(0).to(self.device)

            matches = self.model(input_dict)

        # Extract matches
        matches_idx = matches['matches'][0]  # (N1,) or similar
        confidence = matches['confidence'][0] if 'confidence' in matches else None

        # Filter invalid matches (-1 means no match)
        valid_mask = matches_idx >= 0
        valid_idx1 = torch.arange(len(matches_idx), device=self.device)[valid_mask]
        valid_idx2 = matches_idx[valid_mask]

        matches_idx_tensor = torch.stack([valid_idx1, valid_idx2], dim=1)

        if confidence is not None:
            confidence = confidence[valid_mask]
        else:
            confidence = torch.ones(len(valid_idx1), device=self.device)

        return Matches(
            matches_idx=matches_idx_tensor.cpu(),
            confidence=confidence.cpu(),
            keypoints1=features1.keypoints,
            keypoints2=features2.keypoints
        )

    def _match_bf(self, features1: Features, features2: Features) -> Matches:
        """Match using brute-force with ratio test."""
        desc1 = features1.descriptors.cpu().numpy()
        desc2 = features2.descriptors.cpu().numpy()

        # KNN match with k=2 for ratio test
        matches = self.bf_matcher.knnMatch(desc1, desc2, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for m_list in matches:
            if len(m_list) >= 2:
                m, n = m_list[0], m_list[1]
                if m.distance < self.match_threshold * n.distance:
                    good_matches.append((m.queryIdx, m.trainIdx, m.distance))
            elif len(m_list) == 1:
                good_matches.append((m_list[0].queryIdx, m_list[0].trainIdx, m_list[0].distance))

        if len(good_matches) == 0:
            return Matches(
                matches_idx=torch.zeros(0, 2, dtype=torch.long),
                confidence=torch.zeros(0),
                keypoints1=features1.keypoints,
                keypoints2=features2.keypoints
            )

        # Sort by distance and limit
        good_matches = sorted(good_matches, key=lambda x: x[2])[:self.max_matches]

        matches_idx = torch.tensor([[m[0], m[1]] for m in good_matches], dtype=torch.long)
        distances = torch.tensor([m[2] for m in good_matches])
        # Convert distance to confidence (inverse relationship)
        max_dist = distances.max() + 1e-6
        confidence = 1 - distances / max_dist

        return Matches(
            matches_idx=matches_idx,
            confidence=confidence,
            keypoints1=features1.keypoints,
            keypoints2=features2.keypoints
        )

    def _match_flann(self, features1: Features, features2: Features) -> Matches:
        """Match using FLANN with ratio test."""
        desc1 = features1.descriptors.cpu().numpy().astype(np.float32)
        desc2 = features2.descriptors.cpu().numpy().astype(np.float32)

        matches = self.flann_matcher.knnMatch(desc1, desc2, k=2)

        # Apply ratio test
        good_matches = []
        for m_list in matches:
            if len(m_list) >= 2:
                m, n = m_list[0], m_list[1]
                if m.distance < self.match_threshold * n.distance:
                    good_matches.append((m.queryIdx, m.trainIdx, m.distance))

        if len(good_matches) == 0:
            return Matches(
                matches_idx=torch.zeros(0, 2, dtype=torch.long),
                confidence=torch.zeros(0),
                keypoints1=features1.keypoints,
                keypoints2=features2.keypoints
            )

        good_matches = sorted(good_matches, key=lambda x: x[2])[:self.max_matches]

        matches_idx = torch.tensor([[m[0], m[1]] for m in good_matches], dtype=torch.long)
        distances = torch.tensor([m[2] for m in good_matches])
        max_dist = distances.max() + 1e-6
        confidence = 1 - distances / max_dist

        return Matches(
            matches_idx=matches_idx,
            confidence=confidence,
            keypoints1=features1.keypoints,
            keypoints2=features2.keypoints
        )

    def match_multiple(
        self,
        features_list: List[Features],
        images: Optional[List[torch.Tensor]] = None
    ) -> Dict[Tuple[int, int], Matches]:
        """
        Match features between all pairs of views.

        Args:
            features_list: List of Features from each view
            images: Optional list of images for context

        Returns:
            Dictionary mapping (view_i, view_j) to Matches
        """
        n_views = len(features_list)
        all_matches = {}

        for i in range(n_views):
            for j in range(i + 1, n_views):
                img1 = images[i] if images else None
                img2 = images[j] if images else None

                matches = self.match(features_list[i], features_list[j], img1, img2)
                all_matches[(i, j)] = matches

        return all_matches


def filter_matches_ransac(
    matches: Matches,
    method: str = "fundamental",
    threshold: float = 3.0,
    confidence: float = 0.99
) -> Tuple[Matches, np.ndarray]:
    """
    Filter matches using RANSAC with fundamental or essential matrix.

    Args:
        matches: Input matches
        method: 'fundamental' or 'essential'
        threshold: RANSAC threshold in pixels
        confidence: RANSAC confidence

    Returns:
        Tuple of (filtered_matches, inlier_mask)
    """
    if len(matches) < 8:
        return matches, np.ones(len(matches), dtype=bool)

    pts1 = matches.matched_keypoints1.numpy()
    pts2 = matches.matched_keypoints2.numpy()

    if method == "fundamental":
        F, mask = cv2.findFundamentalMat(
            pts1, pts2,
            cv2.FM_RANSAC,
            threshold,
            confidence
        )
    else:  # essential
        # For essential matrix, we need camera intrinsics
        # This is a simplified version assuming normalized coordinates
        E, mask = cv2.findEssentialMat(
            pts1, pts2,
            focal=1.0, pp=(0, 0),
            method=cv2.RANSAC,
            prob=confidence,
            threshold=threshold
        )

    if mask is None:
        return matches, np.ones(len(matches), dtype=bool)

    mask = mask.ravel().astype(bool)

    filtered_matches = Matches(
        matches_idx=matches.matches_idx[mask],
        confidence=matches.confidence[mask],
        keypoints1=matches.keypoints1,
        keypoints2=matches.keypoints2
    )

    return filtered_matches, mask


def visualize_matches(
    image1: np.ndarray,
    image2: np.ndarray,
    matches: Matches,
    max_matches: int = 100
) -> np.ndarray:
    """
    Visualize matches between two images.

    Args:
        image1: First image (H, W, 3)
        image2: Second image (H, W, 3)
        matches: Matches object
        max_matches: Maximum matches to display

    Returns:
        Visualization image
    """
    # Ensure uint8
    if image1.max() <= 1.0:
        image1 = (image1 * 255).astype(np.uint8)
        image2 = (image2 * 255).astype(np.uint8)

    # Convert to BGR for OpenCV
    if image1.ndim == 3 and image1.shape[-1] == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

    # Create KeyPoint objects
    kp1 = [cv2.KeyPoint(float(k[0]), float(k[1]), 1) for k in matches.matched_keypoints1[:max_matches]]
    kp2 = [cv2.KeyPoint(float(k[0]), float(k[1]), 1) for k in matches.matched_keypoints2[:max_matches]]

    # Create DMatch objects
    dm = [cv2.DMatch(i, i, 0) for i in range(min(len(kp1), max_matches))]

    # Draw matches
    vis = cv2.drawMatches(image1, kp1, image2, kp2, dm, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
