"""
Loss functions for 3D Gaussian Splatting optimization.

Includes photometric losses, structural similarity, temporal consistency,
and regularization terms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import math


class PhotometricLoss(nn.Module):
    """
    Combined L1 and SSIM photometric loss.
    """

    def __init__(
        self,
        lambda_ssim: float = 0.2,
        ssim_window_size: int = 11
    ):
        """
        Initialize photometric loss.

        Args:
            lambda_ssim: Weight for SSIM component (1-lambda for L1)
            ssim_window_size: Window size for SSIM computation
        """
        super().__init__()
        self.lambda_ssim = lambda_ssim
        self.ssim = SSIM(window_size=ssim_window_size)

    def forward(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute photometric loss.

        Args:
            rendered: Rendered image (H, W, 3)
            target: Target image (H, W, 3)
            mask: Optional mask (H, W) for valid regions

        Returns:
            Dict with 'total', 'l1', and 'ssim' losses
        """
        if mask is not None:
            rendered = rendered * mask.unsqueeze(-1)
            target = target * mask.unsqueeze(-1)

        # L1 loss
        l1_loss = F.l1_loss(rendered, target)

        # SSIM loss
        # Convert to (B, C, H, W) format for SSIM
        rendered_bchw = rendered.permute(2, 0, 1).unsqueeze(0)
        target_bchw = target.permute(2, 0, 1).unsqueeze(0)
        ssim_loss = 1.0 - self.ssim(rendered_bchw, target_bchw)

        total = (1 - self.lambda_ssim) * l1_loss + self.lambda_ssim * ssim_loss

        return {
            'total': total,
            'l1': l1_loss,
            'ssim': ssim_loss
        }


class SSIM(nn.Module):
    """
    Structural Similarity Index (SSIM) module.
    """

    def __init__(
        self,
        window_size: int = 11,
        channel: int = 3,
        data_range: float = 1.0
    ):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.data_range = data_range

        # Create Gaussian window
        self.register_buffer('window', self._create_window(window_size, channel))

    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Create Gaussian window."""
        sigma = 1.5
        gauss = torch.Tensor([
            math.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()

        # 2D window
        _2d_window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        window = _2d_window.expand(channel, 1, window_size, window_size).contiguous()

        return window

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM between two images.

        Args:
            img1: First image (B, C, H, W)
            img2: Second image (B, C, H, W)

        Returns:
            SSIM value (scalar)
        """
        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2

        window = self.window.to(img1.device)
        channel = img1.shape[1]

        # Compute means
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Compute variances
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()


class TemporalConsistencyLoss(nn.Module):
    """
    Loss for temporal consistency in dynamic Gaussians.
    """

    def __init__(
        self,
        velocity_weight: float = 1.0,
        acceleration_weight: float = 0.5
    ):
        """
        Initialize temporal consistency loss.

        Args:
            velocity_weight: Weight for velocity consistency term
            acceleration_weight: Weight for acceleration (smoothness) term
        """
        super().__init__()
        self.velocity_weight = velocity_weight
        self.acceleration_weight = acceleration_weight

    def forward(
        self,
        positions_t0: torch.Tensor,
        positions_t1: torch.Tensor,
        velocities: torch.Tensor,
        dt: float,
        positions_t2: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute temporal consistency loss.

        Args:
            positions_t0: Positions at time t (N, 3)
            positions_t1: Positions at time t+dt (N, 3)
            velocities: Predicted velocities (N, 3)
            dt: Time step
            positions_t2: Optional positions at t+2dt for acceleration

        Returns:
            Dict with loss components
        """
        # Velocity consistency: position change should match velocity
        expected_positions = positions_t0 + velocities * dt
        velocity_loss = F.mse_loss(positions_t1, expected_positions)

        total = self.velocity_weight * velocity_loss

        result = {
            'total': total,
            'velocity': velocity_loss
        }

        # Acceleration (smoothness)
        if positions_t2 is not None:
            vel_t0_t1 = (positions_t1 - positions_t0) / dt
            vel_t1_t2 = (positions_t2 - positions_t1) / dt
            acceleration = (vel_t1_t2 - vel_t0_t1) / dt

            accel_loss = (acceleration ** 2).mean()
            result['acceleration'] = accel_loss
            result['total'] = total + self.acceleration_weight * accel_loss

        return result


class OpticalFlowLoss(nn.Module):
    """
    Loss for consistency with observed optical flow.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        predicted_flow: torch.Tensor,
        observed_flow: torch.Tensor,
        confidence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute optical flow consistency loss.

        Args:
            predicted_flow: Predicted 2D flow from Gaussian motion (H, W, 2)
            observed_flow: Observed optical flow (H, W, 2)
            confidence: Optional confidence weights (H, W)

        Returns:
            Flow loss value
        """
        diff = predicted_flow - observed_flow
        loss = (diff ** 2).sum(dim=-1)  # (H, W)

        if confidence is not None:
            loss = loss * confidence

        return self.weight * loss.mean()


class GaussianRegularization(nn.Module):
    """
    Regularization losses for Gaussians.
    """

    def __init__(
        self,
        scale_weight: float = 0.001,
        opacity_weight: float = 0.001,
        scale_limit: float = 0.5
    ):
        """
        Initialize regularization.

        Args:
            scale_weight: Weight for scale regularization
            opacity_weight: Weight for opacity regularization
            scale_limit: Maximum allowed scale (log space)
        """
        super().__init__()
        self.scale_weight = scale_weight
        self.opacity_weight = opacity_weight
        self.scale_limit = scale_limit

    def forward(
        self,
        scales: torch.Tensor,
        opacities: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute regularization losses.

        Args:
            scales: Gaussian scales in log space (N, 3)
            opacities: Gaussian opacities in logit space (N, 1)

        Returns:
            Dict with regularization losses
        """
        # Scale regularization: penalize very large or very small Gaussians
        scale_loss = (scales ** 2).mean()

        # Penalize scales beyond limit
        excess = F.relu(scales.abs() - self.scale_limit)
        scale_loss = scale_loss + 10 * (excess ** 2).mean()

        # Opacity regularization: encourage sparse representation
        opacity_sigmoid = torch.sigmoid(opacities)
        opacity_loss = (opacity_sigmoid * (1 - opacity_sigmoid)).mean()  # Encourage 0 or 1

        total = self.scale_weight * scale_loss + self.opacity_weight * opacity_loss

        return {
            'total': total,
            'scale': scale_loss,
            'opacity': opacity_loss
        }


class GaussianLosses(nn.Module):
    """
    Combined loss function for Gaussian Splatting optimization.
    """

    def __init__(
        self,
        photometric_weight: float = 1.0,
        ssim_weight: float = 0.2,
        temporal_weight: float = 0.1,
        flow_weight: float = 0.5,
        regularization_weight: float = 0.001
    ):
        """
        Initialize combined losses.

        Args:
            photometric_weight: Weight for photometric loss
            ssim_weight: SSIM component in photometric loss
            temporal_weight: Weight for temporal consistency
            flow_weight: Weight for optical flow loss
            regularization_weight: Weight for regularization
        """
        super().__init__()

        self.photometric = PhotometricLoss(lambda_ssim=ssim_weight)
        self.temporal = TemporalConsistencyLoss()
        self.flow_loss = OpticalFlowLoss(weight=flow_weight)
        self.regularization = GaussianRegularization()

        self.weights = {
            'photometric': photometric_weight,
            'temporal': temporal_weight,
            'regularization': regularization_weight
        }

    def forward(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        gaussians,  # GaussianCloud or DynamicGaussianCloud
        optical_flow: Optional[torch.Tensor] = None,
        predicted_flow: Optional[torch.Tensor] = None,
        temporal_data: Optional[Dict] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.

        Args:
            rendered: Rendered image (H, W, 3)
            target: Target image (H, W, 3)
            gaussians: Gaussian representation
            optical_flow: Observed optical flow (H, W, 2)
            predicted_flow: Predicted flow from Gaussian motion
            temporal_data: Dict with temporal consistency data
            mask: Optional mask for valid regions

        Returns:
            Dict with all loss values
        """
        losses = {}

        # Photometric loss
        photo_losses = self.photometric(rendered, target, mask)
        losses['photometric'] = photo_losses['total']
        losses['l1'] = photo_losses['l1']
        losses['ssim'] = photo_losses['ssim']

        # Optical flow loss
        if optical_flow is not None and predicted_flow is not None:
            losses['flow'] = self.flow_loss(predicted_flow, optical_flow)
        else:
            losses['flow'] = torch.tensor(0.0, device=rendered.device)

        # Temporal consistency loss
        if temporal_data is not None:
            temp_losses = self.temporal(
                temporal_data['positions_t0'],
                temporal_data['positions_t1'],
                temporal_data['velocities'],
                temporal_data['dt'],
                temporal_data.get('positions_t2')
            )
            losses['temporal'] = temp_losses['total']
        else:
            losses['temporal'] = torch.tensor(0.0, device=rendered.device)

        # Regularization
        reg_losses = self.regularization(
            gaussians._scales if hasattr(gaussians, '_scales') else gaussians.base_gaussians._scales,
            gaussians._opacities if hasattr(gaussians, '_opacities') else gaussians.base_gaussians._opacities
        )
        losses['regularization'] = reg_losses['total']

        # Total loss
        losses['total'] = (
            self.weights['photometric'] * losses['photometric'] +
            losses['flow'] +
            self.weights['temporal'] * losses['temporal'] +
            self.weights['regularization'] * losses['regularization']
        )

        return losses


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio.

    Args:
        img1: First image
        img2: Second image

    Returns:
        PSNR value in dB
    """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 10 * torch.log10(1.0 / mse)


def compute_lpips(
    img1: torch.Tensor,
    img2: torch.Tensor,
    net: str = 'vgg'
) -> torch.Tensor:
    """
    Compute LPIPS perceptual similarity.

    Requires lpips package to be installed.

    Args:
        img1: First image (H, W, 3)
        img2: Second image (H, W, 3)
        net: Network to use ('vgg' or 'alex')

    Returns:
        LPIPS value (lower is more similar)
    """
    try:
        import lpips
        loss_fn = lpips.LPIPS(net=net)
        loss_fn = loss_fn.to(img1.device)

        # Convert to (B, C, H, W) and [-1, 1] range
        img1_bchw = img1.permute(2, 0, 1).unsqueeze(0) * 2 - 1
        img2_bchw = img2.permute(2, 0, 1).unsqueeze(0) * 2 - 1

        return loss_fn(img1_bchw, img2_bchw).item()
    except ImportError:
        return torch.tensor(0.0)
