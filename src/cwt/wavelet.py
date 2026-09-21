"""
PyTorch-based GPU Continuous Wavelet Transform (CWT) for FHR and UC time series.

Provides deterministic, window-wise 2D time-frequency scalograms covering the
physiologically meaningful FHR frequency spectrum (0.01 Hz - 1.0 Hz).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ContinuousWaveletTransform(nn.Module):
    """
    Computes Continuous Wavelet Transform (CWT) using Complex Morlet wavelet via FFT.

    Args:
        num_scales (int): Number of frequency scales (default: 64).
        f_min (float): Minimum frequency in Hz (default: 0.01 Hz).
        f_max (float): Maximum frequency in Hz (default: 1.0 Hz).
        fs (float): Sampling frequency in Hz (default: 4.0 Hz).
        w0 (float): Central frequency parameter of Morlet wavelet (default: 6.0).
        time_pool (int): Temporal pooling factor to produce compact scalograms (default: 8 -> 4800 to 600).
    """
    def __init__(self, num_scales: int = 64, f_min: float = 0.01, f_max: float = 1.0,
                 fs: float = 4.0, w0: float = 6.0, time_pool: int = 8):
        super().__init__()
        self.num_scales = num_scales
        self.f_min = f_min
        self.f_max = f_max
        self.fs = fs
        self.w0 = w0
        self.time_pool = time_pool

        # Logarithmically spaced frequencies from f_min to f_max
        frequencies = np.geomspace(f_min, f_max, num_scales)
        # Scales: s = w0 * fs / (2 * pi * f)
        scales = (w0 * fs) / (2.0 * math.pi * frequencies)

        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))
        self.register_buffer("frequencies", torch.tensor(frequencies, dtype=torch.float32))

    def _morlet_ft(self, omega: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Fourier transform of Morlet wavelet:
        Psi_hat(s * omega) = pi^(-1/4) * H(omega) * exp(-0.5 * (s * omega - w0)^2)
        """
        # s_omega shape: (num_scales, N_freqs)
        s_omega = scale.unsqueeze(1) * omega.unsqueeze(0)
        norm_const = (math.pi ** (-0.25)) * math.sqrt(2.0 * math.pi)
        
        # Gaussian envelope centered at w0
        exponent = -0.5 * (s_omega - self.w0) ** 2
        psi_hat = norm_const * torch.exp(exponent)
        # Heaviside step function: only positive frequencies
        psi_hat = psi_hat * (omega.unsqueeze(0) > 0).float()
        return psi_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Channels, Time) e.g. (B, 1, 4800) or (B, 2, 4800)
        Returns:
            scalogram: Tensor of shape (Batch, Channels, num_scales, Time // time_pool)
        """
        B, C, N = x.shape
        device = x.device

        # Angular frequency grid for rfft (positive frequencies)
        # For N points, rfft produces N // 2 + 1 points with freq spacing 2*pi*fs / N
        k = torch.arange(N // 2 + 1, device=device, dtype=torch.float32)
        omega = (2.0 * math.pi * self.fs / N) * k

        # Wavelet FT filters: (num_scales, N // 2 + 1)
        psi_hat = self._morlet_ft(omega, self.scales) # (num_scales, N_freqs)

        # RFFT of input: (B * C, N_freqs) complex
        x_flat = x.view(B * C, N)
        x_fft = torch.fft.rfft(x_flat, dim=-1) # (B * C, N_freqs)

        # Broadcast multiply: (B * C, 1, N_freqs) * (1, num_scales, N_freqs) -> (B * C, num_scales, N_freqs)
        x_fft_expanded = x_fft.unsqueeze(1) # (B * C, 1, N_freqs)
        psi_hat_expanded = psi_hat.unsqueeze(0).to(x_fft.dtype) # (1, num_scales, N_freqs)

        # Scale factor sqrt(s) for energy conservation
        sqrt_scales = torch.sqrt(self.scales).unsqueeze(0).unsqueeze(-1) # (1, num_scales, 1)
        filtered_fft = x_fft_expanded * psi_hat_expanded * sqrt_scales

        # Inverse RFFT -> Complex time-domain analytic wavelet coefficients
        cwt_complex = torch.fft.irfft(filtered_fft, n=N, dim=-1) # (B * C, num_scales, N)

        # Compute modulus / amplitude scalogram
        scalogram = torch.abs(cwt_complex) # (B * C, num_scales, N)
        scalogram = scalogram.view(B, C, self.num_scales, N)

        # Temporal pooling if requested
        if self.time_pool > 1:
            scalogram = F.avg_pool2d(scalogram, kernel_size=(1, self.time_pool), stride=(1, self.time_pool))

        return scalogram
