"""Convolution kernels for the Hybrid scheme (Bennedsen et al., 2017).

This module implements the exact lower-triangular convolution matrix required
for the hybrid simulation scheme of fractional Brownian motion with arbitrary
Hurst parameter H ∈ (0, 1). It is the core building block for the Rough Bergomi
variance process simulation.

Functions
---------
precompute_convolution_matrix
    Build the convolution matrix K used in the hybrid scheme.
"""
from numba import njit
import numpy as np


@njit
def precompute_convolution_matrix(alpha: float, n_steps: int) -> np.ndarray:
    """
    Pre-compute the lower-triangular convolution kernel matrix K for the Hybrid scheme.

    The kernel is given by:

        K_{i,j} = (i - j + 1)^{α+1} - (i - j)^{α+1}   for j ≤ i
                  ---------------------------------
                                     α + 1

    where α = H - 1/2 and H is the Hurst parameter.

    Parameters
    ----------
    alpha : float
        α = H - 0.5  (must be in (-0.5, 0) for rough vol)
    n_steps : int
        Number of time steps (matrix size = n_steps + 1)

    Returns
    -------
    np.ndarray
        (n_steps+1, n_steps+1) lower-triangular convolution matrix
    """
    if abs(alpha + 1) < 1e-10:
        raise ValueError("alpha + 1 too close to zero")
    n = n_steps + 1
    K = np.zeros((n, n))
    kernel = np.zeros(n)

    # Compute fractional kernel coefficients
    for m in range(1, n):
        kernel[m] = (m + 1) ** (alpha + 1) - m ** (alpha + 1)
    kernel[1:] /= alpha + 1

    # Fill lower-triangular part
    for i in range(n):
        for j in range(i + 1):
            K[i, j] = kernel[i - j]
    return K
