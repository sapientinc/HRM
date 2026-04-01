import math

import torch


def truncated_normal_init_(
    tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0
) -> torch.Tensor:
    """Initialize tensor with truncated normal distribution (JAX-style).

    This is a mathematically correct implementation of truncated normal
    initialization, matching JAX/Flax defaults. Unlike PyTorch's built-in
    trunc_normal_, this ensures the standard deviation is exactly `std`.

    Args:
        tensor: Tensor to initialize in-place
        std: Target standard deviation of the distribution
        lower: Lower truncation bound in units of std (default: -2.0)
        upper: Upper truncation bound in units of std (default: 2.0)

    Returns:
        The initialized tensor (same as input, modified in-place)

    References:
        JAX truncated normal: https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
        Flax initializer: https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199
    """
    # NOTE: PyTorch nn.init.trunc_normal_ is not mathematically correct, the std dev is not actually the std dev of initialized tensor
    # This function is a PyTorch version of jax truncated normal init (default init method in flax)
    # https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
    # https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199

    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower**2)
            pdf_l = c * math.exp(-0.5 * upper**2)
            comp_std = std / math.sqrt(
                1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2
            )

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor
