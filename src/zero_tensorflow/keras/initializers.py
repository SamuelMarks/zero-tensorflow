"""Keras initializers module."""

from typing import Any, Optional
import numpy as np
import zero_tensorflow

__all__ = [
    "Initializer",
    "Constant",
    "Zeros",
    "Ones",
    "Identity",
    "IdentityInitializer",
    "Orthogonal",
    "OrthogonalInitializer",
    "RandomNormal",
    "RandomUniform",
    "TruncatedNormal",
    "VarianceScaling",
    "GlorotNormal",
    "GlorotUniform",
    "HeNormal",
    "HeUniform",
    "LecunNormal",
    "LecunUniform",
    "STFT",
    "STFTInitializer",
    "constant",
    "zeros",
    "ones",
    "identity",
    "orthogonal",
    "random_normal",
    "random_uniform",
    "truncated_normal",
    "variance_scaling",
    "glorot_normal",
    "glorot_uniform",
    "he_normal",
    "he_uniform",
    "lecun_normal",
    "lecun_uniform",
    "stft",
]


def _compute_fans(shape):
    """_compute_fans docstring."""
    if len(shape) < 1:
        return 1, 1
    if len(shape) == 1:
        return shape[0], shape[0]
    if len(shape) == 2:
        return shape[0], shape[1]
    receptive_field_size = 1
    for dim in shape[:-2]:
        receptive_field_size *= dim
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out


class Initializer:
    """Initializer base class: all Keras initializers inherit from this class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """__init__ docstring."""
        pass

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """__call__ docstring."""
        return zero_tensorflow.Tensor(np.zeros(shape))


class Constant(Initializer):
    """Initializer that generates tensors with constant values."""

    def __init__(self, value: float = 0.0):
        """__init__ docstring."""
        self.value = value

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """__call__ docstring."""
        return zero_tensorflow.Tensor(np.full(shape, self.value))


class Zeros(Initializer):
    """Initializer that generates tensors initialized to 0."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """__init__ docstring."""
        pass

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """__call__ docstring."""
        return zero_tensorflow.Tensor(np.zeros(shape))


class Ones(Initializer):
    """Initializer that generates tensors initialized to 1."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """__init__ docstring."""
        pass

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """__call__ docstring."""
        return zero_tensorflow.Tensor(np.ones(shape))


class Identity(Initializer):
    """Initializer that generates the identity matrix."""

    def __init__(self, gain: float = 1.0):
        """__init__ docstring."""
        self.gain = gain

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """__call__ docstring."""
        if len(shape) != 2:
            raise ValueError(
                "Identity matrix initializer can only be used for 2D matrices."
            )
        return zero_tensorflow.Tensor(np.eye(*shape) * self.gain)


class Orthogonal(Initializer):
    """Initializer that generates an orthogonal matrix."""

    def __init__(self, gain: float = 1.0, seed: Optional[int] = None):
        """__init__ docstring."""
        self.gain = gain
        self.seed = seed

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """__call__ docstring."""
        if len(shape) < 2:
            raise ValueError(
                "The tensor to initialize must be at least two-dimensional"
            )
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (max(num_rows, num_cols), min(num_rows, num_cols))

        rng = np.random.default_rng(self.seed)
        a = rng.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return zero_tensorflow.Tensor(self.gain * q)


class RandomNormal(Initializer):
    """Random normal initializer."""

    def __init__(
        self, mean: float = 0.0, stddev: float = 0.05, seed: Optional[int] = None
    ):
        """__init__ docstring."""
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """__call__ docstring."""
        rng = np.random.default_rng(self.seed)
        return zero_tensorflow.Tensor(rng.normal(self.mean, self.stddev, shape))


class RandomUniform(Initializer):
    """Random uniform initializer."""

    def __init__(
        self, minval: float = -0.05, maxval: float = 0.05, seed: Optional[int] = None
    ):
        """__init__ docstring."""
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """__call__ docstring."""
        rng = np.random.default_rng(self.seed)
        return zero_tensorflow.Tensor(rng.uniform(self.minval, self.maxval, shape))


class TruncatedNormal(Initializer):
    """Initializer that generates a truncated normal distribution."""

    def __init__(
        self, mean: float = 0.0, stddev: float = 0.05, seed: Optional[int] = None
    ):
        """__init__ docstring."""
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """__call__ docstring."""
        rng = np.random.default_rng(self.seed)
        # Rejection sampling for truncated normal
        samples = []
        num_samples = int(np.prod(shape))
        while len(samples) < num_samples:
            s = rng.normal(self.mean, self.stddev, num_samples)
            s = s[
                (s >= self.mean - 2 * self.stddev) & (s <= self.mean + 2 * self.stddev)
            ]
            samples.extend(s)
        return zero_tensorflow.Tensor(np.array(samples[:num_samples]).reshape(shape))


class VarianceScaling(Initializer):
    """Initializer that adapts its scale to the shape of its input tensors."""

    def __init__(
        self,
        scale: float = 1.0,
        mode: str = "fan_in",
        distribution: str = "truncated_normal",
        seed: Optional[int] = None,
    ):
        """__init__ docstring."""
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed

    def __call__(self, shape: Any, dtype: Any = None, **kwargs: Any) -> Any:
        """__call__ docstring."""
        fan_in, fan_out = _compute_fans(shape)
        if self.mode == "fan_in":
            scale = self.scale / max(1.0, fan_in)
        elif self.mode == "fan_out":
            scale = self.scale / max(1.0, fan_out)
        else:
            scale = self.scale / max(1.0, (fan_in + fan_out) / 2.0)

        if self.distribution == "truncated_normal":
            stddev = np.sqrt(scale) / 0.87962566103423978
            rng = np.random.default_rng(self.seed)
            samples = []
            num_samples = int(np.prod(shape))
            while len(samples) < num_samples:
                s = rng.normal(0.0, stddev, num_samples)
                s = s[(s >= -2 * stddev) & (s <= 2 * stddev)]
                samples.extend(s)
            return zero_tensorflow.Tensor(
                np.array(samples[:num_samples]).reshape(shape)
            )
        elif self.distribution == "untruncated_normal":
            stddev = np.sqrt(scale)
            rng = np.random.default_rng(self.seed)
            return zero_tensorflow.Tensor(rng.normal(0.0, stddev, shape))
        else:  # uniform
            limit = np.sqrt(3.0 * scale)
            rng = np.random.default_rng(self.seed)
            return zero_tensorflow.Tensor(rng.uniform(-limit, limit, shape))


class GlorotNormal(VarianceScaling):
    """The Glorot normal initializer, also called Xavier normal initializer."""

    def __init__(self, seed: Optional[int] = None):
        """__init__ docstring."""
        super().__init__(
            scale=1.0, mode="fan_avg", distribution="truncated_normal", seed=seed
        )


class GlorotUniform(VarianceScaling):
    """The Glorot uniform initializer, also called Xavier uniform initializer."""

    def __init__(self, seed: Optional[int] = None):
        """__init__ docstring."""
        super().__init__(scale=1.0, mode="fan_avg", distribution="uniform", seed=seed)


class HeNormal(VarianceScaling):
    """He normal initializer."""

    def __init__(self, seed: Optional[int] = None):
        """__init__ docstring."""
        super().__init__(
            scale=2.0, mode="fan_in", distribution="truncated_normal", seed=seed
        )


class HeUniform(VarianceScaling):
    """He uniform variance scaling initializer."""

    def __init__(self, seed: Optional[int] = None):
        """__init__ docstring."""
        super().__init__(scale=2.0, mode="fan_in", distribution="uniform", seed=seed)


class LecunNormal(VarianceScaling):
    """Lecun normal initializer."""

    def __init__(self, seed: Optional[int] = None):
        """__init__ docstring."""
        super().__init__(
            scale=1.0, mode="fan_in", distribution="truncated_normal", seed=seed
        )


class LecunUniform(VarianceScaling):
    """Lecun uniform initializer."""

    def __init__(self, seed: Optional[int] = None):
        """__init__ docstring."""
        super().__init__(scale=1.0, mode="fan_in", distribution="uniform", seed=seed)


class STFT(Initializer):
    """Initializer of Conv kernels for Short-term Fourier Transformation (STFT)."""

    def __init__(
        self,
        side: str = "real",
        window: str = "hann",
        scaling: str = "density",
        periodic: bool = False,
    ):
        """__init__ docstring."""
        self.side = side
        self.window = window
        self.scaling = scaling
        self.periodic = periodic


class IdentityInitializer(Identity):
    """Initializer that generates the identity matrix."""

    pass


class OrthogonalInitializer(Orthogonal):
    """Initializer that generates an orthogonal matrix."""

    pass


class STFTInitializer(STFT):
    """Initializer of Conv kernels for Short-term Fourier Transformation (STFT)."""

    pass


class constant(Constant):
    """Initializer that generates tensors with constant values."""

    pass


class zeros(Zeros):
    """Initializer that generates tensors initialized to 0."""

    pass


class ones(Ones):
    """Initializer that generates tensors initialized to 1."""

    pass


class identity(Identity):
    """Initializer that generates the identity matrix."""

    pass


class orthogonal(Orthogonal):
    """Initializer that generates an orthogonal matrix."""

    pass


class random_normal(RandomNormal):
    """Random normal initializer."""

    pass


class random_uniform(RandomUniform):
    """Random uniform initializer."""

    pass


class truncated_normal(TruncatedNormal):
    """Initializer that generates a truncated normal distribution."""

    pass


class variance_scaling(VarianceScaling):
    """Initializer that adapts its scale to the shape of its input tensors."""

    pass


class glorot_normal(GlorotNormal):
    """The Glorot normal initializer, also called Xavier normal initializer."""

    pass


class glorot_uniform(GlorotUniform):
    """The Glorot uniform initializer, also called Xavier uniform initializer."""

    pass


class he_normal(HeNormal):
    """He normal initializer."""

    pass


class he_uniform(HeUniform):
    """He uniform variance scaling initializer."""

    pass


class lecun_normal(LecunNormal):
    """Lecun normal initializer."""

    pass


class lecun_uniform(LecunUniform):
    """Lecun uniform initializer."""

    pass


class stft(STFT):
    """Initializer of Conv kernels for Short-term Fourier Transformation (STFT)."""

    pass
