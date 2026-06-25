"""TensorFlow random module."""

from typing import Any, Optional
import random as _pyrandom

Tensor = Any

__all__ = [
    "Generator",
    "set_seed",
    "stateless_uniform",
    "stateless_normal",
    "gamma",
    "poisson",
    "categorical",
    "uniform",
    "normal",
]


def set_seed(seed: int) -> None:
    """
    Set the global random seed.

    Args:
        seed: The integer seed.
    """
    _pyrandom.seed(seed)
    # Also set numpy seed if used in backends
    import numpy as np

    np.random.seed(seed)


class Generator:
    """Random-number generator."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the object.

        Args:
            seed: Optional integer seed.
        """
        self._seed = seed

    @classmethod
    def from_seed(cls, seed: int, alg: Optional[int] = None) -> "Generator":
        """
        Create a generator from a seed.

        Args:
            seed: The integer seed.
            alg: Optional RNG algorithm.

        Returns:
            A Generator instance.
        """
        return cls(seed=seed)

    def uniform(
        self,
        shape: Any,
        minval: float = 0,
        maxval: float = 1,
        dtype: Any = None,
        name: Optional[str] = None,
    ) -> Any:
        """
        Output random values from a uniform distribution.

        Args:
            shape: A 1-D integer Tensor or Python array.
            minval: A 0-D Tensor or Python value of type `dtype`.
            maxval: A 0-D Tensor or Python value of type `dtype`.
            dtype: The type of the output.
            name: A name for the operation (optional).

        Returns:
            A tensor of the specified shape.
        """
        import numpy as np

        rng = np.random.RandomState(self._seed)
        return rng.uniform(minval, maxval, tuple(shape))

    def normal(
        self,
        shape: Any,
        mean: float = 0.0,
        stddev: float = 1.0,
        dtype: Any = None,
        name: Optional[str] = None,
    ) -> Any:
        """
        Output random values from a normal distribution.

        Args:
            shape: A 1-D integer Tensor or Python array.
            mean: A 0-D Tensor or Python value of type `dtype`.
            stddev: A 0-D Tensor or Python value of type `dtype`.
            dtype: The type of the output.
            name: A name for the operation (optional).

        Returns:
            A tensor of the specified shape.
        """
        import numpy as np

        rng = np.random.RandomState(self._seed)
        return rng.normal(mean, stddev, tuple(shape))


def stateless_uniform(
    shape: Any,
    seed: Any,
    minval: float = 0,
    maxval: float = 1,
    dtype: Any = None,
    name: Optional[str] = None,
) -> Any:
    """
    Output random values from a uniform distribution.

    Args:
        shape: A 1-D integer Tensor or Python array.
        seed: A shape [2] integer Tensor of seeds.
        minval: A 0-D Tensor or Python value of type `dtype`.
        maxval: A 0-D Tensor or Python value of type `dtype`.
        dtype: The type of the output.
        name: A name for the operation (optional).

    Returns:
        A tensor of the specified shape.
    """
    import numpy as np

    rng = np.random.RandomState(seed[0] if isinstance(seed, (list, tuple)) else seed)
    return rng.uniform(minval, maxval, tuple(shape))


def stateless_normal(
    shape: Any,
    seed: Any,
    mean: float = 0.0,
    stddev: float = 1.0,
    dtype: Any = None,
    name: Optional[str] = None,
) -> Any:
    """
    Output deterministic random values from a normal distribution.

    Args:
        shape: A 1-D integer Tensor or Python array.
        seed: A shape [2] integer Tensor of seeds.
        mean: A 0-D Tensor or Python value of type `dtype`.
        stddev: A 0-D Tensor or Python value of type `dtype`.
        dtype: The type of the output.
        name: A name for the operation (optional).

    Returns:
        A tensor of the specified shape.
    """
    import numpy as np

    rng = np.random.RandomState(seed[0] if isinstance(seed, (list, tuple)) else seed)
    return rng.normal(mean, stddev, tuple(shape))


def gamma(
    shape: Any,
    alpha: Any,
    beta: Optional[Any] = None,
    dtype: Any = None,
    seed: Optional[int] = None,
    name: Optional[str] = None,
) -> Any:
    """
    Draw random samples from the Gamma distribution.

    Args:
        shape: A 1-D integer Tensor or Python array.
        alpha: A Tensor or Python value or N-D array of type `dtype`.
        beta: A Tensor or Python value or N-D array of type `dtype`.
        dtype: The type of the output.
        seed: A Python integer.
        name: A name for the operation (optional).

    Returns:
        A tensor of the specified shape.
    """
    import numpy as np

    if seed is not None:
        rng = np.random.RandomState(seed)
        return rng.gamma(alpha, beta if beta is not None else 1.0, tuple(shape))
    return np.random.gamma(alpha, beta if beta is not None else 1.0, tuple(shape))


def poisson(
    shape: Any,
    lam: Any,
    dtype: Any = None,
    seed: Optional[int] = None,
    name: Optional[str] = None,
) -> Any:
    """
    Draw random samples from the Poisson distribution.

    Args:
        shape: A 1-D integer Tensor or Python array.
        lam: A Tensor or Python value or N-D array of type `dtype`.
        dtype: The type of the output.
        seed: A Python integer.
        name: A name for the operation (optional).

    Returns:
        A tensor of the specified shape.
    """
    import numpy as np

    if seed is not None:
        rng = np.random.RandomState(seed)
        return rng.poisson(lam, tuple(shape))
    return np.random.poisson(lam, tuple(shape))


def categorical(
    logits: Any,
    num_samples: int,
    dtype: Any = None,
    seed: Optional[int] = None,
    name: Optional[str] = None,
) -> Any:
    """
    Draw random samples from a categorical distribution.

    Args:
        logits: 2-D Tensor with shape `[batch_size, num_classes]`.
        num_samples: 0-D. Number of independent samples to draw.
        dtype: The type of the output.
        seed: A Python integer.
        name: A name for the operation (optional).

    Returns:
        A tensor of the specified shape.
    """
    import numpy as np

    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    # logits shape [batch_size, num_classes]
    logits = np.array(logits)
    batch_size = logits.shape[0]
    res = np.zeros((batch_size, num_samples), dtype=np.int64)
    for i in range(batch_size):
        p = np.exp(logits[i]) / np.sum(np.exp(logits[i]))
        res[i] = rng.choice(logits.shape[1], size=num_samples, p=p)
    return res


def uniform(
    shape: Any,
    minval: float = 0,
    maxval: float = 1,
    dtype: Any = None,
    seed: Optional[int] = None,
    name: Optional[str] = None,
) -> Any:
    """
    Output random values from a uniform distribution.

    Args:
        shape: A 1-D integer Tensor or Python array.
        minval: A 0-D Tensor or Python value of type `dtype`.
        maxval: A 0-D Tensor or Python value of type `dtype`.
        dtype: The type of the output.
        seed: A Python integer.
        name: A name for the operation (optional).

    Returns:
        A tensor of the specified shape.
    """
    import numpy as np

    if seed is not None:
        rng = np.random.RandomState(seed)
        return rng.uniform(minval, maxval, tuple(shape))
    return np.random.uniform(minval, maxval, tuple(shape))


def normal(
    shape: Any,
    mean: float = 0.0,
    stddev: float = 1.0,
    dtype: Any = None,
    seed: Optional[int] = None,
    name: Optional[str] = None,
) -> Any:
    """
    Output random values from a normal distribution.

    Args:
        shape: A 1-D integer Tensor or Python array.
        mean: A 0-D Tensor or Python value of type `dtype`.
        stddev: A 0-D Tensor or Python value of type `dtype`.
        dtype: The type of the output.
        seed: A Python integer.
        name: A name for the operation (optional).

    Returns:
        A tensor of the specified shape.
    """
    import numpy as np

    if seed is not None:
        rng = np.random.RandomState(seed)
        return rng.normal(mean, stddev, tuple(shape))
    return np.random.normal(mean, stddev, tuple(shape))


def all_candidate_sampler(*args: Any, **kwargs: Any) -> Any:
    pass


def create_rng_state(*args: Any, **kwargs: Any) -> Any:
    pass


def experimental(*args: Any, **kwargs: Any) -> Any:
    pass


def fold_in(*args: Any, **kwargs: Any) -> Any:
    pass


def get_global_generator(*args: Any, **kwargs: Any) -> Any:
    pass


def set_global_generator(*args: Any, **kwargs: Any) -> Any:
    pass


def shuffle(value: Any, seed: Optional[int] = None, name: Optional[str] = None) -> Any:
    import numpy as np

    val = np.array(value)
    if seed is not None:
        np.random.RandomState(seed).shuffle(val)
    else:
        np.random.shuffle(val)
    return val


def split(*args: Any, **kwargs: Any) -> Any:
    pass


def stateless_binomial(*args: Any, **kwargs: Any) -> Any:
    pass


def stateless_categorical(*args: Any, **kwargs: Any) -> Any:
    pass


def stateless_gamma(*args: Any, **kwargs: Any) -> Any:
    pass


def stateless_poisson(*args: Any, **kwargs: Any) -> Any:
    pass


def truncated_normal(
    shape: Any,
    mean: float = 0.0,
    stddev: float = 1.0,
    dtype: Any = None,
    seed: Optional[int] = None,
    name: Optional[str] = None,
) -> Any:
    import numpy as np

    if seed is not None:
        rng = np.random.RandomState(seed)
        return rng.normal(mean, stddev, tuple(shape))  # simplified
    return np.random.normal(mean, stddev, tuple(shape))
