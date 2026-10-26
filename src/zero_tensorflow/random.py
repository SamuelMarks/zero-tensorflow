"""TensorFlow `random` number generation module."""

from __future__ import annotations

import math
import random as _pyrandom
from collections.abc import Sequence

from zero_tensorflow._typing import DTypeLike, ShapeLike, TensorOperand
from zero_tensorflow.core_tensor import Tensor, _to_tensor, _wrap_tensor

import ml_switcheroo_compiler.core as msc_core
import ml_switcheroo_compiler.ops as msc_ops
import zero_keras.random as zk_random

__all__ = [
    "Generator",
    "all_candidate_sampler",
    "categorical",
    "create_rng_state",
    "experimental",
    "fold_in",
    "gamma",
    "get_global_generator",
    "normal",
    "poisson",
    "set_global_generator",
    "set_seed",
    "shuffle",
    "split",
    "stateless_binomial",
    "stateless_categorical",
    "stateless_gamma",
    "stateless_normal",
    "stateless_poisson",
    "stateless_uniform",
    "truncated_normal",
    "uniform",
]


def _extract_seed(seed: int | Sequence[int] | None) -> int | None:
    """Extract an integer seed from a scalar or 2-element seed sequence.

    Args:
        seed: Integer seed, sequence of integers, or None.

    Returns:
        int | None: Resolved scalar integer seed or None.
    """
    if isinstance(seed, int):
        return seed
    if isinstance(seed, Sequence):
        if len(seed) >= 2:
            return (
                int(seed[0]) * 6364136223846793005 + int(seed[1]) + 1442695040888963407
            ) & 0x7FFFFFFF
        return int(seed[0]) if seed else 0
    return None


def set_seed(seed: int) -> None:
    """Set the global random seed.

    Args:
        seed: The integer seed.
    """
    _pyrandom.seed(seed)


class Generator:
    """Random-number generator class."""

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the `Generator` instance.

        Args:
            seed: Optional integer seed.
        """
        self._seed = seed

    def _next_seed(self) -> int:
        """Advance generator state and return next step seed.

        Returns:
            int: Next step seed.
        """
        base = self._seed if self._seed is not None else 0
        self._seed = (base * 1664525 + 1013904223) & 0x7FFFFFFF
        return self._seed

    @classmethod
    def from_seed(cls, seed: int, alg: int | str | None = None) -> Generator:
        """Create a `Generator` from a seed.

        Args:
            seed: The integer seed.
            alg: Optional RNG algorithm identifier.

        Returns:
            Generator: A newly initialized `Generator` instance.
        """
        del alg
        return cls(seed=seed)

    def uniform(
        self,
        shape: ShapeLike,
        minval: float = 0.0,
        maxval: float = 1.0,
        dtype: DTypeLike = None,
        name: str | None = None,
    ) -> Tensor:
        """Output random values from a uniform distribution.

        Args:
            shape: A 1-D integer Tensor or Python sequence.
            minval: Lower bound on the range of random values to generate.
            maxval: Upper bound on the range of random values to generate.
            dtype: The type of the output.
            name: Optional name for the operation.

        Returns:
            Tensor: A tensor of the specified shape filled with uniform values.
        """
        del name
        res: msc_core.Tensor = zk_random.uniform(
            shape, minval, maxval, dtype, seed=self._next_seed()
        )
        return _wrap_tensor(res)

    def normal(
        self,
        shape: ShapeLike,
        mean: float = 0.0,
        stddev: float = 1.0,
        dtype: DTypeLike = None,
        name: str | None = None,
    ) -> Tensor:
        """Output random values from a normal distribution.

        Args:
            shape: A 1-D integer Tensor or Python sequence.
            mean: Mean of the normal distribution.
            stddev: Standard deviation of the normal distribution.
            dtype: The type of the output.
            name: Optional name for the operation.

        Returns:
            Tensor: A tensor of the specified shape filled with normal values.
        """
        del name
        res: msc_core.Tensor = zk_random.normal(shape, mean, stddev, dtype, seed=self._next_seed())
        return _wrap_tensor(res)


def stateless_uniform(
    shape: ShapeLike,
    seed: int | Sequence[int],
    minval: float = 0.0,
    maxval: float = 1.0,
    dtype: DTypeLike = None,
    name: str | None = None,
) -> Tensor:
    """Output deterministic random values from a uniform distribution.

    Args:
        shape: A 1-D integer Tensor or Python sequence.
        seed: A shape `[2]` integer sequence or scalar seed.
        minval: Lower bound on the range of random values to generate.
        maxval: Upper bound on the range of random values to generate.
        dtype: The type of the output.
        name: Optional name for the operation.

    Returns:
        Tensor: A tensor of the specified shape.
    """
    del name
    res: msc_core.Tensor = zk_random.uniform(shape, minval, maxval, dtype, seed=_extract_seed(seed))
    return _wrap_tensor(res)


def stateless_normal(
    shape: ShapeLike,
    seed: int | Sequence[int],
    mean: float = 0.0,
    stddev: float = 1.0,
    dtype: DTypeLike = None,
    name: str | None = None,
) -> Tensor:
    """Output deterministic random values from a normal distribution.

    Args:
        shape: A 1-D integer Tensor or Python sequence.
        seed: A shape `[2]` integer sequence or scalar seed.
        mean: Mean of the normal distribution.
        stddev: Standard deviation of the normal distribution.
        dtype: The type of the output.
        name: Optional name for the operation.

    Returns:
        Tensor: A tensor of the specified shape.
    """
    del name
    res: msc_core.Tensor = zk_random.normal(shape, mean, stddev, dtype, seed=_extract_seed(seed))
    return _wrap_tensor(res)


def _apply_gamma_beta(
    res: msc_core.Tensor,
    beta: Tensor | TensorOperand,
) -> msc_core.Tensor:
    """Scale gamma samples by inverse-scale parameter `beta` if provided.

    Args:
        res: Raw gamma sample tensor.
        beta: Optional inverse-scale parameter.

    Returns:
        msc_core.Tensor: Scaled gamma sample tensor.
    """
    if beta is None:
        return res
    beta_t = _to_tensor(beta)
    return msc_ops.multiply(res, beta_t)


def gamma(
    shape: ShapeLike,
    alpha: Tensor | TensorOperand,
    beta: Tensor | TensorOperand = None,
    dtype: DTypeLike = None,
    seed: int | None = None,
    name: str | None = None,
) -> Tensor:
    """Draw random samples from the Gamma distribution.

    Args:
        shape: A 1-D integer Tensor or Python sequence.
        alpha: Shape parameter `alpha` of the Gamma distribution.
        beta: Optional inverse scale parameter `beta` of the Gamma distribution.
        dtype: The type of the output.
        seed: Optional Python integer seed.
        name: Optional name for the operation.

    Returns:
        Tensor: A tensor of the specified shape.
    """
    del name
    res: msc_core.Tensor = zk_random.gamma(shape, alpha, dtype, seed)
    return _wrap_tensor(_apply_gamma_beta(res, beta))


def _sample_poisson_scalar(lam_val: float, rng: _pyrandom.Random) -> float:
    """Draw a single Poisson sample using Knuth's algorithm.

    Args:
        lam_val: Non-negative rate parameter `lambda`.
        rng: Python `Random` instance.

    Returns:
        float: Sampled non-negative integer count as a float.
    """
    if lam_val <= 0.0:
        return 0.0
    limit = math.exp(-min(lam_val, 50.0))
    k = 0
    p = 1.0
    while p > limit:
        k += 1
        p *= rng.random()
    return float(k - 1)


def _build_poisson_nested(
    dims: tuple[int, ...],
    lam_val: float,
    rng: _pyrandom.Random,
) -> TensorOperand:
    """Recursively build nested lists of Poisson samples for `dims`.

    Args:
        dims: Remaining dimensions tuple.
        lam_val: Rate parameter `lambda`.
        rng: Python `Random` instance.

    Returns:
        TensorOperand: Nested Python list or scalar of Poisson samples.
    """
    if not dims:
        return _sample_poisson_scalar(lam_val, rng)
    first, rest = dims[0], dims[1:]
    return [_build_poisson_nested(rest, lam_val, rng) for _ in range(first)]


def poisson(
    shape: ShapeLike,
    lam: Tensor | TensorOperand,
    dtype: DTypeLike = None,
    seed: int | None = None,
    name: str | None = None,
) -> Tensor:
    """Draw random samples from the Poisson distribution.

    Args:
        shape: A 1-D integer Tensor or Python sequence.
        lam: Rate parameter `lambda` of the Poisson distribution.
        dtype: The type of the output.
        seed: Optional Python integer seed.
        name: Optional name for the operation.

    Returns:
        Tensor: A tensor of the specified shape filled with Poisson samples.
    """
    del name
    resolved_shape: tuple[int, ...] = (
        tuple(int(d) for d in shape if d is not None)
        if isinstance(shape, Sequence)
        else ((int(shape),) if isinstance(shape, int) else (1,))
    )
    lam_float: float = (
        float(lam) if isinstance(lam, (int, float)) and not isinstance(lam, bool) else 1.0
    )
    rng = _pyrandom.Random(seed)
    samples = _build_poisson_nested(resolved_shape, lam_float, rng)
    return _wrap_tensor(_to_tensor(samples, dtype=dtype))


def categorical(
    logits: Tensor | TensorOperand,
    num_samples: int,
    dtype: DTypeLike = None,
    seed: int | None = None,
    name: str | None = None,
) -> Tensor:
    """Draw random samples from a categorical distribution.

    Args:
        logits: 2-D Tensor with shape `[batch_size, num_classes]`.
        num_samples: 0-D number of independent samples to draw for each row.
        dtype: Optional integer output dtype.
        seed: Optional Python integer seed.
        name: Optional name for the operation.

    Returns:
        Tensor: 2-D Tensor of sampled class indices.
    """
    del name
    res: msc_core.Tensor = zk_random.categorical(_to_tensor(logits), num_samples, dtype, seed=seed)
    return _wrap_tensor(res)


def uniform(
    shape: ShapeLike,
    minval: float = 0.0,
    maxval: float = 1.0,
    dtype: DTypeLike = None,
    seed: int | None = None,
    name: str | None = None,
) -> Tensor:
    """Output random values from a uniform distribution.

    Args:
        shape: A 1-D integer Tensor or Python sequence.
        minval: Lower bound on the range of random values to generate.
        maxval: Upper bound on the range of random values to generate.
        dtype: The type of the output.
        seed: Optional Python integer seed.
        name: Optional name for the operation.

    Returns:
        Tensor: A tensor of the specified shape.
    """
    del name
    res: msc_core.Tensor = zk_random.uniform(shape, minval, maxval, dtype, seed)
    return _wrap_tensor(res)


def normal(
    shape: ShapeLike,
    mean: float = 0.0,
    stddev: float = 1.0,
    dtype: DTypeLike = None,
    seed: int | None = None,
    name: str | None = None,
) -> Tensor:
    """Output random values from a normal distribution.

    Args:
        shape: A 1-D integer Tensor or Python sequence.
        mean: Mean of the normal distribution.
        stddev: Standard deviation of the normal distribution.
        dtype: The type of the output.
        seed: Optional Python integer seed.
        name: Optional name for the operation.

    Returns:
        Tensor: A tensor of the specified shape.
    """
    del name
    res: msc_core.Tensor = zk_random.normal(shape, mean, stddev, dtype, seed)
    return _wrap_tensor(res)


def all_candidate_sampler(
    *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
) -> Tensor:
    """Generate the set of all classes as candidates.

    Args:
        *args: Positional candidate sampler arguments.
        **kwargs: Keyword candidate sampler arguments (`num_sampled`, `seed`).

    Returns:
        Tensor: Sampled candidate tensor.
    """
    del args
    num_sampled_raw = kwargs.get("num_sampled", 1)
    num_sampled = int(num_sampled_raw) if isinstance(num_sampled_raw, int) else 1
    seed_raw = kwargs.get("seed")
    seed_val = int(seed_raw) if isinstance(seed_raw, int) else None
    res: msc_core.Tensor = zk_random.uniform([num_sampled], 0, 1, seed=seed_val)
    return _wrap_tensor(res)


def create_rng_state(
    seed: int = 0,
    alg: int | str | None = None,
) -> list[int]:
    """Create a RNG state vector from a seed and algorithm.

    Args:
        seed: Integer seed value.
        alg: Optional RNG algorithm identifier.

    Returns:
        list[int]: Two-element RNG state list `[seed, 0]`.
    """
    del alg
    return [seed, 0]


class experimental:
    """Experimental random number generation APIs."""


def fold_in(
    seed: int | Sequence[int] = 0,
    data: int = 0,
    alg: int | str | None = None,
) -> list[int]:
    """Fold data into an RNG seed to form a new RNG seed.

    Args:
        seed: Existing RNG seed (scalar or sequence).
        data: Integer data to fold into the seed.
        alg: Optional RNG algorithm identifier.

    Returns:
        list[int]: New 2-element RNG seed list.
    """
    del alg
    seed_scalar = seed if isinstance(seed, int) else (int(seed[0]) if seed else 0)
    return [seed_scalar, int(data)]


_GLOBAL_GENERATOR: Generator = Generator()


def get_global_generator() -> Generator:
    """Return the global default `tf.random.Generator` instance.

    Returns:
        Generator: Global generator instance.
    """
    return _GLOBAL_GENERATOR


def set_global_generator(generator: Generator | None = None) -> None:
    """Replace the global default `tf.random.Generator` instance.

    Args:
        generator: New `Generator` instance to set as global.
    """
    global _GLOBAL_GENERATOR
    _GLOBAL_GENERATOR = generator if generator is not None else Generator()


def shuffle(
    value: Tensor | TensorOperand,
    seed: int | None = None,
    name: str | None = None,
) -> Tensor:
    """Randomly shuffle a tensor along its first dimension.

    Args:
        value: The Tensor or sequence to be shuffled.
        seed: Optional Python integer seed.
        name: Optional name for the operation.

    Returns:
        Tensor: Shuffled tensor or sequence.
    """
    del name
    val_in = _to_tensor(value) if isinstance(value, Tensor) else value
    res: msc_core.Tensor = zk_random.shuffle(val_in, seed=seed)
    return _wrap_tensor(res)


def split(
    seed: int | Sequence[int] = 0,
    num: int = 2,
    alg: int | str | None = None,
) -> list[list[int]]:
    """Split an RNG seed into `num` new seeds by adding a leading axis.

    Args:
        seed: Existing RNG seed (scalar or sequence).
        num: Number of new seeds to generate.
        alg: Optional RNG algorithm identifier.

    Returns:
        list[list[int]]: List of `num` 2-element RNG seeds.
    """
    del alg
    seed_val = seed if isinstance(seed, int) else (int(seed[0]) if seed else 0)
    return [[seed_val + i, 0] for i in range(num)]


def stateless_binomial(
    shape: ShapeLike = (1,),
    seed: int | Sequence[int] = (0, 0),
    counts: Tensor | TensorOperand = 0,
    probs: Tensor | TensorOperand = 0,
    output_dtype: DTypeLike = None,
    name: str | None = None,
) -> Tensor:
    """Output deterministic random values from a binomial distribution.

    Args:
        shape: Output tensor shape.
        seed: 2-element integer seed or scalar.
        counts: Number of trials parameter.
        probs: Probability of success parameter.
        output_dtype: Output data type.
        name: Optional operation name.

    Returns:
        Tensor: Sampled binomial tensor.
    """
    del name
    res: msc_core.Tensor = zk_random.binomial(
        shape, counts, probs, output_dtype, seed=_extract_seed(seed)
    )
    return _wrap_tensor(res)


def stateless_categorical(
    logits: Tensor | TensorOperand = None,
    num_samples: int = 1,
    seed: int | Sequence[int] = (0, 0),
    dtype: DTypeLike = None,
    name: str | None = None,
) -> Tensor:
    """Draw deterministic random samples from a categorical distribution.

    Args:
        logits: 2-D Tensor with shape `[batch_size, num_classes]`.
        num_samples: 0-D number of independent samples to draw.
        seed: 2-element integer seed or scalar.
        dtype: Output integer data type.
        name: Optional operation name.

    Returns:
        Tensor: Sampled categorical indices tensor.
    """
    del name
    logits_in = _to_tensor(logits) if logits is not None else [[1.0]]
    res: msc_core.Tensor = zk_random.categorical(
        logits_in, num_samples, dtype, seed=_extract_seed(seed)
    )
    return _wrap_tensor(res)


def stateless_gamma(
    shape: ShapeLike = (1,),
    seed: int | Sequence[int] = (0, 0),
    alpha: Tensor | TensorOperand = 0,
    beta: Tensor | TensorOperand = None,
    dtype: DTypeLike = None,
    name: str | None = None,
) -> Tensor:
    """Output deterministic random values from a gamma distribution.

    Args:
        shape: Output tensor shape.
        seed: 2-element integer seed or scalar.
        alpha: Shape parameter `alpha`.
        beta: Optional inverse scale parameter `beta`.
        dtype: Output data type.
        name: Optional operation name.

    Returns:
        Tensor: Sampled gamma tensor.
    """
    del name
    res: msc_core.Tensor = zk_random.gamma(shape, alpha, dtype, seed=_extract_seed(seed))
    return _wrap_tensor(_apply_gamma_beta(res, beta))


def stateless_poisson(
    shape: ShapeLike = (1,),
    seed: int | Sequence[int] = (0, 0),
    lam: Tensor | TensorOperand = 0,
    dtype: DTypeLike = None,
    name: str | None = None,
) -> Tensor:
    """Output deterministic random values from a Poisson distribution.

    Args:
        shape: Output tensor shape.
        seed: 2-element integer seed or scalar.
        lam: Rate parameter `lambda`.
        dtype: Output data type.
        name: Optional operation name.

    Returns:
        Tensor: Sampled Poisson tensor.
    """
    return poisson(shape, lam, dtype=dtype, seed=_extract_seed(seed), name=name)


def truncated_normal(
    shape: ShapeLike,
    mean: float = 0.0,
    stddev: float = 1.0,
    dtype: DTypeLike = None,
    seed: int | None = None,
    name: str | None = None,
) -> Tensor:
    """Output random values from a truncated normal distribution.

    Args:
        shape: A 1-D integer Tensor or Python sequence.
        mean: Mean of the truncated normal distribution.
        stddev: Standard deviation of the normal distribution before truncation.
        dtype: The type of the output.
        seed: Optional Python integer seed.
        name: Optional name for the operation.

    Returns:
        Tensor: Sampled truncated normal tensor.
    """
    del name
    res: msc_core.Tensor = zk_random.truncated_normal(shape, mean, stddev, dtype, seed)
    return _wrap_tensor(res)
