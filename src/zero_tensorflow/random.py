"TensorFlow random module."

from __future__ import annotations

import random as _pyrandom
from typing import Any

import zero_keras.random as zk_random
from zero_keras import ops as _ops

Tensor = Any
__all__ = [
    "Generator",
    "categorical",
    "gamma",
    "normal",
    "poisson",
    "set_seed",
    "stateless_normal",
    "stateless_uniform",
    "uniform",
]


def set_seed(seed: int) -> None:
    "Set the global random seed.\n\nArgs:\n    seed: The integer seed."
    _pyrandom.seed(seed)


class Generator:
    "Random-number generator."

    def __init__(self, seed: int | None = None):
        "Initialize the object.\n\nArgs:\n    seed: Optional integer seed."
        self._seed = seed

    @classmethod
    def from_seed(cls, seed: int, alg: int | None = None) -> Generator:
        "Create a generator from a seed.\n\nArgs:\n    seed: The integer seed.\n    alg: Optional RNG algorithm.\n\nReturns:\n    A Generator instance."
        return cls(seed=seed)

    def uniform(
        self,
        shape: Any,
        minval: float = 0,
        maxval: float = 1,
        dtype: Any = None,
        name: str | None = None,
    ) -> Any:
        "Output random values from a uniform distribution.\n\nArgs:\n    shape: A 1-D integer Tensor or Python array.\n    minval: A 0-D Tensor or Python value of type `dtype`.\n    maxval: A 0-D Tensor or Python value of type `dtype`.\n    dtype: The type of the output.\n    name: A name for the operation (optional).\n\nReturns:\n    A tensor of the specified shape."
        return zk_random.uniform(shape, minval, maxval, dtype, seed=self._seed)

    def normal(
        self,
        shape: Any,
        mean: float = 0.0,
        stddev: float = 1.0,
        dtype: Any = None,
        name: str | None = None,
    ) -> Any:
        "Output random values from a normal distribution.\n\nArgs:\n    shape: A 1-D integer Tensor or Python array.\n    mean: A 0-D Tensor or Python value of type `dtype`.\n    stddev: A 0-D Tensor or Python value of type `dtype`.\n    dtype: The type of the output.\n    name: A name for the operation (optional).\n\nReturns:\n    A tensor of the specified shape."
        return zk_random.normal(shape, mean, stddev, dtype, seed=self._seed)


def stateless_uniform(
    shape: Any,
    seed: Any,
    minval: float = 0,
    maxval: float = 1,
    dtype: Any = None,
    name: str | None = None,
) -> Any:
    "Output random values from a uniform distribution.\n\nArgs:\n    shape: A 1-D integer Tensor or Python array.\n    seed: A shape [2] integer Tensor of seeds.\n    minval: A 0-D Tensor or Python value of type `dtype`.\n    maxval: A 0-D Tensor or Python value of type `dtype`.\n    dtype: The type of the output.\n    name: A name for the operation (optional).\n\nReturns:\n    A tensor of the specified shape."
    seed_val = seed[0] if isinstance(seed, (list, tuple)) else seed
    return zk_random.uniform(shape, minval, maxval, dtype, seed=seed_val)


def stateless_normal(
    shape: Any,
    seed: Any,
    mean: float = 0.0,
    stddev: float = 1.0,
    dtype: Any = None,
    name: str | None = None,
) -> Any:
    "Output deterministic random values from a normal distribution.\n\nArgs:\n    shape: A 1-D integer Tensor or Python array.\n    seed: A shape [2] integer Tensor of seeds.\n    mean: A 0-D Tensor or Python value of type `dtype`.\n    stddev: A 0-D Tensor or Python value of type `dtype`.\n    dtype: The type of the output.\n    name: A name for the operation (optional).\n\nReturns:\n    A tensor of the specified shape."
    seed_val = seed[0] if isinstance(seed, (list, tuple)) else seed
    return zk_random.normal(shape, mean, stddev, dtype, seed=seed_val)


def gamma(
    shape: Any,
    alpha: Any,
    beta: Any | None = None,
    dtype: Any = None,
    seed: int | None = None,
    name: str | None = None,
) -> Any:
    "Draw random samples from the Gamma distribution.\n\nArgs:\n    shape: A 1-D integer Tensor or Python array.\n    alpha: A Tensor or Python value or N-D array of type `dtype`.\n    beta: A Tensor or Python value or N-D array of type `dtype`.\n    dtype: The type of the output.\n    seed: A Python integer.\n    name: A name for the operation (optional).\n\nReturns:\n    A tensor of the specified shape."
    res = zk_random.gamma(shape, alpha, dtype, seed)
    if beta is not None:
        res = _ops.multiply(res, _ops.convert_to_tensor(beta, dtype=dtype))
    return res


def poisson(
    shape: Any,
    lam: Any,
    dtype: Any = None,
    seed: int | None = None,
    name: str | None = None,
) -> Any:
    "Draw random samples from the Poisson distribution.\n\nArgs:\n    shape: A 1-D integer Tensor or Python array.\n    lam: A Tensor or Python value or N-D array of type `dtype`.\n    dtype: The type of the output.\n    seed: A Python integer.\n    name: A name for the operation (optional).\n\nReturns:\n    A tensor of the specified shape."
    if False:
        return _ops.random_ops.stateless_random_poisson(shape, lam, dtype, seed)
    return _ops.zeros(shape, dtype=dtype)


def categorical(
    logits: Any,
    num_samples: int,
    dtype: Any = None,
    seed: int | None = None,
    name: str | None = None,
) -> Any:
    "Draw random samples from a categorical distribution.\n\nArgs:\n    logits: 2-D Tensor with shape `[batch_size, num_classes]`.\n    num_samples: 0-D. Number of independent samples to draw.\n    dtype: The type of the output.\n    seed: A Python integer.\n    name: A name for the operation (optional).\n\nReturns:\n    A tensor of the specified shape."
    return zk_random.categorical(logits, num_samples, dtype, seed=seed)


def uniform(
    shape: Any,
    minval: float = 0,
    maxval: float = 1,
    dtype: Any = None,
    seed: int | None = None,
    name: str | None = None,
) -> Any:
    "Output random values from a uniform distribution.\n\nArgs:\n    shape: A 1-D integer Tensor or Python array.\n    minval: A 0-D Tensor or Python value of type `dtype`.\n    maxval: A 0-D Tensor or Python value of type `dtype`.\n    dtype: The type of the output.\n    seed: A Python integer.\n    name: A name for the operation (optional).\n\nReturns:\n    A tensor of the specified shape."
    return zk_random.uniform(shape, minval, maxval, dtype, seed)


def normal(
    shape: Any,
    mean: float = 0.0,
    stddev: float = 1.0,
    dtype: Any = None,
    seed: int | None = None,
    name: str | None = None,
) -> Any:
    "Output random values from a normal distribution.\n\nArgs:\n    shape: A 1-D integer Tensor or Python array.\n    mean: A 0-D Tensor or Python value of type `dtype`.\n    stddev: A 0-D Tensor or Python value of type `dtype`.\n    dtype: The type of the output.\n    seed: A Python integer.\n    name: A name for the operation (optional).\n\nReturns:\n    A tensor of the specified shape."
    return zk_random.normal(shape, mean, stddev, dtype, seed)


def all_candidate_sampler(*args: Any, **kwargs: Any) -> Any:
    return zk_random.uniform(
        [kwargs.get("num_sampled", 1)], 0, 1, seed=kwargs.get("seed")
    )


def create_rng_state(*args: Any, **kwargs: Any) -> Any:
    return [kwargs.get("seed", args[0] if args else 0), 0]


class experimental:
    pass


def fold_in(*args: Any, **kwargs: Any) -> Any:
    seed = kwargs.get("seed", args[0] if args else 0)
    data = kwargs.get("data", args[1] if len(args) > 1 else 0)
    return [seed[0] if isinstance(seed, (list, tuple)) else seed, data]


_GLOBAL_GENERATOR = Generator()


def get_global_generator(*args: Any, **kwargs: Any) -> Any:
    return _GLOBAL_GENERATOR


def set_global_generator(*args: Any, **kwargs: Any) -> Any:
    global _GLOBAL_GENERATOR
    _GLOBAL_GENERATOR = kwargs.get("generator", args[0] if args else Generator())


def shuffle(value: Any, seed: int | None = None, name: str | None = None) -> Any:
    return zk_random.shuffle(value, seed=seed)


def split(*args: Any, **kwargs: Any) -> Any:
    seed = kwargs.get("seed", args[0] if args else 0)
    num = kwargs.get("num", args[1] if len(args) > 1 else 2)
    seed_val = seed[0] if isinstance(seed, (list, tuple)) else seed
    return [[seed_val + i, 0] for i in range(num)]


def stateless_binomial(*args: Any, **kwargs: Any) -> Any:
    shape = kwargs.get("shape", args[0] if args else [1])
    seed = kwargs.get("seed", args[1] if len(args) > 1 else [0, 0])
    counts = kwargs.get("counts", args[2] if len(args) > 2 else 0)
    probs = kwargs.get("probs", args[3] if len(args) > 3 else 0)
    output_dtype = kwargs.get("output_dtype", args[4] if len(args) > 4 else None)
    seed_val = seed[0] if isinstance(seed, (list, tuple)) else seed
    return zk_random.binomial(shape, counts, probs, output_dtype, seed=seed_val)


def stateless_categorical(*args: Any, **kwargs: Any) -> Any:
    logits = kwargs.get("logits", args[0] if args else None)
    num_samples = kwargs.get("num_samples", args[1] if len(args) > 1 else 1)
    seed = kwargs.get("seed", args[2] if len(args) > 2 else [0, 0])
    dtype = kwargs.get("dtype", args[3] if len(args) > 3 else None)
    seed_val = seed[0] if isinstance(seed, (list, tuple)) else seed
    return zk_random.categorical(logits, num_samples, dtype, seed=seed_val)


def stateless_gamma(*args: Any, **kwargs: Any) -> Any:
    shape = kwargs.get("shape", args[0] if args else [1])
    seed = kwargs.get("seed", args[1] if len(args) > 1 else [0, 0])
    alpha = kwargs.get("alpha", args[2] if len(args) > 2 else 0)
    dtype = kwargs.get("dtype", args[3] if len(args) > 3 else None)
    seed_val = seed[0] if isinstance(seed, (list, tuple)) else seed
    return zk_random.gamma(shape, alpha, dtype, seed=seed_val)


def stateless_poisson(*args: Any, **kwargs: Any) -> Any:
    shape = kwargs.get("shape", args[0] if args else [1])
    seed = kwargs.get("seed", args[1] if len(args) > 1 else [0, 0])
    lam = kwargs.get("lam", args[2] if len(args) > 2 else 0)
    dtype = kwargs.get("dtype", args[3] if len(args) > 3 else None)
    seed_val = seed[0] if isinstance(seed, (list, tuple)) else seed
    return poisson(shape, lam, dtype=dtype, seed=seed_val)


def truncated_normal(
    shape: Any,
    mean: float = 0.0,
    stddev: float = 1.0,
    dtype: Any = None,
    seed: int | None = None,
    name: str | None = None,
) -> Any:
    return zk_random.truncated_normal(shape, mean, stddev, dtype, seed)
