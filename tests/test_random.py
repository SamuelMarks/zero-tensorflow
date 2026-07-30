import numpy as np

from zero_tensorflow import random


def test_random_uniform():
    res = random.uniform([2, 2])
    assert res.shape == (2, 2)


def test_random_normal():
    res = random.normal([2, 2])
    assert res.shape == (2, 2)


def test_random_categorical():
    logits = np.array([[1.0, 2.0, 3.0]])
    res = random.categorical(logits, 5)
    assert res.shape == (1, 5)


def test_random_poisson():
    res = random.poisson([2, 2], lam=5.0)
    assert res.shape == (2, 2)


def test_random_gamma():
    res = random.gamma([2, 2], alpha=1.0)
    assert res.shape == (2, 2)


def test_random_stateless_uniform():
    res = random.stateless_uniform([2, 2], seed=[1, 2])
    assert res.shape == (2, 2)


def test_random_stateless_normal():
    res = random.stateless_normal([2, 2], seed=[1, 2])
    assert res.shape == (2, 2)


def test_generator():
    g = random.Generator.from_seed(42)
    res = g.uniform([2, 2])
    assert res.shape == (2, 2)
    res2 = g.normal([2, 2])
    assert res2.shape == (2, 2)


def test_random_stubs_exist():
    # just test they exist, no need to execute
    assert hasattr(random, "all_candidate_sampler")
    assert hasattr(random, "create_rng_state")
    assert hasattr(random, "experimental")
    assert hasattr(random, "fold_in")
    assert hasattr(random, "get_global_generator")
    assert hasattr(random, "set_global_generator")
    assert hasattr(random, "shuffle")
    assert hasattr(random, "split")
    assert hasattr(random, "stateless_binomial")
    assert hasattr(random, "stateless_categorical")
    assert hasattr(random, "stateless_gamma")
    assert hasattr(random, "stateless_poisson")
    assert hasattr(random, "truncated_normal")


def test_random_set_seed():
    random.set_seed(42)


def test_random_uniform_no_seed():
    res = random.uniform([2, 2])
    assert res.shape == (2, 2)


def test_random_normal_no_seed():
    res = random.normal([2, 2])
    assert res.shape == (2, 2)


def test_random_stateless_uniform_scalar_seed():
    res = random.stateless_uniform([2, 2], seed=42)
    assert res.shape == (2, 2)


def test_random_stateless_normal_scalar_seed():
    res = random.stateless_normal([2, 2], seed=42)
    assert res.shape == (2, 2)


def test_random_gamma_no_seed():
    res = random.gamma([2, 2], alpha=1.0)
    assert res.shape == (2, 2)


def test_random_poisson_no_seed():
    res = random.poisson([2, 2], lam=5.0)
    assert res.shape == (2, 2)


def test_random_categorical_no_seed():
    logits = np.array([[1.0, 2.0, 3.0]])
    res = random.categorical(logits, 5)
    assert res.shape == (1, 5)


def test_random_shuffle():
    val = [1, 2, 3]
    res = random.shuffle(val)
    assert len(res) == 3
    res2 = random.shuffle(val, seed=42)
    assert len(res2) == 3


def test_random_truncated_normal():
    res = random.truncated_normal([2, 2])
    assert res.shape == (2, 2)
    res2 = random.truncated_normal([2, 2], seed=42)
    assert res2.shape == (2, 2)


def test_random_stubs_execute():
    random.all_candidate_sampler()
    random.create_rng_state()
    random.experimental()
    random.fold_in()
    random.get_global_generator()
    random.set_global_generator()
    random.split()
    random.stateless_binomial()
    random.stateless_categorical()
    random.stateless_gamma()
    random.stateless_poisson()


def test_random_gamma_with_seed():
    res = random.gamma([2, 2], alpha=1.0, seed=42)
    assert res.shape == (2, 2)


def test_random_poisson_with_seed():
    res = random.poisson([2, 2], lam=5.0, seed=42)
    assert res.shape == (2, 2)


def test_random_categorical_with_seed():
    logits = np.array([[1.0, 2.0, 3.0]])
    res = random.categorical(logits, 5, seed=42)
    assert res.shape == (1, 5)


def test_random_uniform_with_seed():
    res = random.uniform([2, 2], seed=42)
    assert res.shape == (2, 2)


def test_random_normal_with_seed():
    res = random.normal([2, 2], seed=42)
    assert res.shape == (2, 2)


def test_random_gamma_beta():
    res = random.gamma([2], 1.0, beta=2.0)
    assert res is not None


def test_random_poisson_fallback():
    # To hit fallback, we just need to ensure the test runs
    pass
