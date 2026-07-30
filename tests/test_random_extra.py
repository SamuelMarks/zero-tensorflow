from zero_tensorflow.random import (
    Generator,
    create_rng_state,
    experimental,
    fold_in,
    get_global_generator,
    set_global_generator,
    split,
)


def test_global_generator():
    gen1 = get_global_generator()
    assert isinstance(gen1, Generator)

    gen2 = Generator(seed=42)
    set_global_generator(gen2)
    assert get_global_generator() is gen2


def test_rng_state_helpers():
    state = create_rng_state(42, "philox")
    assert state == [42, 0]

    folded = fold_in([42, 0], 5)
    assert folded == [42, 5]

    folded_scalar = fold_in(42, 5)
    assert folded_scalar == [42, 5]

    split_states = split([42, 0], num=3)
    assert split_states == [[42, 0], [43, 0], [44, 0]]

    split_states_scalar = split(42, num=2)
    assert split_states_scalar == [[42, 0], [43, 0]]


def test_experimental_namespace():
    assert isinstance(experimental(), experimental)
