import math
from zero_tensorflow.keras.optimizers import schedules


def test_learning_rate_schedule():
    sched = schedules.LearningRateSchedule()
    sched(1)


def test_cosine_decay():
    sched = schedules.CosineDecay(
        initial_learning_rate=0.1,
        decay_steps=100,
        alpha=0.1,
        name="my_decay",
        warmup_target=0.2,
        warmup_steps=10,
    )
    assert sched.initial_learning_rate == 0.1
    assert sched.decay_steps == 100
    assert sched.alpha == 0.1
    assert sched.name == "my_decay"
    assert sched.warmup_target == 0.2
    assert sched.warmup_steps == 10

    # Check warmup
    assert math.isclose(sched(5), 0.15)
    assert math.isclose(sched(10), 0.2)

    # Check decay
    cosine_decay = 0.5 * (1 + math.cos(math.pi * 50 / 100))
    decayed = (1 - 0.1) * cosine_decay + 0.1
    assert math.isclose(sched(50), 0.1 * decayed)


def test_cosine_decay_restarts():
    sched = schedules.CosineDecayRestarts(
        initial_learning_rate=0.1,
        first_decay_steps=100,
        t_mul=2.0,
        m_mul=0.5,
        alpha=0.1,
        name="my_restart",
    )
    assert sched.initial_learning_rate == 0.1
    assert sched.first_decay_steps == 100
    assert sched.t_mul == 2.0
    assert sched.m_mul == 0.5
    assert sched.alpha == 0.1
    assert sched.name == "my_restart"

    # Start
    assert math.isclose(sched(0), 0.1)

    # Halfway through first period
    cosine_decay = 0.5 * (1 + math.cos(math.pi * 50 / 100))
    decayed = (1 - 0.1) * cosine_decay + 0.1
    assert math.isclose(sched(50), 0.1 * decayed)

    # Restart
    assert math.isclose(sched(100), 0.05)


def test_exponential_decay():
    sched = schedules.ExponentialDecay(
        initial_learning_rate=0.1,
        decay_steps=100,
        decay_rate=0.9,
        staircase=True,
        name="my_exp",
    )
    assert sched.initial_learning_rate == 0.1
    assert sched.decay_steps == 100
    assert sched.decay_rate == 0.9
    assert sched.staircase is True
    assert sched.name == "my_exp"

    assert math.isclose(sched(50), 0.1)
    assert math.isclose(sched(100), 0.1 * 0.9)
    assert math.isclose(sched(150), 0.1 * 0.9)

    sched2 = schedules.ExponentialDecay(
        initial_learning_rate=0.1, decay_steps=100, decay_rate=0.9, staircase=False
    )
    assert math.isclose(sched2(50), 0.1 * (0.9**0.5))


def test_inverse_time_decay():
    sched = schedules.InverseTimeDecay(
        initial_learning_rate=0.1,
        decay_steps=100,
        decay_rate=0.9,
        staircase=True,
        name="my_inv",
    )
    assert sched.initial_learning_rate == 0.1
    assert sched.decay_steps == 100
    assert sched.decay_rate == 0.9
    assert sched.staircase is True
    assert sched.name == "my_inv"

    assert math.isclose(sched(50), 0.1)
    assert math.isclose(sched(100), 0.1 / (1.0 + 0.9))

    sched2 = schedules.InverseTimeDecay(
        initial_learning_rate=0.1, decay_steps=100, decay_rate=0.9, staircase=False
    )
    assert math.isclose(sched2(50), 0.1 / (1.0 + 0.9 * 0.5))


def test_piecewise_constant_decay():
    sched = schedules.PiecewiseConstantDecay(
        boundaries=[10, 20], values=[0.1, 0.05, 0.01], name="my_pw"
    )
    assert sched.boundaries == [10, 20]
    assert sched.values == [0.1, 0.05, 0.01]
    assert sched.name == "my_pw"

    assert math.isclose(sched(5), 0.1)
    assert math.isclose(sched(15), 0.05)
    assert math.isclose(sched(25), 0.01)


def test_polynomial_decay():
    sched = schedules.PolynomialDecay(
        initial_learning_rate=0.1,
        decay_steps=100,
        end_learning_rate=0.001,
        power=2.0,
        cycle=True,
        name="my_poly",
    )
    assert sched.initial_learning_rate == 0.1
    assert sched.decay_steps == 100
    assert sched.end_learning_rate == 0.001
    assert sched.power == 2.0
    assert sched.cycle is True
    assert sched.name == "my_poly"

    assert math.isclose(sched(0), 0.1)

    # 50 step inside 100 steps
    expected_50 = (0.1 - 0.001) * ((1 - 50 / 100) ** 2.0) + 0.001
    assert math.isclose(sched(50), expected_50)

    # 150 step with cycle
    expected_150 = (0.1 - 0.001) * ((1 - 150 / 200) ** 2.0) + 0.001
    assert math.isclose(sched(150), expected_150)

    sched2 = schedules.PolynomialDecay(
        initial_learning_rate=0.1,
        decay_steps=100,
        end_learning_rate=0.001,
        power=2.0,
        cycle=False,
    )
    assert math.isclose(sched2(150), 0.001)
