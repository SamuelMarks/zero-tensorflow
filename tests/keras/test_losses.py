from zero_tensorflow.keras import losses


def test_loss_base():
    _ = losses.Loss()


def test_reduction():
    losses.Reduction()


def test_binary_crossentropy():
    _ = losses.BinaryCrossentropy()


def test_binary_focal_crossentropy():
    _ = losses.BinaryFocalCrossentropy()


def test_ctc():
    _ = losses.CTC()


def test_categorical_crossentropy():
    _ = losses.CategoricalCrossentropy()


def test_categorical_focal_crossentropy():
    _ = losses.CategoricalFocalCrossentropy()


def test_categorical_generalized_cross_entropy():
    _ = losses.CategoricalGeneralizedCrossEntropy()


def test_categorical_hinge():
    _ = losses.CategoricalHinge()


def test_circle():
    _ = losses.Circle()


def test_cosine_similarity():
    _ = losses.CosineSimilarity()


def test_dice():
    _ = losses.Dice()


def test_hinge():
    _ = losses.Hinge()


def test_huber():
    _ = losses.Huber()


def test_kl_divergence():
    _ = losses.KLDivergence()


def test_log_cosh():
    _ = losses.LogCosh()


def test_mean_absolute_error():
    _ = losses.MeanAbsoluteError()


def test_mean_absolute_percentage_error():
    _ = losses.MeanAbsolutePercentageError()


def test_mean_squared_error():
    _ = losses.MeanSquaredError()


def test_mean_squared_logarithmic_error():
    _ = losses.MeanSquaredLogarithmicError()


def test_poisson():
    _ = losses.Poisson()


def test_sparse_categorical_crossentropy():
    _ = losses.SparseCategoricalCrossentropy()


def test_squared_hinge():
    _ = losses.SquaredHinge()


def test_tversky():
    _ = losses.Tversky()
