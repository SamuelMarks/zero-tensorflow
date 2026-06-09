from zero_tensorflow.keras import metrics


def test_metric_base():
    _ = metrics.Metric()


def test_auc():
    _ = metrics.AUC()


def test_accuracy():
    _ = metrics.Accuracy()


def test_binary_accuracy():
    _ = metrics.BinaryAccuracy()


def test_binary_crossentropy():
    _ = metrics.BinaryCrossentropy()


def test_binary_iou():
    _ = metrics.BinaryIoU()


def test_categorical_accuracy():
    _ = metrics.CategoricalAccuracy()


def test_categorical_crossentropy():
    _ = metrics.CategoricalCrossentropy()


def test_categorical_hinge():
    _ = metrics.CategoricalHinge()


def test_concordance_correlation():
    _ = metrics.ConcordanceCorrelation()


def test_cosine_similarity():
    _ = metrics.CosineSimilarity()


def test_f1_score():
    _ = metrics.F1Score()


def test_fbeta_score():
    _ = metrics.FBetaScore()


def test_false_negatives():
    _ = metrics.FalseNegatives()


def test_false_positives():
    _ = metrics.FalsePositives()


def test_hinge():
    _ = metrics.Hinge()


def test_iou():
    _ = metrics.IoU(num_classes=2, target_class_ids=[1])


def test_kl_divergence():
    _ = metrics.KLDivergence()


def test_log_cosh_error():
    _ = metrics.LogCoshError()


def test_mean():
    _ = metrics.Mean()


def test_mean_absolute_error():
    _ = metrics.MeanAbsoluteError()


def test_mean_absolute_percentage_error():
    _ = metrics.MeanAbsolutePercentageError()


def test_mean_iou():
    _ = metrics.MeanIoU(num_classes=2)


def test_mean_metric_wrapper():
    def dummy_fn(y_true, y_pred):
        return y_true - y_pred

    _ = metrics.MeanMetricWrapper(fn=dummy_fn)


def test_mean_squared_error():
    _ = metrics.MeanSquaredError()


def test_mean_squared_logarithmic_error():
    _ = metrics.MeanSquaredLogarithmicError()


def test_one_hot_iou():
    _ = metrics.OneHotIoU(num_classes=2, target_class_ids=[1])


def test_one_hot_mean_iou():
    _ = metrics.OneHotMeanIoU(num_classes=2)


def test_pearson_correlation():
    _ = metrics.PearsonCorrelation()


def test_poisson():
    _ = metrics.Poisson()


def test_precision():
    _ = metrics.Precision()


def test_precision_at_recall():
    _ = metrics.PrecisionAtRecall(recall=0.8)


def test_r2_score():
    _ = metrics.R2Score()


def test_recall():
    _ = metrics.Recall()


def test_recall_at_precision():
    _ = metrics.RecallAtPrecision(precision=0.8)


def test_root_mean_squared_error():
    _ = metrics.RootMeanSquaredError()


def test_sensitivity_at_specificity():
    _ = metrics.SensitivityAtSpecificity(specificity=0.8)


def test_sparse_categorical_accuracy():
    _ = metrics.SparseCategoricalAccuracy()


def test_sparse_categorical_crossentropy():
    _ = metrics.SparseCategoricalCrossentropy()


def test_sparse_top_k_categorical_accuracy():
    _ = metrics.SparseTopKCategoricalAccuracy()


def test_specificity_at_sensitivity():
    _ = metrics.SpecificityAtSensitivity(sensitivity=0.8)


def test_squared_hinge():
    _ = metrics.SquaredHinge()


def test_sum():
    _ = metrics.Sum()


def test_top_k_categorical_accuracy():
    _ = metrics.TopKCategoricalAccuracy()


def test_true_negatives():
    _ = metrics.TrueNegatives()


def test_true_positives():
    _ = metrics.TruePositives()
