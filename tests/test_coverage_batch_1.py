"""Unit tests for `test_coverage_batch_1`."""

from __future__ import annotations

import zero_tensorflow as tf

import numpy as np


def test_autograph_stubs() -> None:
    """Verify `tf.autograph.experimental` classes and functions."""
    feat = tf.autograph.experimental.Feature()
    assert feat is not None

    def fn(*args: tf.Tensor | float, **kwargs: tf.Tensor | float) -> float:
        """Return first numeric argument."""
        del kwargs
        return 1.0 if not args else 2.0

    assert tf.autograph.experimental.do_not_convert(fn) is fn
    assert tf.autograph.experimental.set_loop_options(parallel_iterations=2) is None


def test_debugging_stubs() -> None:
    """Verify `tf.debugging.experimental` functions."""
    assert tf.debugging.experimental.disable_dump_debug_info() is None
    assert tf.debugging.experimental.enable_dump_debug_info("/tmp") is None


def test_dtypes_stubs() -> None:
    """Verify `tf.dtypes.experimental` low-bitwidth dtypes."""
    assert tf.dtypes.experimental.float8_e4m3fn is not None
    assert tf.dtypes.experimental.float8_e5m2 is not None
    assert tf.dtypes.experimental.int4 is not None
    assert tf.dtypes.experimental.uint4 is not None


def test_lookup_stubs() -> None:
    """Verify `tf.lookup.experimental` hash table classes."""
    dht = tf.lookup.experimental.DenseHashTable(tf.int64, tf.int64, -1, -1, -2)
    mht = tf.lookup.experimental.MutableHashTable(tf.int64, tf.int64, -1)
    assert dht.args == (tf.int64, tf.int64, -1, -1, -2)
    assert mht.args == (tf.int64, tf.int64, -1)


def test_summary_stubs() -> None:
    """Verify `tf.summary` and `tf.summary.experimental` functions."""
    tf.summary.experimental.set_step(10)
    assert tf.summary.experimental.get_step() == 10
    with tf.summary.experimental.summary_scope("scope_a") as (s_name, s_val):
        assert s_name == "summary_scope"
        assert s_val is None
    assert tf.summary.experimental.write_raw_pb(b"pb") is None
    tf.summary.image("img", [[1.0]])


def test_test_stubs() -> None:
    """Verify `tf.test.experimental.sync_devices`."""
    assert tf.test.experimental.sync_devices() is None


def test_types_stubs() -> None:
    """Verify `tf.types.experimental` classes and functions."""
    assert tf.types.experimental.AtomicFunction() is not None
    assert tf.types.experimental.Callable() is not None
    assert tf.types.experimental.ConcreteFunction() is not None
    assert tf.types.experimental.FunctionType() is not None
    assert tf.types.experimental.GenericFunction() is not None
    assert tf.types.experimental.PolymorphicFunction() is not None
    assert tf.types.experimental.SupportsTracingProtocol() is not None
    assert tf.types.experimental.TensorLike() is not None
    assert tf.types.experimental.TraceType() is not None
    assert tf.types.experimental.distributed() is None


def test_config_stubs() -> None:
    """Verify `tf.config` experimental, optimizer, and threading APIs."""
    assert tf.config.experimental.ClusterDeviceFilters() is not None
    assert tf.config.experimental.VirtualDeviceConfiguration() is not None
    assert tf.config.experimental.disable_mlir_bridge() is None
    assert tf.config.experimental.enable_mlir_bridge() is None
    assert tf.config.experimental.enable_op_determinism() is None
    assert tf.config.experimental.enable_tensor_float_32_execution(True) is None
    assert tf.config.experimental.get_device_details("CPU:0") is None
    assert tf.config.experimental.get_device_policy() is None
    assert tf.config.experimental.get_memory_growth("GPU:0") is None
    assert tf.config.experimental.get_memory_info("GPU:0") is None
    assert tf.config.experimental.get_memory_usage("GPU:0") is None
    assert tf.config.experimental.get_synchronous_execution() is None
    assert tf.config.experimental.get_virtual_device_configuration("GPU:0") is None
    assert tf.config.experimental.get_visible_devices() is None
    assert tf.config.experimental.list_logical_devices() is None
    assert tf.config.experimental.list_physical_devices() is None
    assert tf.config.experimental.reset_memory_stats("GPU:0") is None
    assert tf.config.experimental.set_device_policy("explicit") is None
    assert tf.config.experimental.set_memory_growth("GPU:0", True) is None
    assert tf.config.experimental.set_synchronous_execution(True) is None
    assert tf.config.experimental.set_virtual_device_configuration("GPU:0", []) is None
    assert tf.config.experimental.set_visible_devices([], "GPU") is None
    assert tf.config.experimental.tensor_float_32_execution_enabled() is None
    assert tf.config.optimizer.get_experimental_options() is None
    assert tf.config.optimizer.get_jit() is None
    assert tf.config.optimizer.set_experimental_options({}) is None
    assert tf.config.optimizer.set_jit(True) is None
    assert tf.config.threading.get_inter_op_parallelism_threads() is None
    assert tf.config.threading.get_intra_op_parallelism_threads() is None
    assert tf.config.threading.set_inter_op_parallelism_threads(2) is None
    assert tf.config.threading.set_intra_op_parallelism_threads(2) is None


def test_losses_stubs() -> None:
    """Verify `tf.losses` functions and `Reduction` enum."""
    assert tf.losses.Reduction("auto") == tf.losses.Reduction.AUTO
    y_true = tf.Tensor([1.0, 2.0])
    y_pred = tf.Tensor([1.0, 2.0])
    np.testing.assert_allclose(tf.losses.mse(y_true, y_pred).numpy(), 0.0)
    np.testing.assert_allclose(tf.losses.mae(y_true, y_pred).numpy(), 0.0)
    np.testing.assert_allclose(tf.losses.mape(y_true, y_pred).numpy(), 0.0)
    np.testing.assert_allclose(tf.losses.msle(y_true, y_pred).numpy(), 0.0)
    np.testing.assert_allclose(tf.losses.logcosh(y_true, y_pred).numpy(), 0.0)
    np.testing.assert_allclose(tf.losses.kld(y_true, y_pred).numpy(), 0.0, atol=1e-5)
    np.testing.assert_allclose(
        tf.losses.kullback_leibler_divergence(y_true, y_pred).numpy(), 0.0, atol=1e-5
    )


def test_metrics_stubs() -> None:
    """Verify `tf.metrics` functional metric aliases."""
    y_true = tf.Tensor([1.0, 2.0])
    y_pred = tf.Tensor([1.0, 2.0])
    np.testing.assert_allclose(tf.metrics.mse(y_true, y_pred).numpy(), 0.0)
    np.testing.assert_allclose(tf.metrics.mae(y_true, y_pred).numpy(), 0.0)
    np.testing.assert_allclose(tf.metrics.mape(y_true, y_pred).numpy(), 0.0)
    np.testing.assert_allclose(tf.metrics.msle(y_true, y_pred).numpy(), 0.0)
    np.testing.assert_allclose(tf.metrics.logcosh(y_true, y_pred).numpy(), 0.0)
    np.testing.assert_allclose(tf.metrics.kld(y_true, y_pred).numpy(), 0.0, atol=1e-5)
    np.testing.assert_allclose(
        tf.metrics.kullback_leibler_divergence(y_true, y_pred).numpy(), 0.0, atol=1e-5
    )


def test_optimizers_stubs() -> None:
    """Verify `tf.optimizers.legacy` optimizer classes."""
    assert tf.optimizers.legacy.Adagrad(learning_rate=0.01) is not None
    assert tf.optimizers.legacy.Adam(learning_rate=0.01) is not None
    assert tf.optimizers.legacy.Ftrl(learning_rate=0.01) is not None
    assert tf.optimizers.legacy.RMSprop(learning_rate=0.01) is not None
    assert tf.optimizers.legacy.SGD(learning_rate=0.01) is not None


def test_quantization_stubs() -> None:
    """Verify `tf.quantization.experimental` classes and functions."""
    assert tf.quantization.experimental.QuantizationComponentSpec() is not None
    assert tf.quantization.experimental.QuantizationMethod() is not None
    assert tf.quantization.experimental.QuantizationOptions() is not None
    assert tf.quantization.experimental.TfRecordRepresentativeDatasetSaver() is not None
    assert tf.quantization.experimental.UnitWiseQuantizationSpec() is not None
    assert tf.quantization.experimental.quantize_saved_model() is None
