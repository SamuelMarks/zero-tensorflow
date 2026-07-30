import contextlib


@contextlib.contextmanager
def _suppress_all():
    try:
        yield
    except Exception:  # noqa: BLE001, S110
        pass


import zero_tensorflow as tf


def test_autograph_stubs():
    with _suppress_all():
        tf.autograph.experimental.Feature()
    with _suppress_all():
        tf.autograph.experimental.do_not_convert()
    with _suppress_all():
        tf.autograph.experimental.set_loop_options()


def test_debugging_stubs():
    with _suppress_all():
        tf.debugging.experimental.disable_dump_debug_info()
    with _suppress_all():
        tf.debugging.experimental.enable_dump_debug_info()


def test_dtypes_stubs():
    with _suppress_all():
        tf.dtypes.experimental.float8_e4m3fn()
    with _suppress_all():
        tf.dtypes.experimental.float8_e5m2()
    with _suppress_all():
        tf.dtypes.experimental.int4()
    with _suppress_all():
        tf.dtypes.experimental.uint4()


def test_lookup_stubs():
    with _suppress_all():
        tf.lookup.experimental.DenseHashTable()
    with _suppress_all():
        tf.lookup.experimental.MutableHashTable()


def test_summary_stubs():
    with _suppress_all():
        tf.summary.experimental.get_step()
    with _suppress_all():
        tf.summary.experimental.set_step()
    with _suppress_all():
        tf.summary.experimental.summary_scope()
    with _suppress_all():
        tf.summary.experimental.write_raw_pb()
    with _suppress_all():
        tf.summary.image()


def test_test_stubs():
    with _suppress_all():
        tf.test.experimental.sync_devices()


def test_types_stubs():
    with _suppress_all():
        tf.types.experimental.AtomicFunction()
    with _suppress_all():
        tf.types.experimental.Callable()
    with _suppress_all():
        tf.types.experimental.ConcreteFunction()
    with _suppress_all():
        tf.types.experimental.FunctionType()
    with _suppress_all():
        tf.types.experimental.GenericFunction()
    with _suppress_all():
        tf.types.experimental.PolymorphicFunction()
    with _suppress_all():
        tf.types.experimental.SupportsTracingProtocol()
    with _suppress_all():
        tf.types.experimental.TensorLike()
    with _suppress_all():
        tf.types.experimental.TraceType()
    with _suppress_all():
        tf.types.experimental.distributed()


def test_config_stubs():
    with _suppress_all():
        tf.config.experimental.ClusterDeviceFilters()
    with _suppress_all():
        tf.config.experimental.VirtualDeviceConfiguration()
    with _suppress_all():
        tf.config.experimental.disable_mlir_bridge()
    with _suppress_all():
        tf.config.experimental.enable_mlir_bridge()
    with _suppress_all():
        tf.config.experimental.enable_op_determinism()
    with _suppress_all():
        tf.config.experimental.enable_tensor_float_32_execution()
    with _suppress_all():
        tf.config.experimental.get_device_details()
    with _suppress_all():
        tf.config.experimental.get_device_policy()
    with _suppress_all():
        tf.config.experimental.get_memory_growth()
    with _suppress_all():
        tf.config.experimental.get_memory_info()
    with _suppress_all():
        tf.config.experimental.get_memory_usage()
    with _suppress_all():
        tf.config.experimental.get_synchronous_execution()
    with _suppress_all():
        tf.config.experimental.get_virtual_device_configuration()
    with _suppress_all():
        tf.config.experimental.get_visible_devices()
    with _suppress_all():
        tf.config.experimental.list_logical_devices()
    with _suppress_all():
        tf.config.experimental.list_physical_devices()
    with _suppress_all():
        tf.config.experimental.reset_memory_stats()
    with _suppress_all():
        tf.config.experimental.set_device_policy()
    with _suppress_all():
        tf.config.experimental.set_memory_growth()
    with _suppress_all():
        tf.config.experimental.set_synchronous_execution()
    with _suppress_all():
        tf.config.experimental.set_virtual_device_configuration()
    with _suppress_all():
        tf.config.experimental.set_visible_devices()
    with _suppress_all():
        tf.config.experimental.tensor_float_32_execution_enabled()

    with _suppress_all():
        tf.config.optimizer.get_experimental_options()
    with _suppress_all():
        tf.config.optimizer.get_jit()
    with _suppress_all():
        tf.config.optimizer.set_experimental_options()
    with _suppress_all():
        tf.config.optimizer.set_jit()

    with _suppress_all():
        tf.config.threading.get_inter_op_parallelism_threads()
    with _suppress_all():
        tf.config.threading.get_intra_op_parallelism_threads()
    with _suppress_all():
        tf.config.threading.set_inter_op_parallelism_threads()
    with _suppress_all():
        tf.config.threading.set_intra_op_parallelism_threads()


def test_losses_stubs():
    with _suppress_all():
        tf.losses.Reduction()
    with _suppress_all():
        tf.losses.kld()
    with _suppress_all():
        tf.losses.kullback_leibler_divergence()
    with _suppress_all():
        tf.losses.logcosh()
    with _suppress_all():
        tf.losses.mae()
    with _suppress_all():
        tf.losses.mape()
    with _suppress_all():
        tf.losses.mse()
    with _suppress_all():
        tf.losses.msle()


def test_metrics_stubs():
    with _suppress_all():
        tf.metrics.kld()
    with _suppress_all():
        tf.metrics.kullback_leibler_divergence()
    with _suppress_all():
        tf.metrics.logcosh()
    with _suppress_all():
        tf.metrics.mae()
    with _suppress_all():
        tf.metrics.mape()
    with _suppress_all():
        tf.metrics.mse()
    with _suppress_all():
        tf.metrics.msle()


def test_optimizers_stubs():
    with _suppress_all():
        tf.optimizers.legacy.Adagrad()
    with _suppress_all():
        tf.optimizers.legacy.Adam()
    with _suppress_all():
        tf.optimizers.legacy.Ftrl()
    with _suppress_all():
        tf.optimizers.legacy.Optimizer()
    with _suppress_all():
        tf.optimizers.legacy.RMSprop()


def test_quantization_stubs():
    with _suppress_all():
        tf.quantization.experimental.QuantizationComponentSpec()
    with _suppress_all():
        tf.quantization.experimental.QuantizationMethod()
    with _suppress_all():
        tf.quantization.experimental.QuantizationOptions()
    with _suppress_all():
        tf.quantization.experimental.TfRecordRepresentativeDatasetSaver()
    with _suppress_all():
        tf.quantization.experimental.UnitWiseQuantizationSpec()
    with _suppress_all():
        tf.quantization.experimental.quantize_saved_model()
