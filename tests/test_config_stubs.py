from zero_tensorflow.config import experimental, optimizer, threading


def test_config_stubs():
    # experimental classes
    assert experimental.ClusterDeviceFilters() is not None
    assert experimental.VirtualDeviceConfiguration() is not None

    # experimental functions
    assert experimental.disable_mlir_bridge() is None
    assert experimental.enable_mlir_bridge() is None
    assert experimental.enable_op_determinism() is None
    assert experimental.enable_tensor_float_32_execution() is None
    assert experimental.get_device_details() is None
    assert experimental.get_device_policy() is None
    assert experimental.get_memory_growth() is None
    assert experimental.get_memory_info() is None
    assert experimental.get_memory_usage() is None
    assert experimental.get_synchronous_execution() is None
    assert experimental.get_virtual_device_configuration() is None
    assert experimental.get_visible_devices() is None
    assert experimental.list_logical_devices() is None
    assert experimental.list_physical_devices() is None
    assert experimental.reset_memory_stats() is None
    assert experimental.set_device_policy() is None
    assert experimental.set_memory_growth() is None
    assert experimental.set_synchronous_execution() is None
    assert experimental.set_virtual_device_configuration() is None
    assert experimental.set_visible_devices() is None
    assert experimental.tensor_float_32_execution_enabled() is None

    # optimizer functions
    assert optimizer.get_experimental_options() is None
    assert optimizer.get_jit() is None
    assert optimizer.set_experimental_options() is None
    assert optimizer.set_jit() is None

    # threading functions
    assert threading.get_inter_op_parallelism_threads() is None
    assert threading.get_intra_op_parallelism_threads() is None
    assert threading.set_inter_op_parallelism_threads() is None
    assert threading.set_intra_op_parallelism_threads() is None
