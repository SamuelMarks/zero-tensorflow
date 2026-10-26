"""Unit tests for `zero_tensorflow.config` stateful configuration."""

from __future__ import annotations

import zero_tensorflow.config as tf_config
from zero_tensorflow.config import (
    LogicalDevice,
    LogicalDeviceConfiguration,
    PhysicalDevice,
    experimental,
    optimizer,
    threading,
)


def test_config_devices_and_functions() -> None:
    """Verify device discovery, logical device configuration, and function execution modes."""
    cpu = PhysicalDevice(name="/physical_device:CPU:0", device_type="CPU")
    gpu = PhysicalDevice(name="/physical_device:GPU:0", device_type="GPU")
    lcpu = LogicalDevice(name="/device:CPU:0", device_type="CPU")
    assert cpu.device_type == "CPU"
    assert lcpu.device_type == "CPU"

    tf_config.set_visible_devices(cpu)
    assert len(tf_config.get_visible_devices()) == 1
    tf_config.set_visible_devices([cpu, gpu])
    assert len(tf_config.list_physical_devices()) == 2
    assert len(tf_config.list_physical_devices("GPU")) == 1
    assert len(tf_config.list_logical_devices()) == 1
    assert len(tf_config.list_logical_devices("CPU")) == 1
    assert tf_config.list_logical_devices("TPU") == []

    tf_config.set_soft_device_placement(False)
    assert tf_config.get_soft_device_placement() is False
    tf_config.set_soft_device_placement(True)
    assert tf_config.get_soft_device_placement() is True

    vcfg = LogicalDeviceConfiguration(memory_limit=1024, experimental_priority=1)
    assert vcfg.memory_limit == 1024
    tf_config.set_logical_device_configuration(cpu, [vcfg])
    assert tf_config.get_logical_device_configuration(cpu) == [vcfg]
    tf_config.set_logical_device_configuration(cpu, None)
    assert tf_config.get_logical_device_configuration(cpu) == []

    tf_config.run_functions_eagerly(True)
    assert tf_config.functions_run_eagerly() is True
    tf_config.experimental_run_functions_eagerly(False)
    assert tf_config.experimental_functions_run_eagerly() is False

    tf_config.experimental_connect_to_cluster("localhost")
    tf_config.experimental_connect_to_host("localhost:1234")


def test_config_stubs() -> None:
    """Verify stateful getters and setters across `experimental`, `optimizer`, and `threading`."""
    cpu = PhysicalDevice(name="/physical_device:CPU:0", device_type="CPU")
    cdf = experimental.ClusterDeviceFilters()
    cdf.set_device_filters("worker", 0, ["/device:CPU:0"])
    assert cdf._filters[("worker", 0)] == ["/device:CPU:0"]

    vdc = experimental.VirtualDeviceConfiguration(memory_limit=512)
    assert vdc.memory_limit == 512

    experimental.enable_mlir_bridge()
    experimental.disable_mlir_bridge()
    experimental.enable_op_determinism()
    experimental.enable_tensor_float_32_execution(False)
    assert experimental.tensor_float_32_execution_enabled() is False
    experimental.enable_tensor_float_32_execution()
    assert experimental.tensor_float_32_execution_enabled() is True

    assert experimental.get_device_details(cpu) == {"device_name": "CPU:0"}
    experimental.set_device_policy("silent")
    assert experimental.get_device_policy() == "silent"
    experimental.set_device_policy()
    assert experimental.get_device_policy() == "explicit"

    experimental.set_memory_growth(cpu, True)
    assert experimental.get_memory_growth(cpu) is True
    experimental.set_memory_growth(False)
    assert experimental.get_memory_growth() is False

    assert experimental.get_memory_info("CPU:0") == {"current": 0, "peak": 0}
    assert experimental.get_memory_usage("CPU:0") == 0
    experimental.reset_memory_stats("CPU:0")

    experimental.set_synchronous_execution(False)
    assert experimental.get_synchronous_execution() is False
    experimental.set_synchronous_execution()
    assert experimental.get_synchronous_execution() is True

    experimental.set_virtual_device_configuration(cpu, [vdc])
    assert experimental.get_virtual_device_configuration(cpu) == [vdc]
    experimental.set_virtual_device_configuration(cpu)
    assert experimental.get_virtual_device_configuration(cpu) == []

    experimental.set_visible_devices(cpu)
    assert len(experimental.get_visible_devices("CPU")) == 1
    experimental.set_visible_devices([cpu])
    assert len(experimental.list_physical_devices("CPU")) == 1
    assert len(experimental.list_logical_devices("CPU")) == 1
    experimental.set_visible_devices()

    optimizer.set_jit(True)
    assert optimizer.get_jit() is True
    optimizer.set_jit("autoclustering")
    assert optimizer.get_jit() == "autoclustering"
    optimizer.set_jit()
    assert optimizer.get_jit() is False

    optimizer.set_experimental_options({"layout_optimizer": True})
    assert optimizer.get_experimental_options() == {"layout_optimizer": True}
    optimizer.set_experimental_options(constant_folding=True)
    assert optimizer.get_experimental_options() == {"constant_folding": True}
    optimizer.set_experimental_options()

    threading.set_inter_op_parallelism_threads(4)
    assert threading.get_inter_op_parallelism_threads() == 4
    threading.set_inter_op_parallelism_threads()
    assert threading.get_inter_op_parallelism_threads() == 0

    threading.set_intra_op_parallelism_threads(8)
    assert threading.get_intra_op_parallelism_threads() == 8
    threading.set_intra_op_parallelism_threads()
    assert threading.get_intra_op_parallelism_threads() == 0
