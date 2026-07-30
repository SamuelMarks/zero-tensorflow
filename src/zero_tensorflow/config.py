"""Stub module for config."""

from typing import Any


class experimental:
    """Stub for experimental module."""

    class ClusterDeviceFilters:
        """Stub for ClusterDeviceFilters."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class VirtualDeviceConfiguration:
        """Stub for VirtualDeviceConfiguration."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    @staticmethod
    def disable_mlir_bridge(*args: Any, **kwargs: Any) -> None:
        """Stub for disable_mlir_bridge."""
        return

    @staticmethod
    def enable_mlir_bridge(*args: Any, **kwargs: Any) -> None:
        """Stub for enable_mlir_bridge."""
        return

    @staticmethod
    def enable_op_determinism(*args: Any, **kwargs: Any) -> None:
        """Stub for enable_op_determinism."""
        return

    @staticmethod
    def enable_tensor_float_32_execution(*args: Any, **kwargs: Any) -> None:
        """Stub for enable_tensor_float_32_execution."""
        return

    @staticmethod
    def get_device_details(*args: Any, **kwargs: Any) -> None:
        """Stub for get_device_details."""
        return

    @staticmethod
    def get_device_policy(*args: Any, **kwargs: Any) -> None:
        """Stub for get_device_policy."""
        return

    @staticmethod
    def get_memory_growth(*args: Any, **kwargs: Any) -> None:
        """Stub for get_memory_growth."""
        return

    @staticmethod
    def get_memory_info(*args: Any, **kwargs: Any) -> None:
        """Stub for get_memory_info."""
        return

    @staticmethod
    def get_memory_usage(*args: Any, **kwargs: Any) -> None:
        """Stub for get_memory_usage."""
        return

    @staticmethod
    def get_synchronous_execution(*args: Any, **kwargs: Any) -> None:
        """Stub for get_synchronous_execution."""
        return

    @staticmethod
    def get_virtual_device_configuration(*args: Any, **kwargs: Any) -> None:
        """Stub for get_virtual_device_configuration."""
        return

    @staticmethod
    def get_visible_devices(*args: Any, **kwargs: Any) -> None:
        """Stub for get_visible_devices."""
        return

    @staticmethod
    def list_logical_devices(*args: Any, **kwargs: Any) -> None:
        """Stub for list_logical_devices."""
        return

    @staticmethod
    def list_physical_devices(*args: Any, **kwargs: Any) -> None:
        """Stub for list_physical_devices."""
        return

    @staticmethod
    def reset_memory_stats(*args: Any, **kwargs: Any) -> None:
        """Stub for reset_memory_stats."""
        return

    @staticmethod
    def set_device_policy(*args: Any, **kwargs: Any) -> None:
        """Stub for set_device_policy."""
        return

    @staticmethod
    def set_memory_growth(*args: Any, **kwargs: Any) -> None:
        """Stub for set_memory_growth."""
        return

    @staticmethod
    def set_synchronous_execution(*args: Any, **kwargs: Any) -> None:
        """Stub for set_synchronous_execution."""
        return

    @staticmethod
    def set_virtual_device_configuration(*args: Any, **kwargs: Any) -> None:
        """Stub for set_virtual_device_configuration."""
        return

    @staticmethod
    def set_visible_devices(*args: Any, **kwargs: Any) -> None:
        """Stub for set_visible_devices."""
        return

    @staticmethod
    def tensor_float_32_execution_enabled(*args: Any, **kwargs: Any) -> None:
        """Stub for tensor_float_32_execution_enabled."""
        return


class optimizer:
    """Stub for optimizer module."""

    @staticmethod
    def get_experimental_options(*args: Any, **kwargs: Any) -> None:
        """Stub for get_experimental_options."""
        return

    @staticmethod
    def get_jit(*args: Any, **kwargs: Any) -> None:
        """Stub for get_jit."""
        return

    @staticmethod
    def set_experimental_options(*args: Any, **kwargs: Any) -> None:
        """Stub for set_experimental_options."""
        return

    @staticmethod
    def set_jit(*args: Any, **kwargs: Any) -> None:
        """Stub for set_jit."""
        return


class threading:
    """Stub for threading module."""

    @staticmethod
    def get_inter_op_parallelism_threads(*args: Any, **kwargs: Any) -> None:
        """Stub for get_inter_op_parallelism_threads."""
        return

    @staticmethod
    def get_intra_op_parallelism_threads(*args: Any, **kwargs: Any) -> None:
        """Stub for get_intra_op_parallelism_threads."""
        return

    @staticmethod
    def set_inter_op_parallelism_threads(*args: Any, **kwargs: Any) -> None:
        """Stub for set_inter_op_parallelism_threads."""
        return

    @staticmethod
    def set_intra_op_parallelism_threads(*args: Any, **kwargs: Any) -> None:
        """Stub for set_intra_op_parallelism_threads."""
        return
