"""TensorFlow runtime and device configuration module."""

from __future__ import annotations

from collections.abc import Sequence

from zero_tensorflow._typing import TensorOperand

from typing_extensions import TypeAlias

__all__ = [
    "LogicalDevice",
    "LogicalDeviceConfiguration",
    "PhysicalDevice",
    "experimental",
    "experimental_connect_to_cluster",
    "experimental_connect_to_host",
    "experimental_functions_run_eagerly",
    "experimental_run_functions_eagerly",
    "functions_run_eagerly",
    "get_logical_device_configuration",
    "get_soft_device_placement",
    "get_visible_devices",
    "list_logical_devices",
    "list_physical_devices",
    "optimizer",
    "run_functions_eagerly",
    "set_logical_device_configuration",
    "set_soft_device_placement",
    "set_visible_devices",
    "threading",
]


class PhysicalDevice:
    """Abstraction for a locally visible physical compute device (`CPU`, `GPU`, `TPU`)."""

    def __init__(self, name: str = "/physical_device:CPU:0", device_type: str = "CPU") -> None:
        """Initialize `PhysicalDevice`.

        Args:
            name: Canonical physical device name.
            device_type: Device type string (`'CPU'`, `'GPU'`, or `'TPU'`).
        """
        self.name = name
        self.device_type = device_type


class LogicalDevice:
    """Abstraction for a runtime-initialized logical compute device."""

    def __init__(self, name: str = "/device:CPU:0", device_type: str = "CPU") -> None:
        """Initialize `LogicalDevice`.

        Args:
            name: Canonical logical device name.
            device_type: Device type string (`'CPU'`, `'GPU'`, or `'TPU'`).
        """
        self.name = name
        self.device_type = device_type


class LogicalDeviceConfiguration:
    """Configuration parameters for virtual/logical devices on a `PhysicalDevice`."""

    def __init__(
        self,
        memory_limit: int | None = None,
        experimental_priority: int | None = None,
        experimental_device_ordinal: int | None = None,
    ) -> None:
        """Initialize `LogicalDeviceConfiguration`.

        Args:
            memory_limit: Optional maximum memory in megabytes.
            experimental_priority: Optional scheduling priority.
            experimental_device_ordinal: Optional device ordinal.
        """
        self.memory_limit = memory_limit
        self.experimental_priority = experimental_priority
        self.experimental_device_ordinal = experimental_device_ordinal


_ConfigArg: TypeAlias = (
    "PhysicalDevice | LogicalDevice | LogicalDeviceConfiguration "
    "| Sequence[PhysicalDevice] | Sequence[LogicalDeviceConfiguration] | TensorOperand"
)
_DEFAULT_CPU = PhysicalDevice(name="/physical_device:CPU:0", device_type="CPU")
_DEFAULT_LOGICAL_CPU = LogicalDevice(name="/device:CPU:0", device_type="CPU")


class _ConfigRuntimeState:
    """In-memory runtime state backing `tf.config` getters and setters."""

    def __init__(self) -> None:
        """Initialize default runtime configuration state."""
        self.visible_devices: list[PhysicalDevice] = [_DEFAULT_CPU]
        self.soft_device_placement: bool = True
        self.run_eagerly: bool = False
        self.mlir_bridge: bool = False
        self.op_determinism: bool = False
        self.tf32_enabled: bool = True
        self.synchronous_execution: bool = True
        self.device_policy: str = "explicit"
        self.memory_growth: bool = False
        self.virtual_configs: list[LogicalDeviceConfiguration] = []
        self.jit_enabled: bool | str = False
        self.experimental_options: dict[str, bool] = {}
        self.inter_op_threads: int = 0
        self.intra_op_threads: int = 0


_STATE = _ConfigRuntimeState()


def list_physical_devices(device_type: str | None = None) -> list[PhysicalDevice]:
    """Return a list of physical devices visible to the host runtime.

    Args:
        device_type: Optional device type filter (e.g., `'CPU'`, `'GPU'`).

    Returns:
        list[PhysicalDevice]: Matching physical devices.
    """
    if device_type is None:
        return list(_STATE.visible_devices)
    return [d for d in _STATE.visible_devices if d.device_type.upper() == device_type.upper()]


def list_logical_devices(device_type: str | None = None) -> list[LogicalDevice]:
    """Return a list of logical devices created by the runtime.

    Args:
        device_type: Optional device type filter (e.g., `'CPU'`, `'GPU'`).

    Returns:
        list[LogicalDevice]: Matching logical devices.
    """
    devices = [_DEFAULT_LOGICAL_CPU]
    if device_type is None:
        return devices
    return [d for d in devices if d.device_type.upper() == device_type.upper()]


def get_visible_devices(device_type: str | None = None) -> list[PhysicalDevice]:
    """Return the list of currently visible `PhysicalDevice`s.

    Args:
        device_type: Optional device type filter.

    Returns:
        list[PhysicalDevice]: Visible physical devices.
    """
    return list_physical_devices(device_type=device_type)


def set_visible_devices(
    devices: PhysicalDevice | Sequence[PhysicalDevice],
    device_type: str | None = None,
) -> None:
    """Set the list of visible `PhysicalDevice`s.

    Args:
        devices: Single `PhysicalDevice` or sequence of `PhysicalDevice`s.
        device_type: Optional device type filter.
    """
    del device_type
    if isinstance(devices, PhysicalDevice):
        _STATE.visible_devices = [devices]
    else:
        _STATE.visible_devices = list(devices)


def get_soft_device_placement() -> bool:
    """Return whether soft device placement is enabled.

    Returns:
        bool: `True` if soft device placement is enabled.
    """
    return _STATE.soft_device_placement


def set_soft_device_placement(enabled: bool) -> None:
    """Enable or disable soft device placement.

    Args:
        enabled: Whether to enable soft device placement.
    """
    _STATE.soft_device_placement = bool(enabled)


def get_logical_device_configuration(
    device: PhysicalDevice | TensorOperand = None,
) -> list[LogicalDeviceConfiguration]:
    """Return the `LogicalDeviceConfiguration` list for `device`.

    Args:
        device: Target physical device.

    Returns:
        list[LogicalDeviceConfiguration]: Configured virtual device list.
    """
    del device
    return list(_STATE.virtual_configs)


def set_logical_device_configuration(
    device: PhysicalDevice | TensorOperand = None,
    logical_devices: Sequence[LogicalDeviceConfiguration] | None = None,
) -> None:
    """Set the `LogicalDeviceConfiguration` list for `device`.

    Args:
        device: Target physical device.
        logical_devices: Sequence of `LogicalDeviceConfiguration` instances.
    """
    del device
    _STATE.virtual_configs = list(logical_devices) if logical_devices is not None else []


def functions_run_eagerly() -> bool:
    """Return whether `@tf.function` callables are forced to execute eagerly.

    Returns:
        bool: `True` if eager execution of functions is enabled.
    """
    return _STATE.run_eagerly


def run_functions_eagerly(run_eagerly: bool = True) -> None:
    """Configure `@tf.function` callables to execute eagerly or as graphs.

    Args:
        run_eagerly: Boolean flag enabling eager function execution.
    """
    _STATE.run_eagerly = bool(run_eagerly)


experimental_functions_run_eagerly = functions_run_eagerly
experimental_run_functions_eagerly = run_functions_eagerly


def experimental_connect_to_cluster(
    cluster_spec_or_resolver: TensorOperand = None,
    job_name: str = "localhost",
    task_index: int = 0,
    protocol: str | None = None,
    make_master_device_default: bool = True,
    cluster_device_filters: TensorOperand = None,
) -> None:
    """Connect to the given cluster specification or resolver.

    Args:
        cluster_spec_or_resolver: Cluster spec or resolver.
        job_name: Name of the local job.
        task_index: Task index within the job.
        protocol: Optional communication protocol.
        make_master_device_default: Whether to make the master device default.
        cluster_device_filters: Optional cluster device filters.
    """
    del (
        cluster_spec_or_resolver,
        job_name,
        task_index,
        protocol,
        make_master_device_default,
        cluster_device_filters,
    )


def experimental_connect_to_host(
    remote_host: str | Sequence[str] | None = None,
    job_name: str = "worker",
) -> None:
    """Connect to one or more remote TensorFlow hosts.

    Args:
        remote_host: Single host address or sequence of host addresses.
        job_name: Remote job name.
    """
    del remote_host, job_name


class experimental:
    """Experimental configuration APIs."""

    class ClusterDeviceFilters:
        """Configuration for filtering visible devices across cluster tasks."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `ClusterDeviceFilters`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs
            self._filters: dict[tuple[str, int], list[str]] = {}

        def set_device_filters(
            self,
            job_name: str,
            task_index: int,
            devices: Sequence[str],
        ) -> None:
            """Set allowed device filters for `(job_name, task_index)`.

            Args:
                job_name: Target job name.
                task_index: Target task index.
                devices: Allowed device strings.
            """
            self._filters[(job_name, task_index)] = list(devices)

    VirtualDeviceConfiguration = LogicalDeviceConfiguration

    @staticmethod
    def disable_mlir_bridge(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Disable the MLIR compiler bridge.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        _STATE.mlir_bridge = False

    @staticmethod
    def enable_mlir_bridge(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Enable the MLIR compiler bridge.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        _STATE.mlir_bridge = True

    @staticmethod
    def enable_op_determinism(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Configure operations to run deterministically.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        _STATE.op_determinism = True

    @staticmethod
    def enable_tensor_float_32_execution(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Enable or disable TensorFloat-32 execution on supported hardware.

        Args:
            *args: Positional boolean flag (defaults to `True`).
            **kwargs: Keyword arguments.
        """
        del kwargs
        _STATE.tf32_enabled = bool(args[0]) if args else True

    @staticmethod
    def get_device_details(*args: _ConfigArg, **kwargs: _ConfigArg) -> dict[str, str]:
        """Return a dictionary of details about a physical device.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict[str, str]: Device metadata dictionary.
        """
        del args, kwargs
        return {"device_name": "CPU:0"}

    @staticmethod
    def get_device_policy(*args: _ConfigArg, **kwargs: _ConfigArg) -> str:
        """Return the current device placement policy.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            str: Active device placement policy string.
        """
        del args, kwargs
        return _STATE.device_policy

    @staticmethod
    def get_memory_growth(*args: _ConfigArg, **kwargs: _ConfigArg) -> bool:
        """Return whether memory growth is enabled for a physical device.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            bool: Memory growth flag.
        """
        del args, kwargs
        return _STATE.memory_growth

    @staticmethod
    def get_memory_info(*args: _ConfigArg, **kwargs: _ConfigArg) -> dict[str, int]:
        """Return memory usage info (`current`, `peak`) for a device.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict[str, int]: Dictionary with `'current'` and `'peak'` byte counts.
        """
        del args, kwargs
        return {"current": 0, "peak": 0}

    @staticmethod
    def get_memory_usage(*args: _ConfigArg, **kwargs: _ConfigArg) -> int:
        """Return current memory usage in bytes for a device.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            int: Current byte count.
        """
        del args, kwargs
        return 0

    @staticmethod
    def get_synchronous_execution(*args: _ConfigArg, **kwargs: _ConfigArg) -> bool:
        """Return whether operations are executed synchronously.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            bool: Synchronous execution flag.
        """
        del args, kwargs
        return _STATE.synchronous_execution

    @staticmethod
    def get_virtual_device_configuration(
        *args: _ConfigArg,
        **kwargs: _ConfigArg,
    ) -> list[LogicalDeviceConfiguration]:
        """Return virtual device configurations for a physical device.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            list[LogicalDeviceConfiguration]: Virtual device configurations.
        """
        del args, kwargs
        return list(_STATE.virtual_configs)

    @staticmethod
    def get_visible_devices(*args: _ConfigArg, **kwargs: _ConfigArg) -> list[PhysicalDevice]:
        """Return visible physical devices.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            list[PhysicalDevice]: Visible physical devices.
        """
        del kwargs
        dev_type = str(args[0]) if args and isinstance(args[0], str) else None
        return get_visible_devices(device_type=dev_type)

    @staticmethod
    def list_logical_devices(*args: _ConfigArg, **kwargs: _ConfigArg) -> list[LogicalDevice]:
        """Return logical devices created by the runtime.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            list[LogicalDevice]: Logical devices.
        """
        del kwargs
        dev_type = str(args[0]) if args and isinstance(args[0], str) else None
        return list_logical_devices(device_type=dev_type)

    @staticmethod
    def list_physical_devices(
        *args: _ConfigArg,
        **kwargs: _ConfigArg,
    ) -> list[PhysicalDevice]:
        """Return physical devices available to the runtime.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            list[PhysicalDevice]: Physical devices.
        """
        del kwargs
        dev_type = str(args[0]) if args and isinstance(args[0], str) else None
        return list_physical_devices(device_type=dev_type)

    @staticmethod
    def reset_memory_stats(*args: _ConfigArg, **kwargs: _ConfigArg) -> None:
        """Reset tracked peak memory statistics for a device.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs

    @staticmethod
    def set_device_policy(*args: _ConfigArg, **kwargs: _ConfigArg) -> None:
        """Set the current device placement policy.

        Args:
            *args: Positional policy string.
            **kwargs: Keyword arguments.
        """
        del kwargs
        _STATE.device_policy = str(args[0]) if args and args[0] is not None else "explicit"

    @staticmethod
    def set_memory_growth(*args: _ConfigArg, **kwargs: _ConfigArg) -> None:
        """Set whether memory growth should be enabled for a `PhysicalDevice`.

        Args:
            *args: Positional `(device, enable)` arguments.
            **kwargs: Keyword arguments.
        """
        del kwargs
        if len(args) >= 2:
            _STATE.memory_growth = bool(args[1])
        else:
            _STATE.memory_growth = bool(args[0]) if args else False

    @staticmethod
    def set_synchronous_execution(*args: _ConfigArg, **kwargs: _ConfigArg) -> None:
        """Specify whether operations are executed synchronously or asynchronously.

        Args:
            *args: Positional boolean flag.
            **kwargs: Keyword arguments.
        """
        del kwargs
        _STATE.synchronous_execution = bool(args[0]) if args else True

    @staticmethod
    def set_virtual_device_configuration(*args: _ConfigArg, **kwargs: _ConfigArg) -> None:
        """Set the virtual device configuration for a `PhysicalDevice`.

        Args:
            *args: Positional `(device, virtual_devices)` arguments.
            **kwargs: Keyword arguments.
        """
        del kwargs
        if len(args) >= 2 and isinstance(args[1], Sequence):
            _STATE.virtual_configs = [
                c for c in args[1] if isinstance(c, LogicalDeviceConfiguration)
            ]
        else:
            _STATE.virtual_configs = []

    @staticmethod
    def set_visible_devices(*args: _ConfigArg, **kwargs: _ConfigArg) -> None:
        """Set the list of visible devices.

        Args:
            *args: Positional `(devices, device_type)` arguments.
            **kwargs: Keyword arguments.
        """
        del kwargs
        if args and isinstance(args[0], PhysicalDevice):
            _STATE.visible_devices = [args[0]]
        elif args and isinstance(args[0], Sequence):
            _STATE.visible_devices = [d for d in args[0] if isinstance(d, PhysicalDevice)]
        else:
            _STATE.visible_devices = []

    @staticmethod
    def tensor_float_32_execution_enabled(*args: _ConfigArg, **kwargs: _ConfigArg) -> bool:
        """Return whether TensorFloat-32 execution is currently enabled.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            bool: TensorFloat-32 execution state.
        """
        del args, kwargs
        return _STATE.tf32_enabled


class optimizer:
    """Graph optimizer configuration namespace."""

    @staticmethod
    def get_experimental_options(*args: _ConfigArg, **kwargs: _ConfigArg) -> dict[str, bool]:
        """Return a dictionary of experimental graph optimizer options.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict[str, bool]: Active optimizer options.
        """
        del args, kwargs
        return dict(_STATE.experimental_options)

    @staticmethod
    def get_jit(*args: _ConfigArg, **kwargs: _ConfigArg) -> bool | str:
        """Return whether XLA JIT compilation is enabled.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            bool | str: Current JIT compilation mode.
        """
        del args, kwargs
        return _STATE.jit_enabled

    @staticmethod
    def set_experimental_options(*args: _ConfigArg, **kwargs: _ConfigArg) -> None:
        """Set experimental graph optimizer options.

        Args:
            *args: Positional options dictionary.
            **kwargs: Keyword arguments.
        """
        from collections.abc import Mapping as _Mapping

        if args and isinstance(args[0], _Mapping):
            _STATE.experimental_options = {str(k): bool(v) for k, v in args[0].items()}
        elif kwargs:
            _STATE.experimental_options = {str(k): bool(v) for k, v in kwargs.items()}

    @staticmethod
    def set_jit(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Configure XLA JIT compilation mode.

        Args:
            *args: Positional boolean or mode string (`'autoclustering'`).
            **kwargs: Keyword arguments.
        """
        del kwargs
        if args and isinstance(args[0], (bool, str)):
            _STATE.jit_enabled = args[0]
        else:
            _STATE.jit_enabled = False


class threading:
    """Thread pool configuration namespace."""

    @staticmethod
    def get_inter_op_parallelism_threads(*args: TensorOperand, **kwargs: TensorOperand) -> int:
        """Return number of threads used for parallelism between independent operations.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            int: Configured inter-op thread count.
        """
        del args, kwargs
        return _STATE.inter_op_threads

    @staticmethod
    def get_intra_op_parallelism_threads(*args: TensorOperand, **kwargs: TensorOperand) -> int:
        """Return number of threads used within an individual operation for parallelism.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            int: Configured intra-op thread count.
        """
        del args, kwargs
        return _STATE.intra_op_threads

    @staticmethod
    def set_inter_op_parallelism_threads(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Set number of threads used for parallelism between independent operations.

        Args:
            *args: Positional thread count integer.
            **kwargs: Keyword arguments.
        """
        del kwargs
        _STATE.inter_op_threads = int(args[0]) if args and isinstance(args[0], int) else 0

    @staticmethod
    def set_intra_op_parallelism_threads(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Set number of threads used within individual operations.

        Args:
            *args: Positional thread count integer.
            **kwargs: Keyword arguments.
        """
        del kwargs
        _STATE.intra_op_threads = int(args[0]) if args and isinstance(args[0], int) else 0
