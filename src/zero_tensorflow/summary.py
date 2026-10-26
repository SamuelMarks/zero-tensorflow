"""TensorFlow TensorBoard summary logging module."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable, Iterator
from pathlib import Path

from zero_tensorflow._typing import (
    ArrayLikeProtocol,
    NumpyConvertibleProtocol,
    TensorOperand,
)
from zero_tensorflow.core_tensor import Tensor

from ml_switcheroo_compiler.core.shape import TensorOrScalar

_SUMMARY_STEP: int | TensorOperand | None = None
_RECORD_CONDITION: bool | Callable[[], bool] = True
_DEFAULT_WRITER: SummaryWriter | None = None

__all__ = [
    "SummaryWriter",
    "audio",
    "create_file_writer",
    "create_noop_writer",
    "experimental",
    "flush",
    "histogram",
    "image",
    "record_if",
    "scalar",
    "should_record_summaries",
    "text",
    "write",
]


class SummaryWriter:
    """In-memory and optional file-backed TensorBoard summary writer."""

    def __init__(
        self,
        logdir: str | None = None,
        max_queue: int | None = None,
        flush_millis: int | None = None,
        filename_suffix: str | None = None,
        name: str | None = None,
        is_noop: bool = False,
    ) -> None:
        """Initialize a `SummaryWriter`.

        Args:
            logdir: Optional directory path where summary event logs are written.
            max_queue: Maximum number of events pending before an automatic flush.
            flush_millis: Flush interval in milliseconds.
            filename_suffix: Optional suffix appended to event filenames.
            name: Optional writer name.
            is_noop: If `True`, discard all summary writes.
        """
        self.logdir = logdir
        self.max_queue = max_queue
        self.flush_millis = flush_millis
        self.filename_suffix = filename_suffix or ""
        self.name = name
        self.is_noop = is_noop
        self.is_closed = False
        self.events: list[dict[str, str | int | float]] = []

    def init(self) -> None:
        """Initialize the summary writer resource."""
        if self.logdir and not self.is_noop:
            Path(self.logdir).mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def as_default(self, step: int | None = None) -> Iterator[SummaryWriter]:
        """Context manager setting this writer as the active default writer.

        Args:
            step: Optional default step integer for the duration of the context.

        Yields:
            SummaryWriter: Active summary writer instance.
        """
        global _DEFAULT_WRITER, _SUMMARY_STEP
        prev_writer = _DEFAULT_WRITER
        prev_step = _SUMMARY_STEP
        _DEFAULT_WRITER = self
        if step is not None:
            _SUMMARY_STEP = step
        try:
            yield self
        finally:
            _DEFAULT_WRITER = prev_writer
            if step is not None:
                _SUMMARY_STEP = prev_step

    def set_as_default(self, step: int | None = None) -> None:
        """Permanently set this writer as the default summary writer.

        Args:
            step: Optional default step value.
        """
        global _DEFAULT_WRITER, _SUMMARY_STEP
        _DEFAULT_WRITER = self
        if step is not None:
            _SUMMARY_STEP = step

    def record_event(
        self,
        kind: str,
        tag: str,
        value: str | int | float,
        step: int,
    ) -> bool:
        """Record a summary event if not a no-op writer and not closed.

        Args:
            kind: Summary type (`"scalar"`, `"histogram"`, `"image"`, `"text"`, `"audio"`, `"raw"`).
            tag: Summary tag name.
            value: Serialized or numeric summary value.
            step: Step index.

        Returns:
            bool: `True` if the event was recorded, else `False`.
        """
        if self.is_noop or self.is_closed or not should_record_summaries():
            return False
        self.events.append(
            {
                "kind": kind,
                "tag": tag,
                "value": value,
                "step": step,
            }
        )
        return True

    def flush(self) -> None:
        """Flush buffered summary events to `logdir` if configured."""
        if self.is_noop or self.is_closed or not self.logdir:
            return
        log_path = Path(self.logdir)
        log_path.mkdir(parents=True, exist_ok=True)
        out_file = log_path / f"events.out.tfevents{self.filename_suffix}.json"
        out_file.write_text(json.dumps(self.events), encoding="utf-8")

    def close(self) -> None:
        """Flush and close the summary writer."""
        self.flush()
        self.is_closed = True


_GLOBAL_MEMORY_WRITER = SummaryWriter(name="default_memory_writer")


def _active_writer() -> SummaryWriter:
    """Return the active default `SummaryWriter` or the module-level fallback writer.

    Returns:
        SummaryWriter: Active summary writer.
    """
    return _DEFAULT_WRITER if _DEFAULT_WRITER is not None else _GLOBAL_MEMORY_WRITER


def _resolve_step(step: int | TensorOperand | TensorOrScalar | None) -> int:
    """Resolve `step` or `_SUMMARY_STEP` to an integer.

    Args:
        step: Explicit step argument or `None`.

    Returns:
        int: Resolved integer step (defaults to `0`).
    """
    candidate: int | TensorOperand | TensorOrScalar | None = (
        step if step is not None else _SUMMARY_STEP
    )
    if isinstance(candidate, NumpyConvertibleProtocol):
        candidate = candidate.numpy()
    if isinstance(candidate, ArrayLikeProtocol):
        candidate = candidate.item()
    if isinstance(candidate, (int, float)):
        return int(candidate)
    return 0


def _summarize_value(data: Tensor | TensorOperand | TensorOrScalar) -> str | float:
    """Convert `data` into a scalar `float` or string representation for summary storage.

    Args:
        data: Summary tensor or value.

    Returns:
        str | float: Numeric float or string representation.
    """
    if isinstance(data, NumpyConvertibleProtocol):
        return _summarize_value(data.numpy())
    if isinstance(data, ArrayLikeProtocol):
        return _summarize_value(data.tolist())
    if isinstance(data, (int, float, bool)):
        return float(data)
    return str(data)


def should_record_summaries() -> bool:
    """Return `True` if summaries should currently be recorded.

    Returns:
        bool: Recording state boolean.
    """
    if callable(_RECORD_CONDITION):
        return bool(_RECORD_CONDITION())
    return bool(_RECORD_CONDITION)


@contextlib.contextmanager
def record_if(condition: bool | Callable[[], bool]) -> Iterator[None]:
    """Context manager controlling whether summaries are recorded.

    Args:
        condition: Boolean or zero-argument callable returning a boolean.

    Yields:
        None: Context scope.
    """
    global _RECORD_CONDITION
    prev = _RECORD_CONDITION
    _RECORD_CONDITION = condition
    try:
        yield
    finally:
        _RECORD_CONDITION = prev


def create_file_writer(
    logdir: str,
    max_queue: int | None = None,
    flush_millis: int | None = None,
    filename_suffix: str | None = None,
    name: str | None = None,
    experimental_trackable: bool = False,
) -> SummaryWriter:
    """Create a summary file writer for the given `logdir`.

    Args:
        logdir: Directory path where event files will be written.
        max_queue: Optional maximum queue size before flush.
        flush_millis: Optional flush period in milliseconds.
        filename_suffix: Optional event file suffix.
        name: Optional writer name.
        experimental_trackable: Unused trackable flag.

    Returns:
        SummaryWriter: Newly created file-backed `SummaryWriter`.
    """
    del experimental_trackable
    writer = SummaryWriter(
        logdir=logdir,
        max_queue=max_queue,
        flush_millis=flush_millis,
        filename_suffix=filename_suffix,
        name=name,
        is_noop=False,
    )
    writer.init()
    return writer


def create_noop_writer() -> SummaryWriter:
    """Create a summary writer that discards all summary events.

    Returns:
        SummaryWriter: No-op `SummaryWriter`.
    """
    return SummaryWriter(is_noop=True)


def flush(writer: SummaryWriter | None = None, name: str | None = None) -> None:
    """Flush the specified `writer` or the active default summary writer.

    Args:
        writer: Optional `SummaryWriter` to flush.
        name: Optional operation name.
    """
    del name
    target = writer if writer is not None else _active_writer()
    target.flush()


def write(
    tag: str,
    tensor: Tensor | TensorOperand,
    step: int | TensorOperand | None = None,
    metadata: bytes | str | None = None,
    name: str | None = None,
) -> bool:
    """Write a generic summary event to the active summary writer.

    Args:
        tag: Summary tag string.
        tensor: Summary payload tensor or scalar.
        step: Step index (`int64`-convertible).
        metadata: Optional serialized `SummaryMetadata`.
        name: Optional operation name.

    Returns:
        bool: `True` if written, else `False`.
    """
    del metadata, name
    return _active_writer().record_event(
        "write",
        tag,
        _summarize_value(tensor),
        _resolve_step(step),
    )


def scalar(
    name: str,
    data: Tensor | TensorOperand,
    step: int | TensorOperand | None = None,
    description: str | None = None,
) -> bool:
    """Write a scalar summary value.

    Args:
        name: Summary tag name.
        data: Real numeric scalar value or tensor.
        step: Step index.
        description: Optional markdown description.

    Returns:
        bool: `True` if recorded, else `False`.
    """
    del description
    return _active_writer().record_event(
        "scalar",
        name,
        _summarize_value(data),
        _resolve_step(step),
    )


def histogram(
    name: str,
    data: Tensor | TensorOperand,
    step: int | TensorOperand | None = None,
    buckets: int | None = None,
    description: str | None = None,
) -> bool:
    """Write a histogram summary.

    Args:
        name: Summary tag name.
        data: Tensor of any shape containing histogram values.
        step: Step index.
        buckets: Optional number of histogram buckets.
        description: Optional markdown description.

    Returns:
        bool: `True` if recorded, else `False`.
    """
    del buckets, description
    return _active_writer().record_event(
        "histogram",
        name,
        _summarize_value(data),
        _resolve_step(step),
    )


def image(
    name: str = "image",
    data: Tensor | TensorOperand = None,
    step: int | TensorOperand | None = None,
    max_outputs: int = 3,
    description: str | None = None,
) -> bool:
    """Write an image summary.

    Args:
        name: Summary tag name.
        data: 4-D image tensor `[batch, height, width, channels]`.
        step: Step index.
        max_outputs: Maximum number of batch elements to emit.
        description: Optional markdown description.

    Returns:
        bool: `True` if recorded, else `False`.
    """
    del max_outputs, description
    return _active_writer().record_event(
        "image",
        name,
        _summarize_value(data),
        _resolve_step(step),
    )


def text(
    name: str,
    data: Tensor | TensorOperand,
    step: int | TensorOperand | None = None,
    description: str | None = None,
) -> bool:
    """Write a text summary.

    Args:
        name: Summary tag name.
        data: String scalar or tensor.
        step: Step index.
        description: Optional markdown description.

    Returns:
        bool: `True` if recorded, else `False`.
    """
    del description
    return _active_writer().record_event(
        "text",
        name,
        str(_summarize_value(data)),
        _resolve_step(step),
    )


def audio(
    name: str,
    data: Tensor | TensorOperand,
    sample_rate: int | float,
    step: int | TensorOperand | None = None,
    max_outputs: int = 3,
    encoding: str | None = None,
    description: str | None = None,
) -> bool:
    """Write an audio summary.

    Args:
        name: Summary tag name.
        data: Floating-point audio tensor `[batch, frames, channels]`.
        sample_rate: Audio sample rate in Hz.
        step: Step index.
        max_outputs: Maximum audio clips to record.
        encoding: Optional audio encoding format (`'wav'`).
        description: Optional markdown description.

    Returns:
        bool: `True` if recorded, else `False`.
    """
    del sample_rate, max_outputs, encoding, description
    return _active_writer().record_event(
        "audio",
        name,
        _summarize_value(data),
        _resolve_step(step),
    )


class experimental:
    """Experimental summary step and scope utilities."""

    @staticmethod
    def get_step() -> int | TensorOperand | None:
        """Return the current default summary step for the thread.

        Returns:
            int | TensorOperand | None: Current summary step value or None.
        """
        return _SUMMARY_STEP

    @staticmethod
    def set_step(step: int | TensorOperand | None = 0) -> None:
        """Set the default summary step for the current thread.

        Args:
            step: An `int64`-convertible scalar step value.
        """
        global _SUMMARY_STEP
        _SUMMARY_STEP = step

    @staticmethod
    @contextlib.contextmanager
    def summary_scope(*args: TensorOperand, **kwargs: TensorOperand) -> Iterator[tuple[str, None]]:
        """Context manager for grouping summary operations under a name scope.

        Args:
            *args: Positional scope arguments.
            **kwargs: Keyword scope arguments.

        Yields:
            tuple[str, None]: Tuple of `(scope_name, tag_metadata)`.
        """
        del args, kwargs
        yield ("summary_scope", None)

    @staticmethod
    def write_raw_pb(
        tensor: bytes | str | TensorOperand = b"",
        step: int | TensorOperand | None = None,
        name: str | None = None,
    ) -> bool:
        """Write raw serialized `Summary` protocol buffers.

        Args:
            tensor: Serialized `Summary` protobuf bytes.
            step: Optional step index.
            name: Optional operation name.

        Returns:
            bool: `True` if recorded, else `False`.
        """
        tag = name or "raw_pb"
        return _active_writer().record_event(
            "raw",
            tag,
            str(tensor),
            _resolve_step(step),
        )
