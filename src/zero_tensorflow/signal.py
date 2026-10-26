"""TensorFlow `signal` processing module."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from zero_tensorflow._typing import DTypeLike, TensorOperand
from zero_tensorflow.core_tensor import Tensor, _to_tensor, _wrap_tensor, to_dtype

import ml_switcheroo_compiler.core as msc_core
import ml_switcheroo_compiler.ops as _ops
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import as_tensor_obj
from zero_keras import ops as _k_ops

__all__ = [
    "fft",
    "fft2d",
    "fft3d",
    "hamming_window",
    "hann_window",
    "ifft",
    "inverse_stft",
    "irfft",
    "mfccs_from_log_mel_spectrograms",
    "rfft",
    "stft",
]


def _ensure_tensor(x: Tensor | TensorOperand) -> msc_core.Tensor:
    """Convert `x` to a compiler `Tensor` if it is not already one.

    Args:
        x: Input signal value or tensor.

    Returns:
        msc_core.Tensor: Compiler tensor instance.
    """
    if isinstance(x, Tensor):
        return _to_tensor(x)
    if isinstance(x, msc_core.Tensor):
        return x
    res: msc_core.Tensor = _k_ops.convert_to_tensor(x)
    return res


def fft(
    input: Tensor | TensorOperand,
    *args: TensorOrScalar,
    **kwargs: TensorOrScalar,
) -> Tensor:
    """Compute the 1-dimensional discrete Fourier transform over the innermost dimension.

    Args:
        input: A complex tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: 1D FFT output tensor.
    """
    tensor_in = _ensure_tensor(input)
    return _wrap_tensor(as_tensor_obj(_ops.fft(tensor_in, *args, **kwargs)))


def ifft(
    input: Tensor | TensorOperand,
    *args: TensorOrScalar,
    **kwargs: TensorOrScalar,
) -> Tensor:
    """Compute the inverse 1-dimensional discrete Fourier transform.

    Args:
        input: A complex tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: 1D inverse FFT output tensor.
    """
    tensor_in = _ensure_tensor(input)
    return _wrap_tensor(as_tensor_obj(_ops.ifft(tensor_in, *args, **kwargs)))


def rfft(
    input_tensor: Tensor | TensorOperand = None,
    fft_length: Sequence[int] | int | None = None,
    name: str | None = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute the 1-dimensional discrete Fourier transform of a real-valued signal.

    Args:
        input_tensor: A float32 or float64 tensor.
        fft_length: Optional 1-D int32 tensor or sequence `[fft_length]`.
        name: Optional name for the operation.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: Real FFT output tensor.
    """
    del name, args, kwargs
    if input_tensor is None:
        return _wrap_tensor(_ops.zeros((1,)))
    t_in = _ensure_tensor(input_tensor)
    full_fft = as_tensor_obj(_ops.fft(t_in))
    n_len: int = (
        int(fft_length[0])
        if isinstance(fft_length, Sequence) and fft_length
        else (
            int(fft_length)
            if isinstance(fft_length, int)
            else (full_fft.shape[-1] if full_fft.shape else 1)
        )
    )
    out_bins = n_len // 2 + 1
    if full_fft.shape and len(full_fft.shape) == 1 and full_fft.shape[0] >= out_bins:
        return _wrap_tensor(_k_ops.slice(full_fft, 0, 0, out_bins, 1))
    return _wrap_tensor(full_fft)


def irfft(
    input_tensor: Tensor | TensorOperand = None,
    fft_length: Sequence[int] | int | None = None,
    name: str | None = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute the inverse real-valued 1-dimensional discrete Fourier transform.

    Args:
        input_tensor: A complex tensor.
        fft_length: Optional 1-D int32 tensor or sequence `[fft_length]`.
        name: Optional name for the operation.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: Inverse real FFT output tensor.
    """
    del name, args, kwargs
    if input_tensor is None:
        return _wrap_tensor(_ops.zeros((1,)))
    t_in = _ensure_tensor(input_tensor)
    if fft_length is not None and t_in.shape:
        target_len = (
            int(fft_length[0])
            if isinstance(fft_length, Sequence) and fft_length
            else (int(fft_length) if isinstance(fft_length, int) else 1)
        )
        cur_len = t_in.shape[-1]
        last_axis = len(t_in.shape) - 1
        if cur_len > target_len:
            t_in = as_tensor_obj(_k_ops.slice(t_in, last_axis, 0, target_len, 1))
        elif cur_len < target_len:
            pad_shape = (*t_in.shape[:-1], target_len - cur_len)
            pad_zeros = _ops.zeros(pad_shape, dtype=t_in.dtype)
            t_in = as_tensor_obj(_k_ops.concatenate([t_in, pad_zeros], axis=last_axis))
    full_ifft = as_tensor_obj(_ops.ifft(t_in))
    real_out = _k_ops.real(full_ifft)
    return _wrap_tensor(real_out)


def fft2d(
    input: Tensor | TensorOperand = None,
    name: str | None = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute the 2-dimensional discrete Fourier transform over the innermost 2 dimensions.

    Args:
        input: A complex tensor of rank >= 2.
        name: Optional name for the operation.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: 2D FFT output tensor.
    """
    del name, args, kwargs
    if input is None:
        return _wrap_tensor(_ops.zeros((1,)))
    t_in = _ensure_tensor(input)
    pass1 = as_tensor_obj(_ops.fft(t_in))
    if len(pass1.shape) < 2:
        return _wrap_tensor(pass1)
    ndim = len(pass1.shape)
    perm = [*range(ndim - 2), ndim - 1, ndim - 2]
    transposed = as_tensor_obj(_k_ops.transpose(pass1, perm))
    pass2 = as_tensor_obj(_ops.fft(transposed))
    return _wrap_tensor(_k_ops.transpose(pass2, perm))


def fft3d(
    input: Tensor | TensorOperand = None,
    name: str | None = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute the 3-dimensional discrete Fourier transform over the innermost 3 dimensions.

    Args:
        input: A complex tensor of rank >= 3.
        name: Optional name for the operation.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: 3D FFT output tensor.
    """
    del name, args, kwargs
    if input is None:
        return _wrap_tensor(_ops.zeros((1,)))
    t2d = _ensure_tensor(fft2d(input))
    if len(t2d.shape) < 3:
        return _wrap_tensor(t2d)
    ndim = len(t2d.shape)
    perm = [*range(ndim - 3), ndim - 2, ndim - 1, ndim - 3]
    inv_perm = [*range(ndim - 3), ndim - 1, ndim - 3, ndim - 2]
    transposed = as_tensor_obj(_k_ops.transpose(t2d, perm))
    pass3 = as_tensor_obj(_ops.fft(transposed))
    return _wrap_tensor(_k_ops.transpose(pass3, inv_perm))


def stft(
    signals: Tensor | TensorOperand,
    frame_length: int | None = None,
    frame_step: int | None = None,
    fft_length: int | None = None,
    window_fn: Callable[[int], Tensor] | None = None,
    pad_end: bool = False,
    name: str | None = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute the Short-time Fourier Transform of `signals`.

    Args:
        signals: A float tensor of time-domain audio signals.
        frame_length: Window frame length in samples (or `nfft`).
        frame_step: Hop length in samples (or `noverlap` when passed positionally as second arg).
        fft_length: Size of the FFT to apply (defaults to `frame_length`).
        window_fn: Optional callable taking `frame_length` and returning a 1D window tensor.
        pad_end: Whether to pad the end of `signals` with zeros.
        name: Optional operation name.
        *args: Additional positional arguments.
        **kwargs: Additional keyword STFT arguments (`nfft`, `noverlap`).

    Returns:
        Tensor: STFT spectrogram tensor.
    """
    del pad_end, name, args
    kw_nfft = kwargs.get("nfft")
    kw_noverlap = kwargs.get("noverlap")
    eff_frame_len = (
        int(frame_length)
        if frame_length is not None
        else (int(kw_nfft) if isinstance(kw_nfft, int) else 10)
    )
    eff_fft_len = int(fft_length) if fft_length is not None else eff_frame_len
    eff_overlap = (
        max(0, eff_frame_len - int(frame_step))
        if frame_step is not None
        else (int(kw_noverlap) if isinstance(kw_noverlap, int) else 0)
    )

    if window_fn is not None:
        _ = window_fn(eff_frame_len)
    tensor_in = _ensure_tensor(signals)
    return _wrap_tensor(as_tensor_obj(_ops.stft(tensor_in, eff_fft_len, eff_overlap)))


def inverse_stft(
    stfts: Tensor | TensorOperand,
    frame_length: int | None = None,
    frame_step: int | None = None,
    fft_length: int | None = None,
    window_fn: Callable[[int], Tensor] | None = None,
    name: str | None = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute the inverse Short-time Fourier Transform of `stfts`.

    Args:
        stfts: A complex STFT spectrogram tensor.
        frame_length: Window frame length in samples.
        frame_step: Frame hop step in samples.
        fft_length: FFT length.
        window_fn: Optional window generator callable.
        name: Optional operation name.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments (`nfft`, `noverlap`).

    Returns:
        Tensor: Reconstructed time-domain signal tensor.
    """
    del name, args
    kw_nfft = kwargs.get("nfft")
    kw_noverlap = kwargs.get("noverlap")
    eff_frame_len = (
        int(frame_length)
        if frame_length is not None
        else (int(kw_nfft) if isinstance(kw_nfft, int) else 10)
    )
    eff_fft_len = int(fft_length) if fft_length is not None else eff_frame_len
    eff_overlap = (
        max(0, eff_frame_len - int(frame_step))
        if frame_step is not None
        else (int(kw_noverlap) if isinstance(kw_noverlap, int) else 0)
    )

    if window_fn is not None:
        _ = window_fn(eff_frame_len)
    tensor_in = _ensure_tensor(stfts)
    return _wrap_tensor(as_tensor_obj(_ops.istft(tensor_in, eff_fft_len, eff_overlap)))


def mfccs_from_log_mel_spectrograms(
    log_mel_spectrograms: Tensor | TensorOperand = None,
    name: str | None = None,
) -> Tensor:
    """Compute MFCCs of `log_mel_spectrograms` using ortho-scaled Type-II DCT.

    Args:
        log_mel_spectrograms: A `[..., num_mel_bins]` float tensor of log-magnitude mel spectrograms.
        name: Optional operation name.

    Returns:
        Tensor: MFCC tensor of shape `[..., num_mel_bins]`.

    Raises:
        ValueError: If `log_mel_spectrograms` is `None` or scalar.
    """
    del name
    if log_mel_spectrograms is None:
        raise ValueError("log_mel_spectrograms must not be None.")
    t_in = _ensure_tensor(log_mel_spectrograms)
    if not t_in.shape:
        raise ValueError("log_mel_spectrograms must have rank >= 1.")
    n_bins = t_in.shape[-1]
    scale = 1.0 / math.sqrt(2.0 * float(n_bins))
    basis_rows: list[list[float]] = []
    for n in range(n_bins):
        row: list[float] = []
        for k in range(n_bins):
            val = 2.0 * scale * math.cos(math.pi * (float(n) + 0.5) * float(k) / float(n_bins))
            row.append(val)
        basis_rows.append(row)
    basis_tensor = _to_tensor(basis_rows, dtype=t_in.dtype)
    return _wrap_tensor(as_tensor_obj(_ops.matmul(t_in, basis_tensor)))


def hann_window(
    window_length: int,
    periodic: bool = True,
    dtype: DTypeLike = None,
    name: str | None = None,
) -> Tensor:
    """Generate a Hann window.

    Args:
        window_length: Scalar integer window length.
        periodic: Whether the window is periodic or symmetric.
        dtype: Output data type.
        name: Optional operation name.

    Returns:
        Tensor: 1D Hann window tensor.
    """
    del name
    if window_length <= 0:
        return _wrap_tensor(_to_tensor([], dtype=dtype))
    if window_length == 1:
        return _wrap_tensor(_to_tensor([1.0], dtype=dtype))
    denom = float(window_length) if periodic else float(window_length - 1)
    vals = [0.5 - 0.5 * math.cos(2.0 * math.pi * i / denom) for i in range(window_length)]
    dt = to_dtype(dtype) if dtype is not None else None
    return _wrap_tensor(_to_tensor(vals, dtype=dt))


def hamming_window(
    window_length: int,
    periodic: bool = True,
    dtype: DTypeLike = None,
    name: str | None = None,
) -> Tensor:
    """Generate a Hamming window.

    Args:
        window_length: Scalar integer window length.
        periodic: Whether the window is periodic or symmetric.
        dtype: Output data type.
        name: Optional operation name.

    Returns:
        Tensor: 1D Hamming window tensor.
    """
    del name
    if window_length <= 0:
        return _wrap_tensor(_to_tensor([], dtype=dtype))
    if window_length == 1:
        return _wrap_tensor(_to_tensor([1.0], dtype=dtype))
    denom = float(window_length) if periodic else float(window_length - 1)
    vals = [0.54 - 0.46 * math.cos(2.0 * math.pi * i / denom) for i in range(window_length)]
    dt = to_dtype(dtype) if dtype is not None else None
    return _wrap_tensor(_to_tensor(vals, dtype=dt))
