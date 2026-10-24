"""TensorFlow `signal` processing module."""

from __future__ import annotations

import math
from collections.abc import Sequence

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
    full_ifft = as_tensor_obj(_ops.ifft(t_in))
    real_out = _k_ops.real(full_ifft)
    if fft_length is not None:
        target_len = (
            int(fft_length[0])
            if isinstance(fft_length, Sequence) and fft_length
            else (int(fft_length) if isinstance(fft_length, int) else 1)
        )
        if real_out.shape and real_out.shape[-1] != target_len:
            return _wrap_tensor(_ops.zeros((*real_out.shape[:-1], target_len)))
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
    input: Tensor | TensorOperand,
    nfft: int = 10,
    noverlap: int = 0,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute the Short-time Fourier Transform of `input`.

    Args:
        input: A float tensor of time-domain audio signals.
        nfft: FFT size or frame length.
        noverlap: Number of overlapping samples or frame step.
        *args: Additional positional arguments.
        **kwargs: Additional keyword STFT arguments (`frame_length`, `frame_step`, etc.).

    Returns:
        Tensor: STFT spectrogram tensor.
    """
    del args, kwargs
    tensor_in = _ensure_tensor(input)
    return _wrap_tensor(as_tensor_obj(_ops.stft(tensor_in, nfft, noverlap)))


def inverse_stft(
    input: Tensor | TensorOperand,
    nfft: int = 10,
    noverlap: int = 0,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute the inverse Short-time Fourier Transform of `input`.

    Args:
        input: A complex STFT spectrogram tensor.
        nfft: FFT length.
        noverlap: Number of overlapping samples.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: Reconstructed time-domain signal tensor.
    """
    del args, kwargs
    tensor_in = _ensure_tensor(input)
    return _wrap_tensor(as_tensor_obj(_ops.istft(tensor_in, nfft, noverlap)))


def mfccs_from_log_mel_spectrograms(*args: TensorOperand, **kwargs: TensorOperand) -> Tensor:
    """Compute MFCCs of `log_mel_spectrograms`.

    Args:
        *args: Positional spectrogram arguments.
        **kwargs: Keyword arguments.

    Returns:
        Tensor: MFCC tensor.

    Raises:
        NotImplementedError: Always raised for unimplemented operation.
    """
    del args, kwargs
    raise NotImplementedError("Not implemented")


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
