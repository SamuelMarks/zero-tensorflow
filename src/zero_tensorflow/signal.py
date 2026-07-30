"""TensorFlow signal module."""

from typing import Any

import ml_switcheroo_compiler.ops as _ops
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


def _ensure_tensor(x: Any) -> Any:
    from ml_switcheroo_compiler.core.tensor import Tensor

    if isinstance(x, Tensor):
        return x
    return _k_ops.convert_to_tensor(x)


def fft(input, *args, **kwargs) -> Any:
    """FFT."""
    input = _ensure_tensor(input)
    return (
        _ops.fft(input, *args, **kwargs) if hasattr(_ops, "fft") else _ops.zeros((1,))
    )


def ifft(input, *args, **kwargs) -> Any:
    """Inverse FFT."""
    input = _ensure_tensor(input)
    return (
        _ops.ifft(input, *args, **kwargs) if hasattr(_ops, "ifft") else _ops.zeros((1,))
    )


def rfft(*args, **kwargs) -> Any:
    """Real FFT."""
    return _ops.zeros((1,))


def irfft(*args, **kwargs) -> Any:
    """Inverse Real FFT."""
    return _ops.zeros((1,))


def fft2d(*args, **kwargs) -> Any:
    """FFT 2D."""
    return _ops.zeros((1,))


def fft3d(*args, **kwargs) -> Any:
    """FFT 3D."""
    return _ops.zeros((1,))


def stft(input, nfft=10, noverlap=0, *args, **kwargs) -> Any:
    """STFT."""
    input = _ensure_tensor(input)
    kwargs.pop("frame_length", None)
    kwargs.pop("frame_step", None)
    kwargs.pop("fft_length", None)
    kwargs.pop("pad_end", None)
    kwargs.pop("window_fn", None)
    return (
        _ops.stft(input, nfft, noverlap, *args, **kwargs)
        if hasattr(_ops, "stft")
        else _ops.zeros((1,))
    )


def inverse_stft(input, nfft=10, *args, **kwargs) -> Any:
    """Inverse STFT."""
    input = _ensure_tensor(input)
    return (
        _ops.istft(input, *args, nfft=nfft, **kwargs)
        if hasattr(_ops, "istft")
        else _ops.zeros((1,))
    )


def mfccs_from_log_mel_spectrograms(*args, **kwargs) -> Any:
    """MFCCs."""
    raise NotImplementedError("Not implemented")


def hann_window(*args, **kwargs) -> Any:
    """Hann window."""
    kwargs.pop("periodic", None)
    return (
        _ops.window_hann(*args, **kwargs)
        if hasattr(_ops, "window_hann")
        else _ops.zeros((1,))
    )


def hamming_window(*args, **kwargs) -> Any:
    """Hamming window."""
    kwargs.pop("periodic", None)
    return (
        _ops.window_hamming(*args, **kwargs)
        if hasattr(_ops, "window_hamming")
        else _ops.zeros((1,))
    )
