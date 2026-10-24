"""Unit tests for `test_signal`."""

from __future__ import annotations

from zero_tensorflow import signal

import numpy as np
import pytest


def test_signal_fft() -> None:
    """Execute `test_signal_fft`."""
    assert signal.fft([1.0, 2.0, 3.0, 4.0]) is not None


def test_signal_ifft() -> None:
    """Execute `test_signal_ifft`."""
    assert signal.ifft([1.0 + 0j, 2.0 + 0j]) is not None


def test_signal_rfft() -> None:
    """Execute `test_signal_rfft`."""
    assert signal.rfft([1.0, 2.0]) is not None
    assert signal.rfft([1.0, 2.0, 3.0, 4.0], fft_length=4) is not None
    assert signal.rfft([[1.0, 2.0], [3.0, 4.0]]) is not None
    assert signal.rfft(None) is not None


def test_signal_irfft() -> None:
    """Execute `test_signal_irfft`."""
    assert signal.irfft([1.0, 2.0]) is not None
    assert signal.irfft([1.0, 2.0], fft_length=6) is not None
    assert signal.irfft([1.0, 2.0], fft_length=[2]) is not None
    assert signal.irfft(None) is not None


def test_signal_fft2d() -> None:
    """Execute `test_signal_fft2d`."""
    assert signal.fft2d([[1.0, 2.0], [3.0, 4.0]]) is not None
    assert signal.fft2d([1.0, 2.0]) is not None
    assert signal.fft2d(None) is not None


def test_signal_fft3d() -> None:
    """Execute `test_signal_fft3d`."""
    assert signal.fft3d(np.zeros((2, 2, 2))) is not None
    assert signal.fft3d([[1.0, 2.0], [3.0, 4.0]]) is not None
    assert signal.fft3d(None) is not None


def test_signal_stft() -> None:
    """Execute `test_signal_stft`."""
    assert signal.stft(np.ones((2, 100)), 10, 5) is not None


def test_signal_inverse_stft() -> None:
    """Execute `test_signal_inverse_stft`."""
    assert signal.inverse_stft(np.ones((2, 6))) is not None


def test_not_implemented() -> None:
    """Execute `test_not_implemented`."""
    with pytest.raises(NotImplementedError):
        signal.mfccs_from_log_mel_spectrograms()


def test_signal_hann() -> None:
    """Execute `test_signal_hann`."""
    assert signal.hann_window(10) is not None


def test_signal_hamming() -> None:
    """Execute `test_signal_hamming`."""
    assert signal.hamming_window(10) is not None


def test_ensure_tensor() -> None:
    """Execute `test_ensure_tensor`."""
    raw = signal._ensure_tensor(1)
    assert raw is not None
    assert signal._ensure_tensor(raw) is raw
