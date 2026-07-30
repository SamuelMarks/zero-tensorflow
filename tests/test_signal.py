import numpy as np
import pytest

from zero_tensorflow import signal


def test_signal_fft():
    assert signal.fft([1.0, 2.0, 3.0, 4.0]) is not None


def test_signal_ifft():
    assert signal.ifft([1.0 + 0j, 2.0 + 0j]) is not None


def test_signal_rfft():
    assert signal.rfft([1.0, 2.0]) is not None


def test_signal_irfft():
    assert signal.irfft([1.0, 2.0]) is not None


def test_signal_fft2d():
    assert signal.fft2d([[1.0, 2.0], [3.0, 4.0]]) is not None


def test_signal_fft3d():
    assert signal.fft3d(np.zeros((2, 2, 2))) is not None


def test_signal_stft():
    assert signal.stft(np.ones((2, 100)), 10, 5) is not None


def test_signal_inverse_stft():
    assert signal.inverse_stft(np.ones((2, 6))) is not None


def test_not_implemented():
    with pytest.raises(NotImplementedError):
        signal.mfccs_from_log_mel_spectrograms()


def test_signal_hann():
    assert signal.hann_window(10) is not None


def test_signal_hamming():
    assert signal.hamming_window(10) is not None


def test_ensure_tensor():
    assert signal._ensure_tensor(1) is not None
