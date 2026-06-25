import numpy as np
from zero_tensorflow import signal


def test_signal_fft():
    res = signal.fft([1.0, 2.0, 3.0, 4.0])
    assert res.shape == (4,)


def test_signal_ifft():
    res = signal.ifft([1.0 + 0j, 2.0 + 0j])
    assert res.shape == (2,)


def test_signal_hann():
    res1 = signal.hann_window(10)
    assert res1.shape == (10,)
    res2 = signal.hann_window(10, periodic=False)
    assert res2.shape == (10,)


def test_signal_hamming():
    res1 = signal.hamming_window(10)
    assert res1.shape == (10,)
    res2 = signal.hamming_window(10, periodic=False)
    assert res2.shape == (10,)


def test_signal_stft():
    sig = np.ones((2, 100))
    res = signal.stft(sig, 10, 5)
    assert len(res.shape) == 3
    assert res.shape[-1] == 6  # 10 // 2 + 1

    res_pad = signal.stft(sig, 10, 5, pad_end=True)
    assert len(res_pad.shape) == 3
