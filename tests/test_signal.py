"""Unit tests for `zero_tensorflow.signal`."""

from __future__ import annotations

import zero_tensorflow as tf
from zero_tensorflow import Tensor, debugging, signal

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
    assert signal.rfft([1.0], fft_length=4) is not None
    assert signal.rfft([[1.0, 2.0], [3.0, 4.0]]) is not None
    assert signal.rfft(None) is not None


def test_signal_irfft() -> None:
    """Verify `signal.irfft` truncates or zero-pads to `fft_length` before `ifft`."""
    assert signal.irfft([1.0, 2.0]).shape == (2,)
    padded = signal.irfft([1.0, 2.0], fft_length=6)
    assert padded.shape == (6,)
    assert any(abs(v) > 1e-6 for v in debugging._flatten_floats(padded))
    truncated = signal.irfft([1.0, 2.0, 3.0, 4.0], fft_length=[2])
    assert truncated.shape == (2,)
    exact = signal.irfft([1.0, 2.0], fft_length=2)
    assert exact.shape == (2,)
    assert signal.irfft(None) is not None


def test_signal_fft2d() -> None:
    """Execute `test_signal_fft2d`."""
    assert signal.fft2d([[1.0, 2.0], [3.0, 4.0]]) is not None
    assert signal.fft2d([1.0, 2.0]) is not None
    assert signal.fft2d(None) is not None


def test_signal_fft3d() -> None:
    """Execute `test_signal_fft3d`, `ifft2d`, `rfft2d`, `irfft2d`, `ifft3d`, `rfft3d`, `irfft3d`, `dct`, `idct`, `frame`, `mdct`, and `inverse_mdct`."""
    assert signal.fft3d(tf.zeros((2, 2, 2))) is not None
    assert signal.fft3d([[1.0, 2.0], [3.0, 4.0]]) is not None
    assert signal.fft3d(None) is not None

    assert signal.ifft2d([[1.0, 2.0], [3.0, 4.0]]).shape == (2, 2)
    assert signal.ifft2d([1.0, 2.0]).shape == (2,)
    assert signal.ifft2d(None).shape == (1,)

    assert signal.rfft2d([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], fft_length=[2, 4]).shape == (
        2,
        3,
    )
    assert signal.rfft2d([1.0], fft_length=[4, 4]).shape == (1,)
    assert signal.rfft2d(None).shape == (1,)
    assert signal.irfft2d([[1.0, 2.0], [3.0, 4.0]]).shape == (2, 2)

    assert signal.ifft3d(tf.zeros((2, 2, 2))).shape == (2, 2, 2)
    assert signal.ifft3d([[1.0, 2.0], [3.0, 4.0]]).shape == (2, 2)
    assert signal.ifft3d(None).shape == (1,)

    assert signal.rfft3d(tf.ones((2, 2, 4)), fft_length=[2, 2, 4]).shape == (2, 2, 3)
    assert signal.rfft3d([1.0], fft_length=[4, 4, 4]).shape == (1,)
    assert signal.rfft3d(None).shape == (1,)
    assert signal.irfft3d(tf.ones((2, 2, 2))).shape == (2, 2, 2)

    d_val = signal.dct(Tensor([[1.0, 2.0, 3.0, 4.0]]))
    id_val = signal.idct(d_val)
    debugging.assert_near(id_val, [[1.0, 2.0, 3.0, 4.0]], atol=1e-4)
    with pytest.raises(ValueError, match="input must not be None"):
        signal.idct(None)

    fr = signal.frame(Tensor([1.0, 2.0, 3.0, 4.0]), frame_length=2, frame_step=1, pad_end=False)
    assert fr.shape == (3, 2)
    fr_pad = signal.frame(
        Tensor([1.0, 2.0, 3.0]), frame_length=2, frame_step=2, pad_end=True, pad_value=0.0
    )
    assert fr_pad.shape == (2, 2)
    fr_exact = signal.frame(
        Tensor([1.0, 2.0, 3.0, 4.0]), frame_length=2, frame_step=2, pad_end=True
    )
    assert fr_exact.shape == (2, 2)
    fr_empty = signal.frame(Tensor([1.0]), frame_length=4, frame_step=2, pad_end=False)
    assert fr_empty.shape == (0, 4)
    fr_ax0 = signal.frame(
        Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        frame_length=2,
        frame_step=1,
        axis=0,
    )
    assert fr_ax0.shape == (2, 2, 2)
    assert signal.frame(None).shape == (0,)
    assert signal.frame(Tensor(1.0)).shape == ()

    m_out = signal.mdct(Tensor([1.0, 2.0, 3.0, 4.0]), frame_length=4, window_fn=signal.hann_window)
    assert m_out.shape == (1, 2)
    m_nowin = signal.mdct(Tensor([1.0, 2.0, 3.0, 4.0]), frame_length=4)
    assert m_nowin.shape == (1, 2)
    im_out = signal.inverse_mdct(m_out)
    assert im_out.shape == (2,)
    assert signal.inverse_mdct(Tensor([1.0])).shape == (1,)


def test_signal_stft() -> None:
    """Verify `signal.stft` honors `frame_length`, `frame_step`, `fft_length`, and `window_fn`."""
    assert signal.stft(tf.ones((2, 100)), 10, 5) is not None
    called_lengths: list[int] = []

    def _custom_win(n: int) -> Tensor:
        """Record requested window length and return a Hann window.

        Args:
            n: Window length.

        Returns:
            Tensor: Window tensor.
        """
        called_lengths.append(n)
        return signal.hann_window(n)

    res = signal.stft(
        tf.ones((2, 32)),
        frame_length=8,
        frame_step=4,
        fft_length=8,
        window_fn=_custom_win,
    )
    assert res is not None
    assert called_lengths == [8]
    assert signal.stft(tf.ones((2, 32)), nfft=8, noverlap=2) is not None


def test_signal_inverse_stft() -> None:
    """Verify `signal.inverse_stft` honors `frame_length`, `frame_step`, `fft_length`, and `window_fn`."""
    assert signal.inverse_stft(tf.ones((2, 6))) is not None
    assert (
        signal.inverse_stft(
            tf.ones((2, 6)),
            frame_length=8,
            frame_step=4,
            fft_length=8,
            window_fn=signal.hann_window,
        )
        is not None
    )
    assert signal.inverse_stft(tf.ones((2, 6)), nfft=8, noverlap=2) is not None


def test_mfccs_from_log_mel_spectrograms() -> None:
    """Verify `signal.mfccs_from_log_mel_spectrograms` computes ortho-scaled DCT-II."""
    with pytest.raises(ValueError):
        signal.mfccs_from_log_mel_spectrograms(None)
    with pytest.raises(ValueError):
        signal.mfccs_from_log_mel_spectrograms(1.0)

    mel = tf.ones((2, 4), dtype=tf.float32)
    mfcc = signal.mfccs_from_log_mel_spectrograms(mel)
    assert mfcc.shape == (2, 4)
    debugging.assert_near(
        mfcc[:, 1:],
        tf.zeros((2, 3), dtype=tf.float32),
        atol=1e-5,
    )
    assert debugging._flatten_floats(mfcc[0][0])[0] > 0.0


def test_signal_hann() -> None:
    """Execute `test_signal_hann`."""
    assert signal.hann_window(10) is not None
    assert signal.hann_window(1).shape == (1,)
    assert signal.hann_window(0).shape == (0,)


def test_signal_hamming() -> None:
    """Execute `test_signal_hamming`."""
    assert signal.hamming_window(10) is not None
    assert signal.hamming_window(1).shape == (1,)
    assert signal.hamming_window(0).shape == (0,)


def test_ensure_tensor() -> None:
    """Execute `test_ensure_tensor`."""
    raw = signal._ensure_tensor(1)
    assert raw is not None
    assert signal._ensure_tensor(raw) is raw
