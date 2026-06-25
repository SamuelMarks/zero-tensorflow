import zero_tensorflow as ztf


def test_audio_decode():
    audio, rate = ztf.audio.decode_wav(b"dummy")
    assert audio.shape == (1, 1)
    assert rate == 44100


def test_audio_encode():
    res = ztf.audio.encode_wav([[0.0]], 44100)
    assert res == b""
