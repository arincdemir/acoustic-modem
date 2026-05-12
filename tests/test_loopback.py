"""
tests/test_loopback.py — End-to-end Tx → Rx round-trip tests (no hardware).

The transmitter generates an audio waveform buffer and the receiver processes
it directly (loopback mode), so no microphone or speaker is required.
"""

import numpy as np
import pytest

from acoustic_modem import config
from acoustic_modem.tx import build_waveform
from acoustic_modem.rx import Receiver


def run_loopback(text: str, noise_amplitude: float = 0.0) -> str:
    """
    Transmit `text` → buffer → receive → return decoded string.

    Parameters
    ----------
    text            : string to transmit
    noise_amplitude : standard deviation of Gaussian noise to add
    """
    waveform = build_waveform(text, config.SAMPLE_RATE)

    if noise_amplitude > 0.0:
        noise = np.random.randn(len(waveform)).astype(config.DTYPE) * noise_amplitude
        waveform = waveform + noise

    received_chars: list[str] = []

    def on_char(ch: str) -> None:
        received_chars.append(ch)

    rx = Receiver(on_char=on_char, sample_rate=config.SAMPLE_RATE)
    rx.start(loopback_buffer=waveform)
    rx.wait_until_done(timeout=60.0)  # generous timeout for slow machines

    return "".join(received_chars)


class TestLoopbackClean:
    def test_single_char(self):
        assert run_loopback("A") == "A"

    def test_hello(self):
        assert run_loopback("Hello") == "Hello"

    def test_numbers(self):
        assert run_loopback("12345") == "12345"

    def test_space(self):
        assert run_loopback("Hi There") == "Hi There"

    def test_empty(self):
        assert run_loopback("") == ""

    def test_single_space(self):
        assert run_loopback(" ") == " "


class TestLoopbackWithNoise:
    """Light noise should not corrupt the signal at 10 baud."""

    def test_hello_with_light_noise(self):
        # Noise at 5% of signal amplitude
        result = run_loopback("Hello", noise_amplitude=0.05)
        assert result == "Hello"

    def test_abc_with_light_noise(self):
        result = run_loopback("ABC", noise_amplitude=0.05)
        assert result == "ABC"
