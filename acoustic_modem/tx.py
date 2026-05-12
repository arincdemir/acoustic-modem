"""
tx.py — Transmitter: text → framed bits → audio waveform → playback.
"""

import numpy as np
import sounddevice as sd

from acoustic_modem import config, framing, dsp


def build_waveform(text: str,
                   sample_rate: int = config.SAMPLE_RATE) -> np.ndarray:
    """
    Convert a text string into a complete audio waveform ready for playback.

    Structure:
        [Preamble: 200ms @ FREQ_1]
        [char0: start + 8 data + stop]
        [char1: start + 8 data + stop]
        ...
    """
    bits = framing.frame_message(text)
    return dsp.bits_to_waveform(bits, sample_rate)


def transmit_message(text: str,
                     sample_rate: int = config.SAMPLE_RATE,
                     device: int | None = None) -> None:
    """
    Transmit `text` over the speakers.

    Blocks until playback is complete.

    Args:
        text:        The string to transmit.
        sample_rate: Audio sample rate (default from config).
        device:      sounddevice output device index.  None = system default.
    """
    if not text:
        return

    waveform = build_waveform(text, sample_rate)
    sd.play(waveform, samplerate=sample_rate, device=device,
            blocking=True)
