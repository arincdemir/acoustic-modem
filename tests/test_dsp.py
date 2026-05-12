"""
tests/test_dsp.py — Unit tests for the DSP module.
"""

import numpy as np
import pytest
from acoustic_modem import dsp, config


SR = config.SAMPLE_RATE


class TestGenerateTone:
    def test_length(self):
        tone = dsp.generate_tone(1000, 0.5, SR)
        assert len(tone) == int(SR * 0.5)

    def test_dtype(self):
        tone = dsp.generate_tone(1000, 0.1, SR)
        assert tone.dtype == np.float32

    def test_amplitude(self):
        tone = dsp.generate_tone(1000, 0.1, SR)
        assert np.max(np.abs(tone)) <= 1.0 + 1e-6


class TestGoertzelEnergy:
    def test_target_freq_dominates(self):
        """Energy at the generated frequency should dominate."""
        for freq in [config.FREQ_0, config.FREQ_1]:
            tone = dsp.generate_tone(freq, config.BIT_DURATION, SR)
            e_target = dsp.goertzel_energy(tone, freq, SR)
            other = config.FREQ_1 if freq == config.FREQ_0 else config.FREQ_0
            e_other = dsp.goertzel_energy(tone, other, SR)
            assert e_target > e_other * 10, (
                f"Expected {freq} Hz energy to dominate, got "
                f"e_target={e_target:.3f} e_other={e_other:.3f}"
            )

    def test_silent_signal(self):
        silence = np.zeros(1000, dtype=np.float32)
        assert dsp.goertzel_energy(silence, config.FREQ_0, SR) == pytest.approx(0.0, abs=1e-6)

    def test_empty_signal(self):
        assert dsp.goertzel_energy(np.array([]), config.FREQ_0, SR) == 0.0

    def test_non_negative(self):
        noise = np.random.randn(SR).astype(np.float32) * 0.01
        assert dsp.goertzel_energy(noise, config.FREQ_0, SR) >= 0.0


class TestDetectBit:
    def test_freq0_returns_0(self):
        tone = dsp.generate_tone(config.FREQ_0, config.BIT_DURATION, SR)
        assert dsp.detect_bit(tone, SR) == 0

    def test_freq1_returns_1(self):
        tone = dsp.generate_tone(config.FREQ_1, config.BIT_DURATION, SR)
        assert dsp.detect_bit(tone, SR) == 1

    def test_freq0_with_light_noise(self):
        tone = dsp.generate_tone(config.FREQ_0, config.BIT_DURATION, SR)
        noise = np.random.randn(len(tone)).astype(np.float32) * 0.05
        assert dsp.detect_bit(tone + noise, SR) == 0

    def test_freq1_with_light_noise(self):
        tone = dsp.generate_tone(config.FREQ_1, config.BIT_DURATION, SR)
        noise = np.random.randn(len(tone)).astype(np.float32) * 0.05
        assert dsp.detect_bit(tone + noise, SR) == 1


class TestBitsToWaveform:
    def test_empty_bits(self):
        """Only the preamble should be present."""
        wav = dsp.bits_to_waveform([], SR)
        expected_len = int(SR * config.PREAMBLE_DURATION)
        assert len(wav) == expected_len

    def test_single_bit_length(self):
        wav = dsp.bits_to_waveform([0], SR)
        expected_len = int(SR * (config.PREAMBLE_DURATION + config.BIT_DURATION))
        assert len(wav) == expected_len

    def test_10_bits_length(self):
        bits = [0, 1, 0, 0, 0, 1, 0, 0, 0, 1]  # one UART frame
        wav = dsp.bits_to_waveform(bits, SR)
        expected = int(SR * (config.PREAMBLE_DURATION + 10 * config.BIT_DURATION))
        assert len(wav) == expected


class TestIsAboveNoiseFloor:
    def test_silence_is_below_floor(self):
        silence = np.zeros(1000, dtype=np.float32)
        assert not dsp.is_above_noise_floor(silence)

    def test_loud_tone_is_above_floor(self):
        tone = dsp.generate_tone(config.FREQ_1, 0.1, SR)
        assert dsp.is_above_noise_floor(tone)
