"""
tests/test_dsp.py — Unit tests for the DSP module (six-tone FSK).
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
        """Energy at the generated frequency should dominate all other FSK frequencies."""
        for i, freq in enumerate(config.FREQUENCIES):
            tone = dsp.generate_tone(freq, config.SYMBOL_DURATION, SR)
            e_target = dsp.goertzel_energy(tone, freq, SR)
            for j, other_freq in enumerate(config.FREQUENCIES):
                if j == i:
                    continue
                e_other = dsp.goertzel_energy(tone, other_freq, SR)
                assert e_target > e_other * 10, (
                    f"Expected {freq} Hz energy to dominate {other_freq} Hz, "
                    f"got e_target={e_target:.3f} e_other={e_other:.3f}"
                )

    def test_silent_signal(self):
        silence = np.zeros(1000, dtype=np.float32)
        assert dsp.goertzel_energy(silence, config.FREQ_00, SR) == pytest.approx(0.0, abs=1e-6)

    def test_empty_signal(self):
        assert dsp.goertzel_energy(np.array([]), config.FREQ_00, SR) == 0.0

    def test_non_negative(self):
        noise = np.random.randn(SR).astype(np.float32) * 0.01
        assert dsp.goertzel_energy(noise, config.FREQ_00, SR) >= 0.0


class TestDetectSymbol:
    def test_each_symbol_clean(self):
        """Each of the 6 pure tones should map to its correct index."""
        for sym_val, freq in enumerate(config.FREQUENCIES):
            tone = dsp.generate_tone(freq, config.SYMBOL_DURATION, SR)
            assert dsp.detect_symbol(tone, SR) == sym_val, (
                f"Tone at {freq} Hz should be detected as symbol {sym_val}"
            )

    def test_each_symbol_with_light_noise(self):
        """Symbol detection should survive 5 % Gaussian noise."""
        for sym_val, freq in enumerate(config.FREQUENCIES):
            tone = dsp.generate_tone(freq, config.SYMBOL_DURATION, SR)
            noise = np.random.randn(len(tone)).astype(np.float32) * 0.05
            assert dsp.detect_symbol(tone + noise, SR) == sym_val, (
                f"Tone at {freq} Hz with noise should still be symbol {sym_val}"
            )

    def test_silence_returns_minus_one(self):
        silence = np.zeros(int(SR * config.SYMBOL_DURATION), dtype=np.float32)
        assert dsp.detect_symbol(silence, SR) == -1


class TestBitsToWaveform:
    def test_empty_bits(self):
        """Only the preamble should be present."""
        wav = dsp.bits_to_waveform([], SR)
        expected_len = int(SR * config.PREAMBLE_DURATION)
        assert len(wav) == expected_len

    def test_single_bit_padded_to_one_symbol(self):
        """One bit is zero-padded to a dibit → one symbol appended."""
        wav = dsp.bits_to_waveform([0], SR)
        expected_len = int(SR * (config.PREAMBLE_DURATION + config.SYMBOL_DURATION))
        assert len(wav) == expected_len

    def test_10_bits_six_symbols(self):
        """One UART frame = START + 4 data + STOP → preamble + 6 symbol segments."""
        bits = [0, 1, 0, 0, 0, 1, 0, 0, 0, 1]  # one UART frame
        wav = dsp.bits_to_waveform(bits, SR)
        # Match how bits_to_waveform accumulates lengths to avoid FP rounding.
        expected = int(SR * config.PREAMBLE_DURATION) + 6 * int(SR * config.SYMBOL_DURATION)
        assert len(wav) == expected

    def test_preamble_tone_is_preamble_freq(self):
        """A chunk-sized window of PREAMBLE_FREQ should be detected as the STOP tone."""
        chunk = dsp.generate_tone(config.PREAMBLE_FREQ, config.CHUNK_DURATION, SR)
        sym = dsp.detect_symbol(chunk, SR)
        assert sym == config.STOP_INDEX, (
            f"Preamble chunk should be the STOP tone (index {config.STOP_INDEX}), got {sym}"
        )


class TestIsAboveNoiseFloor:
    def test_silence_is_below_floor(self):
        silence = np.zeros(1000, dtype=np.float32)
        assert not dsp.is_above_noise_floor(silence)

    def test_loud_tone_is_above_floor(self):
        tone = dsp.generate_tone(config.PREAMBLE_FREQ, 0.1, SR)
        assert dsp.is_above_noise_floor(tone)
