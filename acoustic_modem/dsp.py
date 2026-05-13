"""
dsp.py — DSP utilities: tone generation, Goertzel energy, bit detection.

Two-frequency FSK:
  Logic 0  →  FREQ_0  (2000 Hz)
  Logic 1  →  FREQ_1  (3000 Hz)
"""

import numpy as np
from acoustic_modem import config


# ── Tone generation ──────────────────────────────────────────────────────────

def generate_tone(freq: float, duration: float,
                  sample_rate: int = config.SAMPLE_RATE) -> np.ndarray:
    """Return a normalised sine wave at `freq` Hz for `duration` seconds."""
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate
    return np.sin(2 * np.pi * freq * t).astype(config.DTYPE)


def bits_to_waveform(bits: list[int],
                     sample_rate: int = config.SAMPLE_RATE) -> np.ndarray:
    """
    Convert a flat bit list into a complete audio waveform.

    Prepends the preamble (PREAMBLE_DURATION of FREQ_1), then maps each bit
    to a tone segment of BIT_DURATION seconds.
    """
    segments: list[np.ndarray] = []

    # Preamble
    segments.append(generate_tone(config.FREQ_1, config.PREAMBLE_DURATION,
                                  sample_rate))

    # Bit segments
    for bit in bits:
        freq = config.FREQ_1 if bit == 1 else config.FREQ_0
        segments.append(generate_tone(freq, config.BIT_DURATION, sample_rate))

    return np.concatenate(segments)


# ── Goertzel algorithm ───────────────────────────────────────────────────────

def goertzel_energy(signal: np.ndarray, target_freq: float,
                    sample_rate: int = config.SAMPLE_RATE) -> float:
    """
    Compute the energy (power) of `signal` at `target_freq` Hz using the
    Goertzel algorithm — more efficient than a full FFT for one or two
    target frequencies.

    Returns a non-negative float.
    """
    n = len(signal)
    if n == 0:
        return 0.0

    k = round(n * target_freq / sample_rate)
    omega = 2.0 * np.pi * k / n
    coeff = 2.0 * np.cos(omega)

    s_prev2 = 0.0
    s_prev1 = 0.0
    for sample in signal:
        s = float(sample) + coeff * s_prev1 - s_prev2
        s_prev2 = s_prev1
        s_prev1 = s

    # Power = s_prev1^2 + s_prev2^2 - coeff * s_prev1 * s_prev2
    power = s_prev1 ** 2 + s_prev2 ** 2 - coeff * s_prev1 * s_prev2
    return max(power, 0.0)


def total_energy(signal: np.ndarray) -> float:
    """Sum of squared samples — proportional to total signal power."""
    return float(np.dot(signal, signal))


# ── Bit detection ─────────────────────────────────────────────────────────────

def detect_bit(chunk: np.ndarray,
               sample_rate: int = config.SAMPLE_RATE) -> int:
    """
    Determine the logic value of `chunk` by checking if the SNR for FREQ_0 or FREQ_1
    is above the threshold. Returns 1 if FREQ_1 is dominant and valid, 0 if FREQ_0
    is dominant and valid, or -1 if neither pass the SNR threshold.
    """
    snr_0 = compute_snr(chunk, config.FREQ_0, sample_rate)
    snr_1 = compute_snr(chunk, config.FREQ_1, sample_rate)
    
    if snr_1 >= config.SNR_THRESHOLD and snr_1 >= snr_0:
        return 1
    elif snr_0 >= config.SNR_THRESHOLD:
        return 0
    else:
        return -1


def compute_snr(chunk: np.ndarray, target_freq: float,
                sample_rate: int = config.SAMPLE_RATE) -> float:
    """
    Return the fraction of total signal energy that is at `target_freq`.
    Range: [0.0, 1.0].  Returns 0.0 for silent input.
    """
    total = total_energy(chunk)
    if total < 1e-12:
        return 0.0
    target = goertzel_energy(chunk, target_freq, sample_rate)
    # Scale Goertzel power to the same units as total_energy:
    # Goertzel returns N^2/4 × amplitude^2 for a pure tone.
    # Normalise by len(chunk)^2 / 4 so SNR is in [0, 1].
    n = len(chunk)
    normalised_target = target / (n ** 2 / 4.0) if n > 0 else 0.0
    return min(normalised_target / (total / n), 1.0)


def is_above_noise_floor(chunk: np.ndarray,
                         floor: float = config.NOISE_FLOOR) -> bool:
    """Return True if the chunk's RMS energy exceeds the noise floor."""
    rms = float(np.sqrt(np.mean(chunk ** 2)))
    return rms >= floor
