"""
dsp.py — DSP utilities: tone generation, Goertzel energy, symbol detection.

Four-frequency FSK (4-FSK):
  Dibit 00  →  FREQ_00  (1000 Hz)
  Dibit 01  →  FREQ_01  (2000 Hz)
  Dibit 10  →  FREQ_10  (3000 Hz)
  Dibit 11  →  FREQ_11  (4000 Hz)

Each symbol carries 2 bits.  A 10-bit UART frame is transmitted as 5 symbols.
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
    Convert a flat bit list into a complete 4-FSK audio waveform.

    Prepends the preamble (PREAMBLE_DURATION of PREAMBLE_FREQ), then encodes
    each pair of bits as a tone of SYMBOL_DURATION seconds.

    Dibit encoding (MSB-first):  sym = (bits[i] << 1) | bits[i+1]

    UART-frame-aware encoding (when len(bits) is a multiple of 10):
        For each 10-bit UART frame the last dibit is sent as [stop, d7]
        instead of the sequential [d7, stop].  This guarantees:
          • Start symbols always fall in {0, 1}  (start bit is MSB = 0)
          • Stop  symbols always fall in {2, 3}  (stop  bit is MSB = 1)
        so the WAITING receiver can never confuse the two.

    For inputs that are NOT multiples of 10 bits, simple sequential dibit
    pairing is used (odd-length inputs are zero-padded to even length).
    """
    segments: list[np.ndarray] = []

    segments.append(generate_tone(config.PREAMBLE_FREQ, config.PREAMBLE_DURATION,
                                  sample_rate))

    if len(bits) > 0 and len(bits) % 10 == 0:
        # Full UART frames — use stop-bit-first last dibit per frame.
        for frame_start in range(0, len(bits), 10):
            frame = bits[frame_start:frame_start + 10]
            for j in range(4):
                sym = (frame[2 * j] << 1) | frame[2 * j + 1]
                segments.append(generate_tone(config.FREQUENCIES[sym],
                                              config.SYMBOL_DURATION, sample_rate))
            # Last dibit: [stop=frame[9], d7=frame[8]] → sym ∈ {2, 3}
            sym = (frame[9] << 1) | frame[8]
            segments.append(generate_tone(config.FREQUENCIES[sym],
                                          config.SYMBOL_DURATION, sample_rate))
    else:
        # Non-frame input (edge cases / tests) — simple sequential dibit pairing.
        padded = bits if len(bits) % 2 == 0 else bits + [0]
        for i in range(0, len(padded), 2):
            sym = (padded[i] << 1) | padded[i + 1]
            segments.append(generate_tone(config.FREQUENCIES[sym],
                                          config.SYMBOL_DURATION, sample_rate))

    return np.concatenate(segments)


# ── Goertzel algorithm ───────────────────────────────────────────────────────

def goertzel_energy(signal: np.ndarray, target_freq: float,
                    sample_rate: int = config.SAMPLE_RATE) -> float:
    """
    Compute the energy (power) of `signal` at `target_freq` Hz using the
    Goertzel algorithm — more efficient than a full FFT for a small number of
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

    power = s_prev1 ** 2 + s_prev2 ** 2 - coeff * s_prev1 * s_prev2
    return max(power, 0.0)


def total_energy(signal: np.ndarray) -> float:
    """Sum of squared samples — proportional to total signal power."""
    return float(np.dot(signal, signal))


# ── Symbol detection ─────────────────────────────────────────────────────────

def detect_symbol(chunk: np.ndarray,
                  sample_rate: int = config.SAMPLE_RATE) -> int:
    """
    Identify which of the four 4-FSK tones is present in `chunk`.

    Computes the SNR for each frequency in config.FREQUENCIES, selects the one
    with the highest SNR, and returns its dibit value (0–3) if that SNR clears
    SNR_THRESHOLD.  Returns -1 if no frequency passes the threshold.
    """
    snrs = [compute_snr(chunk, freq, sample_rate) for freq in config.FREQUENCIES]
    best = int(np.argmax(snrs))
    if snrs[best] >= config.SNR_THRESHOLD:
        return best
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
    n = len(chunk)
    normalised_target = target / (n ** 2 / 4.0) if n > 0 else 0.0
    return min(normalised_target / (total / n), 1.0)


def is_above_noise_floor(chunk: np.ndarray,
                         floor: float = config.NOISE_FLOOR) -> bool:
    """Return True if the chunk's RMS energy exceeds the noise floor."""
    rms = float(np.sqrt(np.mean(chunk ** 2)))
    return rms >= floor
