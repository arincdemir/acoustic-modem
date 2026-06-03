"""
dsp.py — DSP utilities: tone generation, Goertzel energy, symbol detection.

Six-tone FSK on the A blues scale (see config.py).  The tones split into two
roles:

  Data tones (4-FSK, two bits each):
    Dibit 00  →  FREQ_00  (880 Hz)
    Dibit 01  →  FREQ_01  (1319 Hz)
    Dibit 10  →  FREQ_10  (2093 Hz)
    Dibit 11  →  FREQ_11  (3136 Hz)

  Single-bit framing tones:
    START     →  FREQ_START (587 Hz)   — index 4
    STOP      →  FREQ_STOP  (4699 Hz)  — index 5 (also the preamble/idle tone)

A character frame is transmitted as 6 symbols: START + 4 data dibits + STOP.
`detect_symbol` returns the index (0–5) of the dominant tone, or -1 if none
clears the SNR threshold.
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


def _generate_tone_sequence(tones: list[tuple[float, float]],
                            sample_rate: int = config.SAMPLE_RATE) -> np.ndarray:
    """Generate tones while carrying oscillator phase across boundaries."""
    segments: list[np.ndarray] = []
    phase = 0.0

    for freq, duration in tones:
        n_samples = int(sample_rate * duration)
        if n_samples <= 0:
            continue

        phase_step = 2 * np.pi * freq / sample_rate
        phases = phase + phase_step * np.arange(n_samples)
        segments.append(np.sin(phases).astype(config.DTYPE))
        phase = (phase + phase_step * n_samples) % (2 * np.pi)

    if not segments:
        return np.array([], dtype=config.DTYPE)
    return np.concatenate(segments)


def apply_release_ramp(waveform: np.ndarray,
                       duration: float = config.TX_RELEASE_DURATION,
                       sample_rate: int = config.SAMPLE_RATE) -> np.ndarray:
    """Return `waveform` with a short fade-out over its final samples."""
    ramp_samples = int(sample_rate * duration)
    if len(waveform) == 0 or ramp_samples <= 0:
        return waveform.astype(config.DTYPE, copy=True)

    ramp_samples = min(ramp_samples, len(waveform))
    faded = waveform.astype(config.DTYPE, copy=True)
    ramp = np.linspace(1.0, 0.0, ramp_samples, dtype=config.DTYPE)
    faded[-ramp_samples:] *= ramp
    return faded


def bits_to_waveform(bits: list[int],
                     sample_rate: int = config.SAMPLE_RATE) -> np.ndarray:
    """
    Convert a flat bit list into a complete six-tone FSK audio waveform.

    Prepends the preamble (PREAMBLE_DURATION of PREAMBLE_FREQ), then encodes
    the bits as a sequence of tones, each SYMBOL_DURATION seconds long.

    UART-frame-aware encoding (when len(bits) is a multiple of 10):
        Each 10-bit frame [start, d0..d7, stop] becomes 6 tones:
          • One START tone   (FREQ_START) — the single start bit.
          • Four data tones  — the 8 data bits, two bits per tone (MSB-first):
                sym = (data[2j] << 1) | data[2j+1]
          • One STOP tone    (FREQ_STOP)  — the single stop bit.
        Because START and STOP have their own dedicated frequencies, the
        receiver can never confuse a framing bit with a data symbol.

    For inputs that are NOT multiples of 10 bits (edge cases / unit tests),
    the bits are simply paired into data dibits with no framing tones
    (odd-length inputs are zero-padded to even length).
    """
    tones: list[tuple[float, float]] = [
        (config.PREAMBLE_FREQ, config.PREAMBLE_DURATION),
    ]

    if len(bits) > 0 and len(bits) % 10 == 0:
        # Full UART frames — START tone, 4 data tones, STOP tone.
        for frame_start in range(0, len(bits), 10):
            frame = bits[frame_start:frame_start + 10]
            tones.append((config.FREQ_START, config.SYMBOL_DURATION))
            data = frame[1:9]  # 8 data bits, LSB-first
            for j in range(4):
                sym = (data[2 * j] << 1) | data[2 * j + 1]
                tones.append((config.DATA_FREQUENCIES[sym],
                              config.SYMBOL_DURATION))
            tones.append((config.FREQ_STOP, config.SYMBOL_DURATION))
    else:
        # Non-frame input (edge cases / tests) — simple sequential dibit pairing.
        padded = bits if len(bits) % 2 == 0 else bits + [0]
        for i in range(0, len(padded), 2):
            sym = (padded[i] << 1) | padded[i + 1]
            tones.append((config.DATA_FREQUENCIES[sym],
                          config.SYMBOL_DURATION))

    return _generate_tone_sequence(tones, sample_rate)


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
    Identify which of the six FSK tones is present in `chunk`.

    Computes the SNR for each frequency in config.FREQUENCIES, selects the one
    with the highest SNR, and returns its index if that SNR clears
    SNR_THRESHOLD.  Returns -1 if no frequency passes the threshold.

    Index meaning (see config.FREQUENCIES):
        0–3 → data dibit value, 4 → START tone, 5 → STOP tone.
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
