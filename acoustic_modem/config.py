"""
config.py — All tuneable constants for the acoustic modem.
"""

# ── Frequencies ──────────────────────────────────────────────────────────────
FREQ_0 = 1175       # Hz  —  Logic 0
FREQ_1 = 2350       # Hz  —  Logic 1

# ── Baud / timing ────────────────────────────────────────────────────────────
BAUD_RATE = 3                         # bits per second
BIT_DURATION = 1.0 / BAUD_RATE        # seconds per bit

# ── UART frame ───────────────────────────────────────────────────────────────
DATA_BITS = 8
# Total bits per character frame: 1 start + 8 data + 1 stop = 10
# Time per character = 10 × BIT_DURATION

# ── Preamble ─────────────────────────────────────────────────────────────────
PREAMBLE_DURATION = 0.2     # s — 200 ms of FREQ_1 before first character

# ── Audio ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100         # Hz
CHANNELS = 1
DTYPE = "float32"

# ── Receiver squelch / detection ─────────────────────────────────────────────
# Minimum RMS amplitude to avoid triggering on silence. Tune for your hardware:
#   • Lower (e.g. 1e-5) if the transmitting laptop is quiet or far away.
#   • Raise (e.g. 5e-4) if background noise keeps falsely triggering the receiver.
NOISE_FLOOR = 1e-4

# Fraction of total spectral energy that must be at FREQ_1 to count as preamble.
# Human voices are broadband; our FSK tones are narrowband, so this can be high.
#   • Lower (e.g. 0.25) in noisy rooms where background raises the noise floor.
#   • Keep at 0.40+ in quiet rooms for best false-positive rejection.
SNR_THRESHOLD = 0.40        # 40 % of total energy must be at FREQ_1

# How long FREQ_1 must be continuously stable before the Rx "arms" itself.
# Must be < PREAMBLE_DURATION (200 ms). Raise to reduce false triggers.
PREAMBLE_LOCK_DURATION = 0.12   # s

# After the last stop bit, how long to wait for the next start bit before
# returning to IDLE.  MUST be > BIT_DURATION or multi-character messages break.
# Derived automatically so it stays correct when BAUD_RATE is changed.
INTER_CHAR_TIMEOUT = 1.5 * BIT_DURATION   # s  (1.5 bit-durations of headroom)

# ── Analysis chunk ───────────────────────────────────────────────────────────
# We analyse audio in chunks of this size. Must be << BIT_DURATION.
# Smaller = more responsive; larger = better frequency resolution.
CHUNK_DURATION = 0.02       # s  (20 ms)
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)   # samples per chunk
