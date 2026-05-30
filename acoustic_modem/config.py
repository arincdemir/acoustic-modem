"""
config.py — All tuneable constants for the acoustic modem.
"""

# ── Frequencies (4-FSK) ───────────────────────────────────────────────────────
FREQ_00 = 1000       # Hz  —  dibit 00
FREQ_01 = 2000       # Hz  —  dibit 01
FREQ_10 = 3000       # Hz  —  dibit 10
FREQ_11 = 4000       # Hz  —  dibit 11

FREQUENCIES = [FREQ_00, FREQ_01, FREQ_10, FREQ_11]  # indexed by dibit value (0–3)
PREAMBLE_FREQ = FREQ_11                               # preamble tone ("all-ones" dibit)

# ── Baud / timing ────────────────────────────────────────────────────────────
BAUD_RATE = 5                           # symbols per second  (each symbol = 2 bits → 10 bps)
SYMBOL_DURATION = 1.0 / BAUD_RATE      # seconds per symbol

# ── UART frame ───────────────────────────────────────────────────────────────
DATA_BITS = 8
# Total bits per character frame: 1 start + 8 data + 1 stop = 10 bits = 5 symbols
# Time per character = 5 × SYMBOL_DURATION

# ── Preamble ─────────────────────────────────────────────────────────────────
PREAMBLE_DURATION = 0.4     # s — 400 ms of PREAMBLE_FREQ before first character

# ── Audio ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100         # Hz
CHANNELS = 1
DTYPE = "float32"

# ── Receiver squelch / detection ─────────────────────────────────────────────
NOISE_FLOOR = 3e-4

# Fraction of total spectral energy that must be at the dominant frequency.
# Human voices are broadband; FSK tones are narrowband, so this can be high.
#   • Lower (e.g. 0.25) in noisy rooms.
#   • Keep at 0.40+ in quiet rooms for best false-positive rejection.
SNR_THRESHOLD = 0.40

# How long PREAMBLE_FREQ must be continuously stable before the Rx "arms" itself.
# Must be < PREAMBLE_DURATION.
PREAMBLE_LOCK_DURATION = 0.2   # s

# After the last stop symbol, how long to wait for the next start symbol before
# returning to IDLE.  MUST be > SYMBOL_DURATION or multi-character messages break.
INTER_CHAR_TIMEOUT = 1.5 * SYMBOL_DURATION   # s  (1.5 symbol-durations of headroom)

# ── Analysis chunk ───────────────────────────────────────────────────────────
# We analyse audio in chunks of this size. Must be << SYMBOL_DURATION.
# Smaller = more responsive; larger = better frequency resolution.
CHUNK_DURATION = 0.02       # s  (20 ms)
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)   # samples per chunk
