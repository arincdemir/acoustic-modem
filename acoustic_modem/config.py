"""
config.py — All tuneable constants for the acoustic modem.
"""

# ── Frequencies (6-tone FSK on the A blues scale) ─────────────────────────────
# Every tone is a note of the A blues scale, whose six pitch classes (semitones
# above A) are {0, 3, 5, 6, 7, 10} = A, C, D, D#, E, G.  The notes are spread
# across several octaves so that neighbouring tones are ~7 semitones (a ~1.5×
# frequency ratio) apart — i.e. as far from one another as possible — which
# keeps detection robust and unambiguous.
#
# Roles:
#   • 4 "data" tones carry one dibit each (4-FSK) → the 8 data bits.
#   • 1 dedicated tone marks the START bit (single bit, its own frequency).
#   • 1 dedicated tone marks the STOP bit  (single bit, its own frequency).
#
# Data tones (4-FSK), indexed by dibit value (0–3):
FREQ_00 = 880    # A5  — data 00
FREQ_01 = 1047   # C6  — data 01
FREQ_10 = 1175   # D6  — data 10
FREQ_11 = 1568   # G6  — data 11

FREQ_START = 1245  # Eb6 / D#6 — start
FREQ_STOP  = 1319  # E6        — stop / preamble


DATA_FREQUENCIES = [FREQ_00, FREQ_01, FREQ_10, FREQ_11]   # indexed by dibit value (0–3)

# All six tones in a fixed index order shared by the detector and the receiver:
#   indices 0..3 → data dibits, index 4 → START, index 5 → STOP.
FREQUENCIES = DATA_FREQUENCIES + [FREQ_START, FREQ_STOP]
START_INDEX = 4
STOP_INDEX  = 5

# The UART idle line rests at the mark (stop) level, so we reuse the STOP tone
# as the preamble / idle tone.
PREAMBLE_FREQ = FREQ_STOP

# ── Baud / timing ────────────────────────────────────────────────────────────
BAUD_RATE = 5                           # symbols (tones) per second
SYMBOL_DURATION = 1.0 / BAUD_RATE      # seconds per symbol

# ── UART frame ───────────────────────────────────────────────────────────────
DATA_BITS = 8
# Tones per character frame: 1 START + 4 data dibits + 1 STOP = 6 symbols.
# Time per character = 6 × SYMBOL_DURATION

# ── Preamble ─────────────────────────────────────────────────────────────────
PREAMBLE_DURATION = 0.4     # s — 400 ms of PREAMBLE_FREQ before first character

# ── Audio ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100         # Hz
CHANNELS = 1
DTYPE = "float32"

# Short fade-out applied only at the end of a transmission, before trailing
# silence, to avoid an audible click when playback stops.
TX_RELEASE_DURATION = 0.005  # s

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
CHUNK_DURATION = 0.05       # s  (20 ms)
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)   # samples per chunk
