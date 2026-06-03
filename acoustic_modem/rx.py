"""
rx.py — Receiver: audio stream → 4-state machine → decoded characters.

State machine:
    IDLE     → waiting for preamble (PREAMBLE_FREQ == STOP tone must be stable)
    ARMED    → preamble detected, waiting for the START tone
    READING  → clocked: sampling 4 data tones then 1 STOP tone
    WAITING  → between characters in the same message; no new preamble needed

Six-tone FSK (see config.py / dsp.py):
    Data tones (index 0–3): FREQ_00, FREQ_01, FREQ_10, FREQ_11  (one dibit each)
    START tone (index 4):   FREQ_START   — the single start bit
    STOP  tone (index 5):   FREQ_STOP    — the single stop bit (and idle/preamble)

Each UART frame (10 bits) is transmitted as 6 tones:
    START, [d0 d1], [d2 d3], [d4 d5], [d6 d7], STOP
Because START and STOP have their own dedicated frequencies, the receiver can
never confuse a framing bit with a data symbol.

The receiver can operate in two modes:
  • Live mode:    reads from the microphone via sounddevice.InputStream.
  • Loopback mode: reads from a pre-generated np.ndarray buffer.

Timing note
-----------
All timing is expressed in **chunk counts** (how many CHUNK_DURATION windows
represent a given duration). This is deterministic in both loopback mode (where
chunks are processed far faster than real time) and live mode (where the audio
card delivers exactly one chunk every CHUNK_DURATION seconds).
"""

from __future__ import annotations

import math
import queue
import threading
from enum import Enum, auto
from typing import Callable

import numpy as np

from acoustic_modem import config, dsp

# Precomputed chunk-count thresholds — chunk counts are deterministic in both
# loopback mode and live mode (each chunk = CHUNK_DURATION seconds of audio).
_PREAMBLE_LOCK_CHUNKS      = math.ceil(config.PREAMBLE_LOCK_DURATION / config.CHUNK_DURATION)
_INTER_CHAR_TIMEOUT_CHUNKS = math.ceil(config.INTER_CHAR_TIMEOUT     / config.CHUNK_DURATION)


# ── State machine states ──────────────────────────────────────────────────────

class RxState(Enum):
    IDLE    = auto()
    ARMED   = auto()
    READING = auto()
    WAITING = auto()


# ── Receiver ──────────────────────────────────────────────────────────────────

class Receiver:
    """
    Acoustic modem receiver.

    Parameters
    ----------
    on_char : callable
        Called with each decoded character as soon as it is received.
    on_state_change : callable, optional
        Called with the new RxState whenever the state changes.
    sample_rate : int
        Audio sample rate.
    device : int | None
        sounddevice input device index.  None = system default.
    """

    def __init__(
        self,
        on_char: Callable[[str], None],
        on_state_change: Callable[[RxState], None] | None = None,
        sample_rate: int = config.SAMPLE_RATE,
        device: int | None = None,
    ) -> None:
        self._on_char = on_char
        self._on_state_change = on_state_change
        self._sample_rate = sample_rate
        self._device = device

        self._chunk_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._muted = threading.Event()   # set = muted (Tx is active)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._state = RxState.IDLE
        self._preamble_chunks = 0   # consecutive chunks detecting preamble
        self._data_bits: list[int] = []
        self._waiting_chunks = 0    # chunks elapsed since entering WAITING
        self._samples_to_skip = 0           # chunks to skip until the next sample point
        self._data_symbols_remaining = 0    # data tones still to read in this frame
        self._reading_stop = False          # True once the next tone is the STOP tone

    # ── Public API ────────────────────────────────────────────────────────────

    def mute(self) -> None:
        """Silence the receiver (call before Tx starts)."""
        self._muted.set()

    def unmute(self) -> None:
        """Resume the receiver (call after Tx finishes)."""
        self._muted.clear()

    def start(self, loopback_buffer: np.ndarray | None = None) -> None:
        """
        Start the receiver.

        If `loopback_buffer` is provided, the receiver processes that buffer
        instead of reading from the microphone.
        """
        self._stop_event.clear()
        if loopback_buffer is not None:
            self._thread = threading.Thread(
                target=self._run_loopback,
                args=(loopback_buffer,),
                daemon=True,
            )
        else:
            self._thread = threading.Thread(
                target=self._run_live,
                daemon=True,
            )
        self._thread.start()

    def stop(self) -> None:
        """Stop the receiver and wait for the thread to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def wait_until_done(self, timeout: float | None = None) -> None:
        """Block until the receiver thread exits (useful in loopback mode)."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ── Internal: live mode ───────────────────────────────────────────────────

    def _sd_callback(self, indata: np.ndarray, frames: int,
                     time_info, status) -> None:
        """sounddevice callback — runs in the audio thread."""
        if not self._muted.is_set():
            chunk = indata[:, 0].copy().astype(config.DTYPE)
            try:
                self._chunk_q.put_nowait(chunk)
            except queue.Full:
                pass  # drop the chunk rather than block the audio thread

    def _run_live(self) -> None:
        """Open the mic stream and process chunks until stop() is called."""
        import sounddevice as sd

        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=config.CHANNELS,
            dtype=config.DTYPE,
            blocksize=config.CHUNK_SIZE,
            device=self._device,
            callback=self._sd_callback,
        ):
            while not self._stop_event.is_set():
                try:
                    chunk = self._chunk_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                self._process_chunk(chunk)

    # ── Internal: loopback mode ───────────────────────────────────────────────

    def _run_loopback(self, buffer: np.ndarray) -> None:
        """
        Slice a pre-generated waveform into CHUNK_SIZE pieces and process each
        one as if it had arrived from the microphone.
        """
        chunk_size = config.CHUNK_SIZE
        n = len(buffer)
        for start in range(0, n, chunk_size):
            if self._stop_event.is_set():
                break
            chunk = buffer[start : start + chunk_size].astype(config.DTYPE)
            if len(chunk) < chunk_size:
                # Pad the final short chunk with silence
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            self._process_chunk(chunk)
        # Feed silence to flush the WAITING→IDLE timeout.
        silence = np.zeros(chunk_size, dtype=config.DTYPE)
        for _ in range(_INTER_CHAR_TIMEOUT_CHUNKS + 2):
            self._process_chunk(silence)

    # ── State machine ─────────────────────────────────────────────────────────

    def _set_state(self, new_state: RxState) -> None:
        if new_state != self._state:
            self._state = new_state
            if self._on_state_change:
                self._on_state_change(new_state)

    def _process_chunk(self, chunk: np.ndarray) -> None:
        """Route each audio chunk through the current state handler."""
        if self._state == RxState.IDLE:
            self._handle_idle(chunk)
        elif self._state == RxState.ARMED:
            self._handle_armed(chunk)
        elif self._state == RxState.READING:
            self._handle_reading(chunk)
        elif self._state == RxState.WAITING:
            self._handle_waiting(chunk)

    # ── IDLE: detect preamble ─────────────────────────────────────────────────

    def _handle_idle(self, chunk: np.ndarray) -> None:
        """
        Check three squelch conditions for the preamble (PREAMBLE_FREQ == STOP tone):
          1. Above noise floor (RMS threshold)
          2. SNR — PREAMBLE_FREQ dominates the spectrum (detect_symbol == STOP_INDEX)
          3. Chunk-count lock — tone must be stable for PREAMBLE_LOCK_CHUNKS consecutive chunks

        Timing is tracked in chunk counts (not wall-clock time) so that
        loopback mode (where chunks are processed instantly) behaves
        identically to live mode.
        """
        above_floor = dsp.is_above_noise_floor(chunk)
        preamble_detected = above_floor and (
            dsp.detect_symbol(chunk, self._sample_rate) == config.STOP_INDEX
        )

        if preamble_detected:
            self._preamble_chunks += 1
            if self._preamble_chunks >= _PREAMBLE_LOCK_CHUNKS:
                # Tone has been stable for long enough — arm the receiver
                self._preamble_chunks = 0
                self._set_state(RxState.ARMED)
        else:
            self._preamble_chunks = 0   # reset lock counter

    # ── ARMED: wait for the START tone ───────────────────────────────────────

    def _handle_armed(self, chunk: np.ndarray) -> None:
        """
        Wait for the transition from the preamble (STOP tone) to the START tone.

        The START tone is a dedicated frequency (index 4) and carries no data —
        it only marks the beginning of a frame.
        """
        if not dsp.is_above_noise_floor(chunk):
            # Lost the signal entirely — back to IDLE
            self._set_state(RxState.IDLE)
            return

        if dsp.detect_symbol(chunk, self._sample_rate) == config.START_INDEX:
            self._begin_reading()

    # ── READING: clock in 4 data tones then the STOP tone ────────────────────

    def _handle_reading(self, chunk: np.ndarray) -> None:
        """
        After the START tone we sample at:
          • 1.5 × SYMBOL_DURATION  for the first data tone (centre of the symbol)
          • every 1.0 × SYMBOL_DURATION  for subsequent symbols

        _data_symbols_remaining counts the data tones still to read (4 in total,
        2 bits each).  Once all four are read the next sample is the STOP tone.
        """
        if self._samples_to_skip > 0:
            self._samples_to_skip -= 1
            return

        sym = dsp.detect_symbol(chunk, self._sample_rate)

        if sym == -1:
            # Signal corrupted or lost — abort character
            self._enter_waiting()
            return

        if not self._reading_stop:
            # Expect a data tone (index 0–3); anything else means a corrupt frame.
            if sym not in (0, 1, 2, 3):
                self._enter_waiting()
                return
            # Data symbol: extract 2 bits (MSB first)
            self._data_bits.append((sym >> 1) & 1)
            self._data_bits.append(sym & 1)
            self._data_symbols_remaining -= 1
            if self._data_symbols_remaining == 0:
                # Done with data tones — the next sample is the STOP tone
                self._reading_stop = True
            self._samples_to_skip = self._symbol_skip()
        else:
            # Expect the STOP tone (index 5).  A valid STOP completes the frame.
            if sym == config.STOP_INDEX:
                try:
                    char = self._decode_char(self._data_bits)
                    self._on_char(char)
                except ValueError:
                    pass  # corrupted frame — silently discard
            # Whether the STOP tone was valid or not, this frame is finished.
            self._enter_waiting()

    # ── WAITING: listen for the next START tone without a new preamble ───────

    def _handle_waiting(self, chunk: np.ndarray) -> None:
        """
        Between characters in the same message: listen for the START tone
        (index 4).  If none is seen within INTER_CHAR_TIMEOUT_CHUNKS chunks,
        go to IDLE.  Timing uses chunk counts for consistency with loopback mode.

        The STOP tone (index 5) at the end of the previous frame cannot trigger
        the START-tone check, so no extra guard is required.
        """
        self._waiting_chunks += 1
        if self._waiting_chunks > _INTER_CHAR_TIMEOUT_CHUNKS:
            self._set_state(RxState.IDLE)
            return

        if dsp.is_above_noise_floor(chunk):
            if dsp.detect_symbol(chunk, self._sample_rate) == config.START_INDEX:
                self._begin_reading()

    # ── Reading helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _symbol_skip() -> int:
        """Chunks to skip to advance exactly one symbol to the next sample."""
        return int(1.0 * config.SYMBOL_DURATION / config.CHUNK_DURATION) - 1

    def _begin_reading(self) -> None:
        """Start reading a frame after a START tone: 4 data tones then STOP."""
        self._data_bits = []
        self._data_symbols_remaining = 4
        self._reading_stop = False
        # Skip to the centre of the first data symbol (1.5 × SYMBOL_DURATION away).
        self._samples_to_skip = int(
            1.5 * config.SYMBOL_DURATION / config.CHUNK_DURATION
        )
        self._set_state(RxState.READING)

    def _enter_waiting(self) -> None:
        """Finish the current frame and listen for the next START tone."""
        self._data_bits = []
        self._waiting_chunks = 0
        self._set_state(RxState.WAITING)

    # ── Decoding helper ───────────────────────────────────────────────────────

    @staticmethod
    def _decode_char(data_bits: list[int]) -> str:
        """Convert 8 data bits (LSB-first) to an ASCII character."""
        if len(data_bits) != config.DATA_BITS:
            raise ValueError(f"Expected {config.DATA_BITS} bits, got {len(data_bits)}")
        code = sum(bit << i for i, bit in enumerate(data_bits))
        if code > 127:
            raise ValueError(f"Non-ASCII character code: {code}")
        return chr(code)
