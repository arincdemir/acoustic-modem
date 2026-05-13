"""
rx.py — Receiver: audio stream → 4-state machine → decoded characters.

State machine:
    IDLE     → waiting for preamble (3 squelch conditions must all pass)
    ARMED    → preamble detected, waiting for start bit (2000 Hz / logic 0)
    READING  → clocked: sampling 8 data bits then stop bit
    WAITING  → between characters in the same message; no new preamble needed

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
import sounddevice as sd

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
        # Feed a silence chunk to flush the WAITING→IDLE timeout
        silence = np.zeros(chunk_size, dtype=config.DTYPE)
        for _ in range(int(config.INTER_CHAR_TIMEOUT / config.CHUNK_DURATION) + 2):
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
        Check three squelch conditions for the preamble (3000 Hz):
          1. Above noise floor (RMS threshold)
          2. SNR — FREQ_1 dominates the spectrum
          3. Chunk-count lock — tone must be stable for PREAMBLE_LOCK_CHUNKS consecutive chunks

        Timing is tracked in chunk counts (not wall-clock time) so that
        loopback mode (where chunks are processed instantly) behaves
        identically to live mode.
        """
        above_floor = dsp.is_above_noise_floor(chunk)
        preamble_detected = above_floor and (dsp.detect_bit(chunk, self._sample_rate) == 1)

        if preamble_detected:
            self._preamble_chunks += 1
            if self._preamble_chunks >= _PREAMBLE_LOCK_CHUNKS:
                # Tone has been stable for long enough — arm the receiver
                self._preamble_chunks = 0
                self._set_state(RxState.ARMED)
        else:
            self._preamble_chunks = 0   # reset lock counter

    # ── ARMED: wait for start bit (0 = 2000 Hz) ──────────────────────────────

    def _handle_armed(self, chunk: np.ndarray) -> None:
        """Wait for the transition from preamble (FREQ_1) to start bit (FREQ_0)."""
        if not dsp.is_above_noise_floor(chunk):
            # Lost the signal entirely — back to IDLE
            self._set_state(RxState.IDLE)
            return

        bit = dsp.detect_bit(chunk, self._sample_rate)
        if bit == 0:
            # Start bit detected — begin reading data bits
            self._data_bits = []
            self._set_state(RxState.READING)
            # Consume the start bit: schedule first data sample at 1.5 × BIT_DURATION
            # We achieve this by skipping 1.5 bit-durations worth of chunks.
            self._samples_to_skip = int(
                1.5 * config.BIT_DURATION / config.CHUNK_DURATION
            )
            self._bits_remaining = config.DATA_BITS
            self._reading_stop = False  # next after data is the stop bit

    # ── READING: clock in data bits then stop bit ─────────────────────────────

    def _handle_reading(self, chunk: np.ndarray) -> None:
        """
        After the start-bit edge we sample at:
          • 1.5 × BIT_DURATION  for the first data bit
          • every 1.0 × BIT_DURATION  for subsequent bits
        We count chunks to track timing.
        """
        if self._samples_to_skip > 0:
            self._samples_to_skip -= 1
            return

        bit = dsp.detect_bit(chunk, self._sample_rate)

        if bit == -1:
            # Signal corrupted or lost, abort character
            self._data_bits = []
            self._waiting_chunks = 0
            self._set_state(RxState.WAITING)
            return

        if not self._reading_stop and self._bits_remaining > 0:
            # Reading a data bit
            self._data_bits.append(bit)
            self._bits_remaining -= 1
            if self._bits_remaining == 0:
                # Done with data — next sample is the stop bit
                self._reading_stop = True
                self._samples_to_skip = int(
                    1.0 * config.BIT_DURATION / config.CHUNK_DURATION
                ) - 1
        elif self._reading_stop:
            # This is the stop bit sample
            if bit == 1:
                # Valid stop bit — decode and emit character
                try:
                    char = self._decode_char(self._data_bits)
                    self._on_char(char)
                except ValueError:
                    pass  # corrupted frame — silently discard
            # Either way, transition to WAITING for the next start bit
            self._data_bits = []
            self._waiting_chunks = 0
            self._set_state(RxState.WAITING)

        # Schedule next sample 1 bit duration away
        if self._state == RxState.READING:
            self._samples_to_skip = int(
                1.0 * config.BIT_DURATION / config.CHUNK_DURATION
            ) - 1

    # ── WAITING: listen for next start bit without a new preamble ────────────

    def _handle_waiting(self, chunk: np.ndarray) -> None:
        """
        Between characters in the same message: listen for 2000 Hz (start bit).
        If no start bit is seen within INTER_CHAR_TIMEOUT_CHUNKS chunks, go to IDLE.
        Timing uses chunk counts for consistency with loopback mode.
        """
        self._waiting_chunks += 1
        if self._waiting_chunks > _INTER_CHAR_TIMEOUT_CHUNKS:
            self._set_state(RxState.IDLE)
            return

        if dsp.is_above_noise_floor(chunk):
            bit = dsp.detect_bit(chunk, self._sample_rate)
            if bit == 0:
                # Next start bit found — go straight to READING
                self._data_bits = []
                self._set_state(RxState.READING)
                self._samples_to_skip = int(
                    1.5 * config.BIT_DURATION / config.CHUNK_DURATION
                )
                self._bits_remaining = config.DATA_BITS
                self._reading_stop = False

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
