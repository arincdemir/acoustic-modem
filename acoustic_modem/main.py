"""
main.py — Entry point for the acoustic modem.

Usage:
    python -m acoustic_modem                          # live mode
    python -m acoustic_modem --loopback "Hello World" # in-process test
    python -m acoustic_modem --diagnose               # show mic signal levels
    python -m acoustic_modem --list-devices           # show audio devices
"""

from __future__ import annotations

import argparse
import sys
import threading

import numpy as np
import sounddevice as sd

from acoustic_modem import config, dsp
from acoustic_modem.rx import Receiver, RxState
from acoustic_modem.tx import transmit_message


# ── Terminal helpers ──────────────────────────────────────────────────────────

_state_label: dict[RxState, str] = {
    RxState.IDLE:    "[IDLE]",
    RxState.ARMED:   "[ARMED]",
    RxState.READING: "[READING]",
    RxState.WAITING: "[WAITING]",
}

_current_state = RxState.IDLE
_state_lock = threading.Lock()


def _print_status(state: RxState) -> None:
    global _current_state
    with _state_lock:
        _current_state = state
    label = _state_label.get(state, str(state))
    # Overwrite current line with state label
    sys.stdout.write(f"\r{label}            \n")
    sys.stdout.flush()


def _on_char(ch: str) -> None:
    sys.stdout.write(ch)
    sys.stdout.flush()


# ── Loopback mode ─────────────────────────────────────────────────────────────

def run_loopback(text: str) -> None:
    from acoustic_modem.tx import build_waveform

    print(f"[LOOPBACK] Transmitting: {text!r}")
    waveform = build_waveform(text, config.SAMPLE_RATE)

    received: list[str] = []

    def on_char(ch: str) -> None:
        received.append(ch)

    rx = Receiver(on_char=on_char, sample_rate=config.SAMPLE_RATE)
    rx.start(loopback_buffer=waveform)
    rx.wait_until_done(timeout=60.0)

    result = "".join(received)
    print(f"[LOOPBACK] Received:     {result!r}")
    if result == text:
        print("[LOOPBACK] ✓ Round-trip successful.")
    else:
        print("[LOOPBACK] ✗ Mismatch!")
        sys.exit(1)


# ── Diagnose mode ────────────────────────────────────────────────────────────

def run_diagnose(input_device: int | None) -> None:
    """
    Listen to the microphone and print live signal metrics every chunk.
    Use this to tune NOISE_FLOOR and SNR_THRESHOLD before a real two-machine test.

    Output columns:
        RMS       — overall mic level  (compare to NOISE_FLOOR)
        SNR_1     — SNR for FREQ_1 (compare to SNR_THRESHOLD)
        SNR_0     — SNR for FREQ_0 (compare to SNR_THRESHOLD)
        state     — what the receiver would decide
    """
    print("Diagnose mode — listening to microphone. Press Ctrl-C to stop.")
    print(f"Config: NOISE_FLOOR={config.NOISE_FLOOR:.0e}  "
          f"SNR_THRESHOLD={config.SNR_THRESHOLD:.2f}\n")
    print(f"{'RMS':>10}  {'SNR_1':>10}  {'SNR_0':>10}  state")
    print("-" * 50)

    chunk_buf: list[np.ndarray] = []

    def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        chunk_buf.append(indata[:, 0].copy().astype(config.DTYPE))

    try:
        with sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            dtype=config.DTYPE,
            blocksize=config.CHUNK_SIZE,
            device=input_device,
            callback=callback,
        ):
            while True:
                if not chunk_buf:
                    sd.sleep(int(config.CHUNK_DURATION * 1000))
                    continue
                chunk = chunk_buf.pop(0)
                rms   = float(np.sqrt(np.mean(chunk ** 2)))
                snr_1 = dsp.compute_snr(chunk, config.FREQ_1, config.SAMPLE_RATE)
                snr_0 = dsp.compute_snr(chunk, config.FREQ_0, config.SAMPLE_RATE)

                above_floor = rms >= config.NOISE_FLOOR
                snr_1_ok    = snr_1 >= config.SNR_THRESHOLD
                snr_0_ok    = snr_0 >= config.SNR_THRESHOLD

                if not above_floor:
                    state = "SILENT"
                elif snr_1_ok and (not snr_0_ok or snr_1 >= snr_0):
                    state = "FREQ_1"
                elif snr_0_ok:
                    state = "FREQ_0"
                else:
                    state = "noise/voice"

                rms_flag = "✓" if above_floor else "✗"
                snr_1_flag = "✓" if snr_1_ok else "✗"
                snr_0_flag = "✓" if snr_0_ok else "✗"
                
                print(f"{rms:>9.5f}{rms_flag}  {snr_1:>9.4f}{snr_1_flag}  "
                      f"{snr_0:>9.4f}{snr_0_flag}  {state}")
    except KeyboardInterrupt:
        print("\nDone.")


# ── Live mode ─────────────────────────────────────────────────────────────────

def run_live(input_device: int | None, output_device: int | None) -> None:
    rx = Receiver(
        on_char=_on_char,
        on_state_change=_print_status,
        sample_rate=config.SAMPLE_RATE,
        device=input_device,
    )
    rx.start()

    print("Acoustic Modem — ready.")
    print("Type a message and press Enter to transmit. Ctrl-C to quit.\n")

    try:
        while True:
            try:
                text = input("> ")
            except EOFError:
                break

            if not text:
                continue

            # Half-duplex: mute Rx while Tx is active
            rx.mute()
            sys.stdout.write("[TRANSMITTING]\n")
            sys.stdout.flush()
            try:
                transmit_message(text, sample_rate=config.SAMPLE_RATE,
                                 device=output_device)
            finally:
                rx.unmute()
            sys.stdout.write("[IDLE]\n")
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        rx.stop()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Half-duplex acoustic modem using FSK / UART framing."
    )
    parser.add_argument(
        "--loopback", metavar="TEXT",
        help="Run a loopback test: transmit TEXT and receive it in-process.",
    )
    parser.add_argument(
        "--input-device", type=int, default=None,
        help="sounddevice input device index (default: system default).",
    )
    parser.add_argument(
        "--output-device", type=int, default=None,
        help="sounddevice output device index (default: system default).",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="Print available audio devices and exit.",
    )
    parser.add_argument(
        "--diagnose", action="store_true",
        help="Show live microphone signal levels to help tune config thresholds.",
    )
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    if args.loopback is not None:
        run_loopback(args.loopback)
        return

    if args.diagnose:
        run_diagnose(input_device=args.input_device)
        return

    run_live(input_device=args.input_device, output_device=args.output_device)


if __name__ == "__main__":
    main()
