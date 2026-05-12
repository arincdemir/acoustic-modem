# acoustic-modem

A half-duplex acoustic modem written in Python. It transmits text between two computers using only their built-in speakers and microphones, encoding data as audio tones via **Frequency Shift Keying (FSK)** and structuring frames with a **UART-like protocol**.

> **Authors:** Arınç Demir, İrem Önen

---

## How it works

### Physical layer — FSK modulation

Each bit is represented by a sine wave at one of two frequencies:

| Logic | Frequency |
|-------|-----------|
| `0`   | 2000 Hz   |
| `1`   | 3000 Hz   |

The **baud rate is 10 bits/s** — each bit lasts 100 ms. This deliberately slow rate lets room echoes fully decay before the next bit is sampled, avoiding inter-symbol interference.

### Data link layer — UART framing

Each character is sent as a 10-bit UART frame:

```
[Preamble: 200 ms @ 3000 Hz] [Start: 0] [b0 b1 b2 b3 b4 b5 b6 b7] [Stop: 1]
```

- **Preamble** is sent **once per message** (not per character) to wake up the receiver.
- Characters within a message are sent back-to-back — no extra preamble between them.
- One character = 10 bits = **1 second** at 10 baud.

### Receiver state machine

```
IDLE → (preamble detected) → ARMED → (start bit) → READING → (stop bit) → WAITING
                                                                               │
                                                          ┌────────────────────┘
                                                          ↓
                                                    (next start bit, no new preamble)
                                                       READING → ...
                                                          │
                                                   (silence timeout)
                                                       IDLE
```

The **squelch** in IDLE requires three conditions before arming:
1. Signal above the noise floor (RMS threshold).
2. 3000 Hz energy constitutes ≥ 40% of total spectral energy (narrowband check).
3. The tone is stable for at least 6 consecutive 20 ms analysis windows (~120 ms).

---

## Requirements

- Python ≥ 3.10
- A speaker and microphone (built-in is fine)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Loopback test (no hardware needed)

Transmits a string, feeds the audio buffer directly into the receiver in-process, and prints the decoded result. Good for verifying the modem works on your machine.

```bash
python -m acoustic_modem --loopback "Hello World"
```

Expected output:
```
[LOOPBACK] Transmitting: 'Hello World'
[LOOPBACK] Received:     'Hello World'
[LOOPBACK] ✓ Round-trip successful.
```

### Live mode (two machines)

Run this on **both** machines. One types, the other receives. Because it's half-duplex, only one side transmits at a time.

```bash
python -m acoustic_modem
```

- Type a message and press **Enter** to transmit.
- Received characters appear on screen in real time as they are decoded.
- Press **Ctrl-C** to quit.

Place the two laptops within ~1–2 metres of each other. Ensure neither machine is playing audio and the room is reasonably quiet.

### List audio devices

```bash
python -m acoustic_modem --list-devices
```

Use the printed device index with `--input-device` / `--output-device` if you need to select a specific microphone or speaker:

```bash
python -m acoustic_modem --input-device 1 --output-device 3
```

### All CLI flags

| Flag | Description |
|------|-------------|
| `--loopback TEXT` | Run in-process loopback test with `TEXT` |
| `--input-device N` | sounddevice input device index |
| `--output-device N` | sounddevice output device index |
| `--list-devices` | Print available audio devices and exit |

---

## Running the tests

```bash
pytest tests/ -v
```

There are **44 tests** across three files:

| File | What it tests |
|------|---------------|
| `tests/test_framing.py` | UART bit framing — round-trips for all printable ASCII, start/stop bit correctness, error detection |
| `tests/test_dsp.py` | Tone generation, Goertzel energy accuracy, bit detection with and without noise, noise floor |
| `tests/test_loopback.py` | Full Tx → Rx round-trips in-process (no audio hardware) including tests with added Gaussian noise |

---

## Project structure

```
acoustic-modem/
├── requirements.txt
├── acoustic_modem/
│   ├── config.py      # All constants: frequencies, baud rate, thresholds
│   ├── framing.py     # UART framing: char ↔ bits with start/stop bits
│   ├── dsp.py         # Tone generation, Goertzel energy, bit detection
│   ├── tx.py          # Transmitter: text → waveform → speakers
│   ├── rx.py          # Receiver: mic → state machine → characters
│   └── main.py        # Entry point and CLI
└── tests/
    ├── test_framing.py
    ├── test_dsp.py
    └── test_loopback.py
```

## Tuning

All parameters are in `acoustic_modem/config.py`. If you experience false triggers or missed messages in a noisy environment, adjust:

| Constant | Effect |
|---|---|
| `NOISE_FLOOR` | Raise to ignore more background noise |
| `SNR_THRESHOLD` | Raise (towards 1.0) to require a purer tone before arming |
| `PREAMBLE_LOCK_DURATION` | Raise to require a longer stable preamble before arming |
| `INTER_CHAR_TIMEOUT` | Raise if characters are being dropped mid-message |