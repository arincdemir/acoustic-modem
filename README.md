# acoustic-modem

A half-duplex acoustic modem written in Python. It transmits text between two computers using nothing but their built-in speakers and microphones, encoding data as audio tones and framing it with a UART-like protocol.

> **Authors:** Arınç Demir, İrem Önen

---

## What we built

We built an end-to-end acoustic link: text typed on one laptop is turned into sound, played out of the speaker, picked up by another laptop's microphone, and decoded back into text. The physical layer is a **six-tone FSK scheme** whose frequencies are all notes of the A blues scale, deliberately spread across several octaves so neighbouring tones sit roughly seven semitones (a ~1.5× frequency ratio) apart and are easy to tell apart. Four of those tones are "data" tones used as a **4-FSK constellation**, each tone carries a *dibit* (two bits), so a full 8-bit byte is sent with just four tones. The remaining two tones are dedicated **START** and **STOP** markers. Because framing has its own frequencies, the receiver can never mistake a framing event for data. Each character therefore travels as six symbols, `START · dibit · dibit · dibit · dibit · STOP`, at a baud rate of 5 symbols/second, and a single preamble tone (we reuse the STOP/idle tone) is sent once per message to wake the receiver up.

On the receiving side, audio is processed in short 50 ms chunks. Instead of a full FFT we use the **Goertzel algorithm** to measure the energy at each of our six target frequencies, and we treat the *fraction* of total chunk energy concentrated at the dominant tone as a signal-to-noise ratio. A four-state machine (`IDLE → ARMED → READING → WAITING`) drives decoding: `IDLE` squelches out noise until it sees a stable preamble, `ARMED` waits for the START tone, `READING` clocks in the four data tones plus a STOP tone by sampling at the centre of each symbol, and `WAITING` allows the next character of the same message to arrive without re-sending a preamble. The whole thing runs against a live microphone stream or, for hardware-free testing, against an in-memory waveform buffer (loopback mode), and a `--diagnose` mode prints live per-frequency SNRs to help tune thresholds.

## Challenges we overcame

The first audible problem was a sharp click, a "pfft", at every symbol boundary. Each tone was starting its sine wave at phase 0 regardless of where the previous tone ended, creating a discontinuity. We fixed it by carrying the oscillator phase across symbol boundaries so the composite waveform stays continuous, and by applying a short amplitude fade-out (release ramp) at the very end of a transmission. A related glitch was the audio backend closing the output stream the instant the last sample was written, which truncated the final symbol; padding the waveform with a brief tail of silence keeps the stream open long enough for the last tone to play out fully.

Reliable detection was the harder, longer fight. A naive "whichever frequency has the higher SNR wins" rule is dangerous: any loud broadband sound (a high-pitched voice, a door slam) can make one tone *relatively* dominant and get decoded as data. We addressed this by requiring the winning tone to clear an absolute SNR threshold, gating everything behind a noise-floor (RMS) check, and choosing widely-spaced blues-scale frequencies so genuine tones stand out as narrowband spikes against broadband noise. False preamble triggers were tamed with a squelch that only arms after the preamble tone has been continuously stable for several chunks. Finally, we made all receiver timing **chunk-count based** rather than wall-clock based, so loopback mode (which processes chunks far faster than real time) behaves identically to live mode, this was essential for getting deterministic, testable round-trips.

## Shortcomings that remain

The link is **half-duplex and slow**, at 5 baud a single character takes about 1.2 seconds, so it is a proof of concept rather than a practical channel. There is **no error correction or checksum**: the START/STOP tones catch gross framing errors, but a corrupted dibit is silently accepted or the frame is quietly dropped, with no retransmission or ACK. It is **7-bit ASCII only**. And it remains **sensitive to its environment**: it expects the two machines to be within about a metre of each other in a quiet room, and noisy conditions may still require hand-tuning the thresholds in `acoustic_modem/config.py` (`NOISE_FLOOR`, `SNR_THRESHOLD`, `PREAMBLE_LOCK_DURATION`, `INTER_CHAR_TIMEOUT`).

---

## Requirements

- Python ≥ 3.10
- A speaker and microphone (built-in is fine)

Create and activate an isolated virtual environment, then install dependencies into it:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All subsequent commands (`python -m acoustic_modem`, `pytest`, etc.) should be run inside this activated environment. To deactivate when you're done:

```bash
deactivate
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

### Diagnose mode (tuning)

Prints live microphone metrics, overall RMS level and the per-frequency SNRs, so you can tune `NOISE_FLOOR` and `SNR_THRESHOLD` before a real two-machine test.

```bash
python -m acoustic_modem --diagnose
```

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
| `--diagnose` | Show live microphone signal levels to help tune thresholds |
| `--input-device N` | sounddevice input device index |
| `--output-device N` | sounddevice output device index |
| `--list-devices` | Print available audio devices and exit |

### Running the tests

```bash
pytest tests/ -v
```
