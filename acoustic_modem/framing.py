"""
framing.py — UART-style framing: characters ↔ bit sequences.

Frame structure per character:
  [start=0] [b0 b1 b2 b3 b4 b5 b6 b7] [stop=1]
   1 bit      8 data bits (LSB first)    1 bit

The preamble is a continuous audio tone, not a bit sequence,
so it is handled by dsp.py / tx.py — not here.
"""

from acoustic_modem import config


def char_to_bits(char: str) -> list[int]:
    """Convert a single ASCII character to 8 bits, LSB first."""
    if len(char) != 1:
        raise ValueError(f"Expected a single character, got: {char!r}")
    code = ord(char)
    if code > 255:
        raise ValueError(f"Character {char!r} is not ASCII (code {code})")
    return [(code >> i) & 1 for i in range(config.DATA_BITS)]


def bits_to_char(bits: list[int]) -> str:
    """Convert 8 bits (LSB first) back to an ASCII character."""
    if len(bits) != config.DATA_BITS:
        raise ValueError(f"Expected {config.DATA_BITS} bits, got {len(bits)}")
    code = sum(bit << i for i, bit in enumerate(bits))
    return chr(code)


def frame_char(char: str) -> list[int]:
    """
    Wrap a character in a UART frame.
    Returns: [start=0] + [8 data bits LSB-first] + [stop=1]
    """
    return [0] + char_to_bits(char) + [1]


def frame_message(text: str) -> list[int]:
    """
    Convert a full text string into a flat bit sequence.
    The preamble is NOT included here — it is prepended as audio by tx.py.
    Characters are framed back-to-back: no gap between stop of char N and
    start of char N+1.
    """
    bits: list[int] = []
    for char in text:
        bits.extend(frame_char(char))
    return bits


def unframe_char(frame_bits: list[int]) -> str:
    """
    Decode a 10-bit UART frame [start + 8 data + stop] into a character.
    Raises ValueError if the start or stop bits are wrong.
    """
    if len(frame_bits) != 1 + config.DATA_BITS + 1:
        raise ValueError(f"Expected 10 bits, got {len(frame_bits)}")
    start, *data, stop = frame_bits
    if start != 0:
        raise ValueError(f"Bad start bit: {start}")
    if stop != 1:
        raise ValueError(f"Bad stop bit: {stop}")
    return bits_to_char(data)
