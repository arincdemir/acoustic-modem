"""
tests/test_framing.py — Unit tests for UART framing.
"""

import pytest
import string
from acoustic_modem import framing, config


class TestCharToBits:
    def test_null_char(self):
        bits = framing.char_to_bits("\x00")
        assert bits == [0] * 8

    def test_all_ones(self):
        bits = framing.char_to_bits("\xff")
        assert bits == [1] * 8

    def test_lsb_first(self):
        # ord('A') = 65 = 0b01000001
        bits = framing.char_to_bits("A")
        assert bits[0] == 1   # LSB
        assert bits[6] == 1   # bit 6
        assert bits[1] == 0
        assert len(bits) == 8

    def test_length(self):
        for ch in string.printable:
            assert len(framing.char_to_bits(ch)) == 8

    def test_invalid_long(self):
        with pytest.raises(ValueError):
            framing.char_to_bits("AB")

    def test_non_ascii(self):
        with pytest.raises(ValueError):
            framing.char_to_bits("€")   # code > 255


class TestBitsToChar:
    def test_round_trip_all_printable(self):
        for ch in string.printable:
            bits = framing.char_to_bits(ch)
            assert framing.bits_to_char(bits) == ch

    def test_wrong_length(self):
        with pytest.raises(ValueError):
            framing.bits_to_char([0, 1, 0])

    def test_null_char(self):
        assert framing.bits_to_char([0] * 8) == "\x00"


class TestFrameChar:
    def test_structure(self):
        frame = framing.frame_char("A")
        assert len(frame) == 10      # 1 start + 8 data + 1 stop
        assert frame[0] == 0         # start bit
        assert frame[-1] == 1        # stop bit

    def test_data_bits_correct(self):
        frame = framing.frame_char("A")
        data = frame[1:9]
        assert data == framing.char_to_bits("A")

    def test_all_printable(self):
        for ch in string.printable:
            frame = framing.frame_char(ch)
            assert frame[0] == 0
            assert frame[-1] == 1
            assert len(frame) == 10


class TestFrameMessage:
    def test_empty_string(self):
        assert framing.frame_message("") == []

    def test_single_char_length(self):
        bits = framing.frame_message("A")
        assert len(bits) == 10

    def test_multi_char_length(self):
        text = "Hello"
        bits = framing.frame_message(text)
        assert len(bits) == len(text) * 10

    def test_back_to_back_frames(self):
        """Each frame starts immediately after the previous stop bit."""
        bits = framing.frame_message("AB")
        # Frame A
        assert bits[0] == 0    # start of A
        assert bits[9] == 1    # stop of A
        # Frame B starts at index 10
        assert bits[10] == 0   # start of B
        assert bits[19] == 1   # stop of B


class TestUnframeChar:
    def test_round_trip(self):
        for ch in string.printable:
            frame = framing.frame_char(ch)
            assert framing.unframe_char(frame) == ch

    def test_bad_start_bit(self):
        frame = framing.frame_char("A")
        frame[0] = 1   # corrupt start
        with pytest.raises(ValueError, match="start"):
            framing.unframe_char(frame)

    def test_bad_stop_bit(self):
        frame = framing.frame_char("A")
        frame[-1] = 0  # corrupt stop
        with pytest.raises(ValueError, match="stop"):
            framing.unframe_char(frame)

    def test_wrong_length(self):
        with pytest.raises(ValueError):
            framing.unframe_char([0, 1])
