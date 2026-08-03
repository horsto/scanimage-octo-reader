"""Parsing of the MATLAB-flavoured values in ScanImage tags."""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest

from scanimage_octo_reader.parsers import (
    parse_aux_trigger,
    parse_bool,
    parse_epoch,
    parse_i2c_data,
    parse_matlab_scalar,
    parse_numeric_array,
)


class TestNumericArrays:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # ScanImage 2022.x writes space-separated values with a trailing space.
            ("[3.439188320 ]", [3.43918832]),
            # The docs' example, two triggers on one line.
            ("[0.313856210 0.331578125]", [0.31385621, 0.331578125]),
            # Older ScanImage versions use commas; `dj-imaging` assumed this form.
            ("[0.313856210, 0.331578125]", [0.31385621, 0.331578125]),
            ("[]", []),
            ("", []),
            ("[1e-3 2E+2]", [0.001, 200.0]),
            ("[-1.5 2]", [-1.5, 2.0]),
        ],
    )
    def test_separator_conventions(self, text, expected):
        assert parse_numeric_array(text) == pytest.approx(np.asarray(expected))

    def test_result_is_float64(self):
        assert parse_numeric_array("[1 2]").dtype == np.float64

    def test_nonfinite_values(self):
        values = parse_numeric_array("[Inf -Inf NaN 1]")
        assert values[0] == math.inf
        assert values[1] == -math.inf
        assert math.isnan(values[2])
        assert values[3] == 1.0

    def test_junk_tokens_are_skipped_not_fatal(self):
        # One unreadable token should cost that value only.
        assert parse_numeric_array("[1 oops 3]") == pytest.approx(np.array([1.0, 3.0]))

    def test_aux_trigger_delegates_to_numeric_array(self):
        assert parse_aux_trigger("[1.5 2.5]") == pytest.approx(np.array([1.5, 2.5]))


class TestBooleans:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("true", True), ("false", False), ("1", True), ("0", False), (True, True), (0, False)],
    )
    def test_both_spellings(self, value, expected):
        """ScanImage writes booleans as 'true'/'false' *and* as '1'/'0'."""
        assert parse_bool(value) is expected

    def test_unknown_value_raises(self):
        with pytest.raises(ValueError, match="ScanImage boolean"):
            parse_bool("perhaps")


class TestEpoch:
    def test_parses_matlab_clock_vector(self):
        assert parse_epoch("[2026  8  3 12 35 19.847]") == datetime(2026, 8, 3, 12, 35, 19, 847000)

    def test_carries_a_rounded_up_fractional_second(self):
        # 59.9999995 s must not become microsecond=1000000, which datetime rejects.
        assert parse_epoch("[2026 1 1 0 0 59.9999996]") == datetime(2026, 1, 1, 0, 1, 0, 0)

    @pytest.mark.parametrize("text", ["[]", "[2026 8]", "[2026 13 40 99 99 99]", "nonsense"])
    def test_unusable_values_return_none(self, text):
        """The epoch is informational, so a bad value must not raise."""
        assert parse_epoch(text) is None


class TestI2C:
    def test_empty(self):
        assert parse_i2c_data("{}") == []
        assert parse_i2c_data("") == []

    def test_char_flavour(self):
        """`I2CStoreAsChar = true`: payloads are strings."""
        packets = parse_i2c_data(
            "{{600.251504290, 'treadmill_9'} {600.294405930, 'treadmill_10'} }"
        )
        assert [p.timestamp for p in packets] == pytest.approx([600.25150429, 600.29440593])
        assert [p.text for p in packets] == ["treadmill_9", "treadmill_10"]
        assert all(packet.data is None for packet in packets)

    def test_byte_flavour(self):
        """`I2CStoreAsChar = false`: payloads are byte arrays."""
        packets = parse_i2c_data("{{600.25, [1 2 3]} {600.29, [4 5 6 7]} }")
        assert len(packets) == 2
        assert packets[0].data.tolist() == [1, 2, 3]
        assert packets[0].data.dtype == np.uint8
        assert packets[1].payload_length == 4

    def test_single_packet_is_not_confused_with_a_pair(self):
        """`matlabstr2py` flattens this case; the packet count must survive."""
        packets = parse_i2c_data("{{1.5, [255]}}")
        assert len(packets) == 1
        assert packets[0].timestamp == 1.5
        assert packets[0].data.tolist() == [255]

    def test_negative_timestamps_are_kept_but_flagged(self):
        """Dropping them silently, as the old code did, loses information."""
        packets = parse_i2c_data("{{-1.0, 'a_1'} {2.0, 'a_2'} }")
        assert len(packets) == 2
        assert packets[0].is_valid_timestamp is False
        assert packets[1].is_valid_timestamp is True

    def test_raw_source_is_retained(self):
        packets = parse_i2c_data("{{1.5, 'x_1'}}")
        assert packets[0].raw == "{1.5, 'x_1'}"

    def test_payload_length_of_text(self):
        assert parse_i2c_data("{{1.0, 'abcd'}}")[0].payload_length == 4


class TestMatlabScalar:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("42", 42),
            ("-3.5", -3.5),
            ("true", True),
            ("false", False),
            ("'hello'", "hello"),
            ("[1 2 3]", [1.0, 2.0, 3.0]),
            ("", ""),
            ("scanimage.types.BeamAdjustTypes.None", "scanimage.types.BeamAdjustTypes.None"),
        ],
    )
    def test_typing(self, text, expected):
        assert parse_matlab_scalar(text) == expected

    def test_inf(self):
        assert parse_matlab_scalar("Inf") == math.inf
