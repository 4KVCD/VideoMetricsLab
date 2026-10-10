"""Reads the film grain of a VVC video, frame by frame, for Video Compare.

A VVC encode can carry film grain as SEI messages ("film grain
characteristics", defined in H.274): the encoder removes the grain to save
bits and describes it instead, and the player is meant to synthesize it
again on each decoded frame. GStreamer's VVC decoder (avdec_h266, built on
FFmpeg 7.1) never does this, so Video Compare synthesizes the grain itself:

  - FilmGrainTracker (this module) reads the access units that GStreamer's
    VVC parser hands to the decoder. It works out which grain message
    applies to each frame, and the frame's picture order count, which seeds
    the grain's random pattern.
  - d3d11_film_grain draws the grain on each decoded frame on the GPU.

Each access unit is numbered as it leaves the parser, and the number goes
with it as a tag that the decoder copies onto the frame it decodes from it
(gstreamer_playback._FilmGrain), so each frame's grain is found by that
number. Timestamps cannot do this on their own: a raw .266 stream has none,
and a decoder that is handed timestamps in decoding order gives them out
again in display order, to other frames. A frame without the tag is matched
by its timestamp.

Which frames a message applies to follows the standard, as FFmpeg's VVC
decoder implements it: a message applies to the frame of its own access
unit, and to the frames decoded after it if its persistence flag is set,
until another message or a cancel replaces it. After a seek, decoding starts
again from a keyframe with no grain remembered. (FFmpeg's own synthesis
never draws a message that does not persist, although it reports it for
that frame: an FFmpeg bug. The standard applies it to its frame, and so
does this app.)
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import NamedTuple

_LOG = logging.getLogger(__name__)

# VVC NAL unit types (H.266 table 5).
RADL_NUT = 2  # a leading picture that can be decoded after a seek
RASL_NUT = 3  # a leading picture that is skipped after a seek
IDR_W_RADL, IDR_N_LP = 7, 8  # instant decoder refresh: a keyframe that resets everything
CRA_NUT = 9  # clean random access: a keyframe that leading pictures may still refer past
GDR_NUT = 10  # gradual decoder refresh
LAST_SLICE_NUT = 11  # types 0-11 are slices (coded picture data)
SPS_NUT, PPS_NUT, PH_NUT = 15, 16, 19  # sequence and picture parameter sets, picture header
AUD_NUT, EOS_NUT = 20, 21  # access unit delimiter, end of sequence
PREFIX_SEI_NUT, SUFFIX_SEI_NUT = 23, 24

# The SEI payload type of a film grain characteristics message (H.274).
FILM_GRAIN_CHARACTERISTICS = 19

# The most frames whose grain is remembered while they wait to come out of
# the decoder. A decoder holds a few dozen frames at most.
_MAX_REMEMBERED_FRAMES = 256

# How far the decoder may move a frame's timestamp. After a seek, the frame
# that overlaps the seek's target comes out stamped with the target's time
# instead of its own (GstVideoDecoder clips it to the playback segment).
# That is less than one frame's duration; one second is a generous limit.
_MAX_TIMESTAMP_SHIFT_NS = 1_000_000_000

# Enough of a slice's start to reach the end of a picture header inside it.
_SLICE_HEADER_BYTES = 64

# The most subpictures a sequence parameter set may give a picture: each
# needs a slice of its own, and no level allows more than 1000 slices in a
# picture (H.266 table A.2). FFmpeg's limit too.
_MAX_SUBPICTURES = 1000


@dataclass(frozen=True)
class FilmGrainH274:
    """One film grain characteristics SEI message (H.274 section 8.5).

    `intervals` holds, for each colour component (Y, Cb, Cr), its intensity
    intervals: (lower bound, upper bound, model values). A component with
    no grain has no intervals. The model values are the grain's scale and
    its horizontal and vertical cutoff frequencies.
    """

    model_id: int
    blending_mode_id: int
    log2_scale_factor: int
    intervals: tuple[tuple[tuple[int, int, tuple[int, ...]], ...], ...]
    persistence: bool


@dataclass(frozen=True)
class FrameGrain:
    """The grain of one decoded frame: the message that applies to it, and
    the seed of its random pattern (the frame's picture order count)."""

    grain: FilmGrainH274
    seed: int


class BitReader:
    """Reads a NAL unit's payload bit by bit, most significant bit first.

    Raises IndexError when reading past the end of the data.
    """

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.position = 0  # in bits

    def read(self, bit_count: int) -> int:
        """Reads an unsigned number of `bit_count` bits (the standard's u(n))."""
        value = 0
        for _ in range(bit_count):
            byte = self.data[self.position >> 3]
            value = (value << 1) | ((byte >> (7 - (self.position & 7))) & 1)
            self.position += 1
        return value

    def read_unsigned_exp_golomb(self) -> int:
        """Reads an unsigned Exp-Golomb code (the standard's ue(v))."""
        leading_zeros = 0
        while self.read(1) == 0:
            leading_zeros += 1
            if leading_zeros > 31:
                raise ValueError("Exp-Golomb code too long")
        return (1 << leading_zeros) - 1 + self.read(leading_zeros)

    def read_signed_exp_golomb(self) -> int:
        """Reads a signed Exp-Golomb code (the standard's se(v))."""
        code = self.read_unsigned_exp_golomb()
        return (code + 1) // 2 if code & 1 else -(code // 2)

    def skip(self, bit_count: int) -> None:
        """Skips bit_count bits."""
        if self.position + bit_count > 8 * len(self.data):
            raise IndexError("reading past the end")
        self.position += bit_count

    def skip_to_byte_boundary(self) -> None:
        """Skips to the start of the next byte."""
        self.position = (self.position + 7) & ~7


def remove_emulation_prevention(payload: bytes) -> bytes:
    """Returns a NAL unit's payload without its emulation prevention bytes.

    An encoder inserts a 0x03 byte after any two zero bytes that would
    otherwise be followed by a byte of 0-3, so that the payload never looks
    like a start code. The payload's syntax is read with them removed.
    """
    if b"\x00\x00\x03" not in payload:
        return payload
    out = bytearray()
    zeros = 0
    for byte in payload:
        if zeros >= 2 and byte == 3:
            zeros = 0
            continue
        out.append(byte)
        zeros = zeros + 1 if byte == 0 else 0
    return bytes(out)


def nal_units(access_unit, length_size: int = 0) -> list[memoryview]:
    """Splits an access unit into its NAL units, as views of its data.

    In Matroska and MP4 ("vvc1"/"vvi1" streams) each NAL unit is prefixed
    by its length in `length_size` bytes. A raw stream (`length_size` 0)
    separates them with start codes (00 00 01) instead.
    """
    data = memoryview(access_unit)
    units = []
    if length_size:
        position, end = 0, len(data)
        while position + length_size <= end:
            size = int.from_bytes(data[position:position + length_size], "big")
            position += length_size
            if size <= 0 or position + size > end:
                break
            units.append(data[position:position + size])
            position += size
        return units
    raw = bytes(data)
    starts = []
    found = raw.find(b"\x00\x00\x01")
    while found >= 0:
        starts.append(found + 3)
        found = raw.find(b"\x00\x00\x01", found + 3)
    for index, start in enumerate(starts):
        end = starts[index + 1] - 3 if index + 1 < len(starts) else len(raw)
        while end > start and raw[end - 1] == 0:  # the zeros before the next start code
            end -= 1
        if end > start:
            units.append(data[start:end])
    return units


def nal_type(unit) -> int:
    """The NAL unit's type (from its two-byte header), or -1 if it has none."""
    return (unit[1] >> 3) & 0x1F if len(unit) >= 2 else -1


def temporal_id(unit) -> int:
    """The NAL unit's temporal layer: 0 for pictures that others depend on most."""
    return (unit[1] & 7) - 1 if len(unit) >= 2 else 0


def nal_payload(unit, limit: int | None = None) -> bytes:
    """The NAL unit's payload after its header, ready to read with BitReader:
    all of it, or its first `limit` bytes."""
    end = len(unit) if limit is None else min(len(unit), 2 + limit)
    return remove_emulation_prevention(bytes(unit[2:end]))


def sei_messages(unit) -> list[tuple[int, bytes]]:
    """The (payload type, payload) of each message in an SEI NAL unit."""
    data = nal_payload(unit)
    messages = []
    position, end = 0, len(data)
    # The payload ends with rbsp_trailing_bits: a single 0x80 byte.
    while position < end and not (position == end - 1 and data[position] == 0x80):
        payload_type, position = _read_sei_number(data, position)
        payload_size, position = _read_sei_number(data, position)
        if payload_type is None or payload_size is None or position + payload_size > end:
            break
        messages.append((payload_type, data[position:position + payload_size]))
        position += payload_size
    return messages


def _read_sei_number(data: bytes, position: int) -> tuple[int | None, int]:
    """Reads an SEI payload type or size: a run of 0xFF bytes (255 each) and
    then a final byte. Returns (None, position) if the data ends first."""
    value = 0
    while position < len(data) and data[position] == 0xFF:
        value += 255
        position += 1
    if position >= len(data):
        return None, position
    return value + data[position], position + 1


def parse_film_grain(payload: bytes) -> FilmGrainH274 | None:
    """Reads a film grain characteristics message (H.274 section 8.5).

    Returns None for a message that cancels the grain. Raises ValueError for
    a malformed message, with the limits FFmpeg's parser checks.
    """
    bits = BitReader(payload)
    try:
        if bits.read(1):  # fg_characteristics_cancel_flag
            return None
        model_id = bits.read(2)
        if model_id > 1:
            raise ValueError(f"film grain model {model_id}")
        if bits.read(1):  # fg_separate_colour_description_present_flag
            bits.skip(3 + 3 + 1 + 8 + 8 + 8)  # bit depths, range, primaries, transfer, matrix
        blending_mode_id = bits.read(2)
        if blending_mode_id > 1:
            raise ValueError(f"film grain blending mode {blending_mode_id}")
        log2_scale_factor = bits.read(4)
        component_present = [bits.read(1) for _ in range(3)]
        intervals = []
        for present in component_present:
            if not present:
                intervals.append(())
                continue
            interval_count = bits.read(8) + 1
            value_count = bits.read(3) + 1
            if value_count > 6:
                raise ValueError(f"{value_count} film grain model values")
            component = []
            for _ in range(interval_count):
                lower, upper = bits.read(8), bits.read(8)
                values = tuple(bits.read_signed_exp_golomb() for _ in range(value_count))
                component.append((lower, upper, values))
            intervals.append(tuple(component))
        persistence = bool(bits.read(1))
    except IndexError as error:
        raise ValueError("film grain characteristics message cut short") from error
    return FilmGrainH274(model_id, blending_mode_id, log2_scale_factor, tuple(intervals), persistence)


# -- Picture order counts ---------------------------------------------------
#
# A frame's grain is seeded by its picture order count (POC): the frame's
# position in display order within its coded video sequence. Each picture
# header gives the low bits of its POC; the high bits follow from the
# pictures before it (H.266 section 8.3.1). Reading the header needs a few
# fields of the sequence and picture parameter sets.


@dataclass(frozen=True)
class SequenceParameterSet:
    """The fields of a sequence parameter set that a picture header needs."""

    log2_max_poc_lsb: int  # how many bits of its POC each picture header gives
    poc_msb_cycle_length: int  # bits of an explicit POC high part; 0 if never given
    extra_picture_header_bits: int  # extension bits in each picture header


class PictureHeader(NamedTuple):
    """What a picture header gives: the picture order count's parts, its
    parameter set, and whether other pictures refer to it."""

    poc_lsb: int  # the low bits of the picture order count
    poc_msb_cycle: int | None  # the high part, where the header gives it
    sps: SequenceParameterSet
    non_reference: bool = False  # no other picture refers to this one (ph_non_ref_pic_flag)


def _skip_general_constraints_info(bits: BitReader) -> None:
    """Skips general_constraints_info() (H.266 section 7.3.3.2)."""
    if bits.read(1):  # gci_present_flag
        bits.skip(71)  # the constraint flags of H.266 section 7.3.3.2
        bits.skip(bits.read(8))  # gci_num_additional_bits, and those bits
    bits.skip_to_byte_boundary()


def _skip_profile_tier_level(bits: BitReader, max_sublayers_minus1: int) -> None:
    """Skips profile_tier_level() (H.266 section 7.3.3.1)."""
    bits.skip(7 + 1 + 8 + 1 + 1)  # profile, tier, level, frame-only and multilayer flags
    _skip_general_constraints_info(bits)
    sublayer_level_present = [bits.read(1) for _ in range(max_sublayers_minus1)]
    bits.skip_to_byte_boundary()
    bits.skip(8 * sum(sublayer_level_present))
    bits.skip(32 * bits.read(8))  # the sub-profiles


def _bits_for(count: int) -> int:
    """The bits needed to write numbers below `count`: Ceil(Log2(count))."""
    return max(0, (count - 1).bit_length())


def parse_sps(payload: bytes) -> tuple[int, SequenceParameterSet]:
    """Reads a sequence parameter set (H.266 section 7.3.2.4) as far as its
    picture header extension bits. Returns (its id, what a picture header needs)."""
    bits = BitReader(payload)
    sps_id = bits.read(4)
    bits.skip(4)  # sps_video_parameter_set_id
    max_sublayers_minus1 = bits.read(3)
    bits.skip(2)  # sps_chroma_format_idc
    ctu_size = 1 << (bits.read(2) + 5)
    if bits.read(1):  # sps_ptl_dpb_hrd_params_present_flag
        _skip_profile_tier_level(bits, max_sublayers_minus1)
    bits.skip(1)  # sps_gdr_enabled_flag
    if bits.read(1):  # sps_ref_pic_resampling_enabled_flag
        bits.skip(1)  # sps_res_change_in_clvs_allowed_flag
    width = bits.read_unsigned_exp_golomb()
    height = bits.read_unsigned_exp_golomb()
    if bits.read(1):  # sps_conformance_window_flag
        for _ in range(4):
            bits.read_unsigned_exp_golomb()
    if bits.read(1):  # sps_subpic_info_present_flag
        _skip_subpicture_info(bits, width, height, ctu_size)
    bits.read_unsigned_exp_golomb()  # sps_bitdepth_minus8
    bits.skip(2)  # entropy coding sync and entry point offsets flags
    log2_max_poc_lsb = bits.read(4) + 4
    poc_msb_cycle_length = bits.read_unsigned_exp_golomb() + 1 if bits.read(1) else 0
    extension_bytes = bits.read(2)
    extra_bits = sum(bits.read(1) for _ in range(8 * extension_bytes))
    return sps_id, SequenceParameterSet(log2_max_poc_lsb, poc_msb_cycle_length, extra_bits)


def _skip_subpicture_info(bits: BitReader, width: int, height: int, ctu_size: int) -> None:
    """Skips the subpicture layout in a sequence parameter set."""
    count_minus1 = bits.read_unsigned_exp_golomb()
    if count_minus1 >= _MAX_SUBPICTURES:
        # A damaged parameter set. Read on, a count this large would keep
        # the parser's thread busy for up to a minute.
        raise ValueError("more subpictures than any level allows")
    independent, same_size = 1, 0
    if count_minus1 > 0:
        independent, same_size = bits.read(1), bits.read(1)
    x_bits = _bits_for((width + ctu_size - 1) // ctu_size)
    y_bits = _bits_for((height + ctu_size - 1) // ctu_size)
    for index in range(count_minus1 + 1 if count_minus1 > 0 else 0):
        if not same_size or index == 0:
            if index > 0 and width > ctu_size:
                bits.skip(x_bits)  # its top-left corner
            if index > 0 and height > ctu_size:
                bits.skip(y_bits)
            if index < count_minus1 and width > ctu_size:
                bits.skip(x_bits)  # its size
            if index < count_minus1 and height > ctu_size:
                bits.skip(y_bits)
        if not independent:
            bits.skip(2)
    id_bits = bits.read_unsigned_exp_golomb() + 1
    if bits.read(1) and bits.read(1):  # ids signalled explicitly, and in the SPS
        bits.skip(id_bits * (count_minus1 + 1))


def parse_pps(payload: bytes) -> tuple[int, int]:
    """Reads a picture parameter set's id and its sequence parameter set's id."""
    bits = BitReader(payload)
    return bits.read(6), bits.read(4)


def read_picture_header(bits: BitReader, pps_to_sps: dict[int, int],
                        sequence_parameter_sets: dict[int, SequenceParameterSet]) -> PictureHeader:
    """Reads a picture header (H.266 section 7.3.2.8) as far as its picture
    order count. Raises KeyError if it refers to a parameter set not seen."""
    gdr_or_irap = bits.read(1)
    non_reference = bits.read(1)
    gdr = bits.read(1) if gdr_or_irap else 0
    if bits.read(1):  # ph_inter_slice_allowed_flag
        bits.skip(1)  # ph_intra_slice_allowed_flag
    pps_id = bits.read_unsigned_exp_golomb()
    sps = sequence_parameter_sets[pps_to_sps[pps_id]]
    poc_lsb = bits.read(sps.log2_max_poc_lsb)
    if gdr:
        bits.read_unsigned_exp_golomb()  # ph_recovery_poc_cnt
    bits.skip(sps.extra_picture_header_bits)
    poc_msb_cycle = None
    if sps.poc_msb_cycle_length and bits.read(1):  # ph_poc_msb_cycle_present_flag
        poc_msb_cycle = bits.read(sps.poc_msb_cycle_length)
    return PictureHeader(poc_lsb, poc_msb_cycle, sps, bool(non_reference))


def decoder_config_units(codec_data: bytes) -> tuple[int, list[bytes]]:
    """Reads a VVC decoder configuration record (ISO/IEC 14496-15 section
    11.2.4.2): Matroska's and MP4's codec private data. A stream's parameter
    sets can be kept here instead of in its access units.

    Returns (the size of the NAL units' length prefixes, the NAL units it holds).
    """
    bits = BitReader(codec_data)
    bits.skip(5)
    length_size = bits.read(2) + 1
    if bits.read(1):  # ptl_present_flag
        bits.skip(9)  # ols_idx
        sublayers = bits.read(3)
        bits.skip(2 + 2 + 3 + 5)  # frame rate, chroma format, bit depth, reserved
        # VvcPTLRecord: the profile, tier and level.
        bits.skip(2)
        constraint_bytes = bits.read(6)
        bits.skip(7 + 1 + 8 + 1 + 1 + 8 * constraint_bytes - 2)
        sublayer_level_present = [bits.read(1) for _ in range(max(0, sublayers - 1))]
        if sublayers > 1:
            bits.skip(9 - sublayers)
        bits.skip(8 * sum(sublayer_level_present))
        bits.skip(32 * bits.read(8))  # the sub-profiles
        bits.skip(16 + 16 + 16)  # maximum width and height, average frame rate
    units = []
    for _ in range(bits.read(8)):  # arrays of NAL units, one type each
        bits.skip(3)
        unit_type = bits.read(5)
        count = 1 if unit_type in (12, 13) else bits.read(16)  # OPI and DCI arrays hold one, uncounted
        for _ in range(count):
            size = bits.read(16)
            start = bits.position >> 3
            units.append(codec_data[start:start + size])
            bits.skip(8 * size)
    return length_size, units


class FilmGrainTracker:
    """Keeps track of the film grain of one VVC stream's frames.

    GStreamer's parser thread hands it the access units in decoding order
    (add_access_unit), and it numbers them; the pipeline's video thread asks
    it for each decoded frame's grain by the number of the frame's access
    unit, or by the frame's timestamp (take_frame_grain).
    """

    def __init__(self) -> None:
        self.length_size = 0  # the NAL units' length prefix size; 0 for start codes
        # The grain message that persists to the next frames, if any.
        self._persisting: FilmGrainH274 | None = None
        # How many access units have been read: the last one's number.
        self._access_units = 0
        # Each frame's timestamp and grain (None for a frame without), by
        # its access unit's number, until the decoder outputs the frame.
        self._frames: OrderedDict[int, tuple[int | None, FrameGrain | None]] = OrderedDict()
        self._lock = threading.Lock()
        self._sequence_parameter_sets: dict[int, SequenceParameterSet] = {}
        self._pps_to_sps: dict[int, int] = {}
        # The picture order count of the last picture that the next one's is
        # worked out from: the previous one in temporal layer 0 that is not a
        # leading picture, and that other pictures may refer to.
        self._previous_poc: int | None = None
        # Whether the next keyframe starts a new sequence: the first one, and
        # the first after an end of sequence or a seek.
        self._sequence_starts = True
        self._warned_about_poc = False

    def read_decoder_config(self, codec_data: bytes) -> None:
        """Reads the stream's decoder configuration record (the caps'
        codec_data): its NAL length size and any parameter sets in it."""
        try:
            self.length_size, units = decoder_config_units(codec_data)
        except (IndexError, ValueError):
            return
        for unit in units:
            self._remember_parameter_set(unit)

    def _remember_parameter_set(self, unit) -> None:
        """Keeps a sequence or picture parameter set, which the picture headers
        that follow refer to."""
        try:
            if nal_type(unit) == SPS_NUT:
                sps_id, sps = parse_sps(nal_payload(unit))
                self._sequence_parameter_sets[sps_id] = sps
            elif nal_type(unit) == PPS_NUT:
                pps_id, sps_id = parse_pps(nal_payload(unit))
                self._pps_to_sps[pps_id] = sps_id
        except (IndexError, ValueError):
            pass  # a damaged parameter set: its pictures' POCs cannot be read

    def add_access_unit(self, data, pts: int | None) -> int:
        """Reads one access unit (in decoding order) and remembers the grain of
        the frame it decodes to, and that frame's timestamp `pts` (None if
        it has none). Returns the access unit's number, by which
        take_frame_grain finds the frame's grain.
        """
        self._access_units += 1
        grain = self._persisting
        header: PictureHeader | None = None
        slice_type = slice_temporal_id = None
        end_of_sequence = False
        for unit in nal_units(data, self.length_size):
            unit_type = nal_type(unit)
            if unit_type in (SPS_NUT, PPS_NUT):
                self._remember_parameter_set(unit)
            elif unit_type == PH_NUT and header is None:
                header = self._read_header(BitReader(nal_payload(unit)))
            elif 0 <= unit_type <= LAST_SLICE_NUT and slice_type is None:
                slice_type, slice_temporal_id = unit_type, temporal_id(unit)
                if header is None:
                    header = self._header_in_slice(unit)
            elif unit_type in (PREFIX_SEI_NUT, SUFFIX_SEI_NUT):
                for payload_type, payload in sei_messages(unit):
                    if payload_type == FILM_GRAIN_CHARACTERISTICS:
                        try:
                            grain = parse_film_grain(payload)
                        except ValueError:
                            grain = None  # FFmpeg drops a malformed message, and the grain with it
            elif unit_type == EOS_NUT:
                end_of_sequence = True
        seed = self._picture_order_count(header, slice_type, slice_temporal_id, pts)
        if end_of_sequence:
            self._sequence_starts = True
        with self._lock:
            self._frames[self._access_units] = (pts, FrameGrain(grain, seed) if grain is not None else None)
            # Frames the decoder drops (those before a seek's target) are
            # never asked for: the oldest records go once there are too many.
            while len(self._frames) > _MAX_REMEMBERED_FRAMES:
                self._frames.popitem(last=False)
        self._persisting = grain if grain is not None and grain.persistence else None
        return self._access_units

    def _read_header(self, bits: BitReader) -> PictureHeader | None:
        """Reads a picture header, or returns None if it cannot be read."""
        try:
            return read_picture_header(bits, self._pps_to_sps, self._sequence_parameter_sets)
        except (IndexError, KeyError, ValueError):
            return None

    def _header_in_slice(self, unit) -> PictureHeader | None:
        """The picture header inside a slice's header, if the picture has it there."""
        bits = BitReader(nal_payload(unit, _SLICE_HEADER_BYTES))
        try:
            in_slice = bits.read(1)  # sh_picture_header_in_slice_header_flag
        except IndexError:
            return None
        return self._read_header(bits) if in_slice else None

    def _picture_order_count(self, header: PictureHeader | None, slice_type: int | None,
                             slice_temporal_id: int | None, pts: int | None) -> int:
        """Works out a picture's order count (H.266 section 8.3.1).

        Where it cannot (a damaged or unreadable header), the frame's time,
        or its access unit's number, stands in for it: the frame still gets
        grain, if not the pattern a decoder would give it.
        """
        if header is None or slice_type is None:
            if slice_type is not None and not self._warned_about_poc:
                self._warned_about_poc = True
                _LOG.info("Film grain: a picture order count could not be read; frame times seed the grain")
            return pts // 1000 if pts is not None else self._access_units
        max_lsb = 1 << header.sps.log2_max_poc_lsb
        keyframe = slice_type in (IDR_W_RADL, IDR_N_LP, CRA_NUT, GDR_NUT)
        if header.poc_msb_cycle is not None:
            msb = header.poc_msb_cycle * max_lsb
        elif (slice_type in (IDR_W_RADL, IDR_N_LP) or (keyframe and self._sequence_starts)
              or self._previous_poc is None):
            msb = 0
        else:
            # The high part follows the previous picture's, stepping up or
            # down when the low part wraps around.
            previous_lsb = self._previous_poc & (max_lsb - 1)
            previous_msb = self._previous_poc - previous_lsb
            if header.poc_lsb < previous_lsb and previous_lsb - header.poc_lsb >= max_lsb // 2:
                msb = previous_msb + max_lsb
            elif header.poc_lsb > previous_lsb and header.poc_lsb - previous_lsb > max_lsb // 2:
                msb = previous_msb - max_lsb
            else:
                msb = previous_msb
        poc = msb + header.poc_lsb
        if keyframe:
            self._sequence_starts = False
        if slice_temporal_id == 0 and slice_type not in (RADL_NUT, RASL_NUT) and not header.non_reference:
            self._previous_poc = poc
        return poc

    def take_frame_grain(self, access_unit: int | None, pts: int | None = None) -> FrameGrain | None:
        """Returns the grain of a decoded frame, or None for a frame without
        grain. The frame is found by the number of the access unit it was
        decoded from (add_access_unit) or, without that, by its timestamp
        `pts`.
        """
        with self._lock:
            if access_unit is not None:
                record = self._frames.pop(access_unit, None)
                return record[1] if record is not None else None
            return self._take_by_timestamp(pts)

    def _take_by_timestamp(self, pts: int | None) -> FrameGrain | None:
        """The grain of the frame at timestamp `pts`.

        The decoder outputs frames in display order, so every frame before
        this one has already been output, or dropped (a seek drops the frames
        before its target). Their records are forgotten here.
        """
        if pts is None:
            return None
        numbers = {time: number for number, (time, _grain) in self._frames.items() if time is not None}
        key = pts if pts in numbers else self._original_timestamp(pts, numbers)
        if key is None:
            return None
        grain = self._frames[numbers[key]][1]
        for time, number in numbers.items():
            if time <= key:
                del self._frames[number]
        return grain

    @staticmethod
    def _original_timestamp(pts: int, times) -> int | None:
        """The timestamp a frame had before the decoder moved it to `pts`
        after a seek: the latest of `times` before `pts`, if close enough."""
        earlier = [time for time in times if time < pts]
        if earlier and pts - max(earlier) < _MAX_TIMESTAMP_SHIFT_NS:
            return max(earlier)
        return None

    def flush(self) -> None:
        """Forgets everything about the frames before a seek: decoding starts
        again from a keyframe."""
        with self._lock:
            self._frames.clear()
        self._persisting = None
        self._previous_poc = None
        self._sequence_starts = True
