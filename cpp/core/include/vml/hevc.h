// HEVC's NAL units (H.265 7.3.1.2 and table 7-1): the types the app reads a
// stream by, and a unit's type and layer from the two bytes of its header.
#pragma once

namespace vml::hevc_unit {

constexpr int radl_n = 6, radl_r = 7;  // leading pictures that refer only to their random access point
constexpr int rasl_n = 8, rasl_r = 9;  // leading pictures that refer to pictures before it
constexpr int last_picture = 31;       // 0-31 hold pictures' slices
constexpr int first_random_access = 16;  // 16-23: BLA (16-18), IDR (19-20) and CRA (21) pictures
constexpr int first_idr = 19;
constexpr int cra = 21;
constexpr int last_random_access = 23;
constexpr int first_parameter_set = 32;  // 32-34: VPS, SPS and PPS
constexpr int last_parameter_set = 34;

constexpr int type_of(int first_byte) { return (first_byte >> 1) & 0x3F; }
constexpr int layer_of(int first_byte, int second_byte) { return ((first_byte & 1) << 5) | (second_byte >> 3); }

}  // namespace vml::hevc_unit
