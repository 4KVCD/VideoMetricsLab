// Whether a video's stream describes film grain for its player to add, read
// from the stream itself without decoding it: H.274 film grain
// characteristics SEI messages in H.264, HEVC and VVC, AOM's AFGS1 grain in a
// T.35 SEI message, and AV1's own film grain. FFmpeg's decoders would say so
// too, but a decoder with delay gives no frame from the first packets, and
// FFmpeg's VVC parser rejects VVenC's --fga 3 (H.274 model 1) messages.
#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "vml/cancel.h"
#include "vml/video.h"

namespace vml {

// The kinds of film grain a stream describes, as bits of VideoInfo::film_grain.
enum FilmGrainKind : unsigned {
    film_grain_h274 = 1,          // H.274's frequency filtering model (model 0)
    film_grain_h274_model_1 = 2,  // H.274's auto-regression model
    film_grain_afgs1 = 4,         // AOM's AFGS1, in a T.35 SEI message
    film_grain_av1 = 8,           // AV1's own, in its sequence header
    film_grain_av2 = 16,          // AV2's film grain model (av2_probe.h finds it)
};

// The kinds' names, as run files keep them ("h274", "h274 model 1", "afgs1",
// "av1"), and back; a name it does not know is left out.
std::vector<std::string> film_grain_names(unsigned kinds);
unsigned film_grain_from_names(const std::vector<std::string>& names);

// The film grain an H.264, HEVC or VVC elementary stream (`codec`: "h264",
// "hevc" or "vvc"; Annex B, start codes and all) describes in its prefix SEI
// messages of layer 0: an H.274 film grain characteristics message that does
// not cancel the grain (with its model), and AFGS1. 0 for none.
unsigned film_grain_in_annex_b(std::string_view stream, std::string_view codec);

// Whether an AV1 stream of OBUs (the low-overhead format, as MKV and MP4 hold
// it) has film grain: its first sequence header's film_grain_params_present.
// Nothing when it has no sequence header, or one that does not end where
// its fields do (damaged, or not read as written).
std::optional<unsigned> film_grain_in_av1(std::string_view obus);

// The FFmpeg command that copies the first frames of `info`'s video out of
// its file as the stream itself, for the functions above.
std::vector<std::string> film_grain_scan_command(const std::string& ffmpeg, const VideoInfo& info);

// The film grain `info`'s video describes (FilmGrainKind bits, 0 for none),
// read from its first 3 frames; 0 without reading for a codec with no film
// grain. Nothing when it cannot be read (FFmpeg missing or failing, no
// sequence header). Throws Cancelled once `cancel` says so.
std::optional<unsigned> detect_film_grain(const VideoInfo& info, const std::string& ffmpeg,
                                          const CancelToken& cancel);

}  // namespace vml
