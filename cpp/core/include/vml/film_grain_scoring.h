// How a video's film grain is scored: with the grain its stream tells the
// player to add (what viewers see) or without it (the coded pictures only),
// as the user chooses for each video (VideoInfo::with_film_grain). One rule,
// effective_film_grain, for the decoders, the routes, the saved scores and
// the labels.
#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "grain_decoding.h"
#include "vml/video.h"

namespace vml {

enum class GrainScoring {
    none,     // the stream describes no film grain, or it is not known (not read)
    with,     // its grain is added, as `kinds` says
    without,  // its grain is left out
};

struct FilmGrainScoring {
    GrainScoring scoring = GrainScoring::none;
    unsigned kinds = 0;  // FilmGrainKind bits (film_grain_detect.h)
    // Why the video is scored without its grain whatever is chosen: grain
    // nothing here adds (H.274's outside VVC, where only VVdeC adds it;
    // AFGS1 in H.264). Empty otherwise.
    std::string why_without;
};

// The film grain `info`'s video is scored with: its stream's grain
// (VideoInfo::film_grain) and the choice (with unless chosen without).
FilmGrainScoring effective_film_grain(const VideoInfo& info);

// Whether `info`'s video can be scored with its film grain or without, as
// the user chooses: grain read, and none that nothing here adds (why_without).
bool film_grain_choosable(const VideoInfo& info);

// The choice Settings::film_grain_default makes for a video with none of its
// own: with its grain (true), without (false), or nothing to ask ("" and any
// other text).
std::optional<bool> film_grain_default_choice(std::string_view setting);

// Whether `grain` is H.274's, added: only the app's software decoder adds it
// as it should (VVC's, by VVdeC, at any bit depth). FFmpeg's own decoder adds
// it at 8 bits only, rounded down, and GPU decoders add none.
bool only_the_app_adds(const FilmGrainScoring& grain);

// `grain` as a decoder is told it (grain_decoding.h's `film_grain`): 1 added,
// 0 left out, -1 as the decoder adds it by itself (none, or not known).
int decoder_film_grain(const FilmGrainScoring& grain);

// How the app's own decoders decode `info` with its grain as it is scored:
// grain_decoding.h's rule, which the scoring decoder takes from its plan.
GrainDecoding video_grain_decoding(const VideoInfo& info);

// The film grain options of an FFmpeg command that shows `info`'s pictures
// (Frame Compare's stills, Video Compare's FFmpeg playback): none where
// FFmpeg adds the grain as the video is scored, else -export_side_data
// film_grain, which leaves it out. Out too where only the app adds it as it
// should (only_the_app_adds): FFmpeg would add other grain, or none.
std::vector<std::string> picture_grain_options(const VideoInfo& info);

// What a saved score records of `info`'s film grain as it is scored
// (score_identity.h's film_grain_*): nothing where every score made before
// the choice existed scored it so too, so that those still answer: no
// grain, grain not known, grain left out that no decoder added then either
// (H.274's deeper than 8 bits or of model 1, AFGS1 in H.264), and on Windows
// AV1's added. Any other was scored one way or another by route, maker and
// FFmpeg version.
std::optional<std::string> film_grain_identity(const VideoInfo& info);

// Gives `into` the film grain of `from`, as read and as chosen: the videos of
// a result saved before either was known take those of the run or row whose
// request found it, so that its scores are saved and shown as they were made.
void take_film_grain(VideoInfo& into, const VideoInfo& from);

// A video to be scored with its film grain where its pictures would come
// from a decoder that adds other grain or none (FFmpeg's command line for
// H.274's): the metric is not calculated, rather than calculated without
// the grain the user asked for. Not a failure of the route: fallbacks pass
// it on, as they do Cancelled.
class FilmGrainRefusal : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

}  // namespace vml
