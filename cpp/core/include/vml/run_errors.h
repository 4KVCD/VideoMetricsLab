// Which errors a run's parts hand on as they are rather than answer
// themselves (another route, a retry, a failure of one metric).
#pragma once

#include <exception>

namespace vml {

// Whether `error` ends the run, or its pass, whatever route scores it: the
// user's Cancel (Cancelled); a test video whose size changed partway
// (VideoSizeChanged), for which run_vmaf and run_perceptual_task make the
// run again with the video scaled first; and a video with film grain the
// app's decoder cannot give the pictures of (FilmGrainRefusal), final
// because every route decodes it with that same decoder (grain_feed.h).
// The parts that try another route or report a metric as failed hand these
// on first, as they are.
bool ends_the_run(const std::exception& error);

// Whether the error `error` holds is an `Error` (or one derived from it), for
// an error a thread kept to hand on (std::current_exception).
template <class Error>
bool holds(const std::exception_ptr& error) {
    try {
        std::rethrow_exception(error);
    } catch (const Error&) {
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace vml
