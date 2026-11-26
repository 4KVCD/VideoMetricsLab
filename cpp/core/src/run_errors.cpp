#include "vml/run_errors.h"

#include "vml/cancel.h"
#include "vml/film_grain_scoring.h"
#include "vml/memory_fit.h"
#include "vml/size_changes.h"

namespace vml {

bool ends_the_run(const std::exception& error) {
    return dynamic_cast<const Cancelled*>(&error) != nullptr ||
           dynamic_cast<const VideoSizeChanged*>(&error) != nullptr ||
           dynamic_cast<const FilmGrainRefusal*>(&error) != nullptr ||
           dynamic_cast<const MemoryShort*>(&error) != nullptr;
}

}  // namespace vml
