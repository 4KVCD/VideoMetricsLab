#include "vml/build_info.h"

namespace vml {

std::string build_stamp() {
    return std::string(VML_VERSION) + " (" + build_commit() + ")";
}

}  // namespace vml
