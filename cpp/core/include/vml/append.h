// One vector's items after another's.
#pragma once

#include <iterator>

namespace vml {

// `from`'s items added at the end of `to`.
template <class Items, class Range>
void append(Items& to, const Range& from) {
    to.insert(to.end(), std::begin(from), std::end(from));
}

}  // namespace vml
