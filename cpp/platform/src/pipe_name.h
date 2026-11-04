// What every system's pipe names are made of (pipe.cpp).
#pragma once

#include <string>

namespace vml::platform {

// 12 random hexadecimal digits, which make a pipe's name unique.
std::string random_part();

}  // namespace vml::platform
