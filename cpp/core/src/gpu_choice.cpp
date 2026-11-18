#include "vml/gpu_choice.h"

#include "vml/platform/system.h"
#include "vml/python_text.h"

namespace vml {

std::optional<std::string> chosen_gpu() {
    const std::string name(python_strip(platform::environment_variable(gpu_variable).value_or("")));
    return name.empty() ? std::nullopt : std::optional(name);
}

void choose_gpu(const std::optional<std::string>& name) {
    platform::set_environment_variable(gpu_variable, name ? std::string(python_strip(*name)) : std::string());
}

std::optional<std::size_t> chosen_among(const std::vector<std::string>& names, std::string_view chosen) {
    const std::string wanted = ascii_lower(std::string(python_strip(std::string(chosen))));
    if (wanted.empty()) {
        return std::nullopt;
    }
    for (std::size_t index = 0; index < names.size(); ++index) {
        if (ascii_lower(names[index]) == wanted) {
            return index;
        }
    }
    for (std::size_t index = 0; index < names.size(); ++index) {
        if (ascii_lower(names[index]).find(wanted) != std::string::npos) {
            return index;
        }
    }
    return std::nullopt;
}

}  // namespace vml
