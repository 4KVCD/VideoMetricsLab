#include "vml/gpu_choice.h"

#include <algorithm>
#include <utility>

#include "vml/app_log.h"
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

namespace {

// The GPUs of `names` named `name` (lower case), in the list's order.
std::vector<std::size_t> gpus_of_name(const std::vector<std::string>& names, const std::string& name) {
    std::vector<std::size_t> found;
    for (std::size_t index = 0; index < names.size(); ++index) {
        if (ascii_lower(names[index]) == name) {
            found.push_back(index);
        }
    }
    return found;
}

// "name #k" as the name and k (from 1); nothing for other text.
std::optional<std::pair<std::string, std::size_t>> name_and_place(const std::string& text) {
    const std::size_t mark = text.rfind(" #");
    if (mark == std::string::npos || mark + 2 == text.size() || text.size() - mark - 2 > 3) {
        return std::nullopt;
    }
    std::size_t place = 0;
    for (std::size_t at = mark + 2; at < text.size(); ++at) {
        if (text[at] < '0' || text[at] > '9') {
            return std::nullopt;
        }
        place = place * 10 + static_cast<std::size_t>(text[at] - '0');
    }
    return place == 0 ? std::nullopt : std::optional(std::pair(text.substr(0, mark), place));
}

}  // namespace

std::vector<std::string> gpu_choice_texts(const std::vector<std::string>& names) {
    std::vector<std::string> texts;
    for (std::size_t index = 0; index < names.size(); ++index) {
        const std::vector<std::size_t> same = gpus_of_name(names, ascii_lower(names[index]));
        if (same.size() < 2) {
            texts.push_back(names[index]);
        } else {
            const auto place = std::find(same.begin(), same.end(), index) - same.begin() + 1;
            texts.push_back(names[index] + " #" + std::to_string(place));
        }
    }
    return texts;
}

std::vector<std::size_t> gpus_named(const std::vector<std::string>& names, std::string_view chosen) {
    const std::string wanted = ascii_lower(std::string(python_strip(std::string(chosen))));
    if (wanted.empty()) {
        return {};
    }
    if (const std::vector<std::size_t> same = gpus_of_name(names, wanted); !same.empty()) {
        return {same.front()};
    }
    if (const auto numbered = name_and_place(wanted)) {
        const std::vector<std::size_t> same = gpus_of_name(names, numbered->first);
        if (!same.empty()) {
            return numbered->second <= same.size() ? std::vector<std::size_t>{same[numbered->second - 1]}
                                                   : std::vector<std::size_t>{};
        }
    }
    std::vector<std::size_t> found;
    for (std::size_t index = 0; index < names.size(); ++index) {
        if (ascii_lower(names[index]).find(wanted) != std::string::npos) {
            found.push_back(index);
        }
    }
    return found;
}

std::optional<std::size_t> chosen_among(const std::vector<std::string>& names, std::string_view chosen) {
    const std::vector<std::size_t> found = gpus_named(names, chosen);
    return found.size() == 1 ? std::optional(found.front()) : std::nullopt;
}

void resolve_chosen_gpu() {
    const std::optional<std::string> chosen = chosen_gpu();
    if (!chosen) {
        return;
    }
    std::vector<std::string> names;
    for (const platform::GpuAdapter& adapter : platform::gpu_adapters()) {
        names.push_back(adapter.name);
    }
    const std::vector<std::size_t> found = gpus_named(names, *chosen);
    if (found.size() == 1) {
        choose_gpu(gpu_choice_texts(names)[found.front()]);
    } else if (found.size() > 1) {
        app_log::write(app_log::Level::warning, "vml.gpu_choice",
                       "The GPU chosen, " + *chosen +
                           ", is part of more than one GPU's name: no GPU is used for the GPU metrics");
    }
}

}  // namespace vml
