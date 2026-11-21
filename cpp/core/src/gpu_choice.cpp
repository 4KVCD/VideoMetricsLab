#include "vml/gpu_choice.h"

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

std::vector<std::size_t> gpus_named(const std::vector<std::string>& names, std::string_view chosen) {
    const std::string wanted = ascii_lower(std::string(python_strip(std::string(chosen))));
    if (wanted.empty()) {
        return {};
    }
    for (std::size_t index = 0; index < names.size(); ++index) {
        if (ascii_lower(names[index]) == wanted) {
            return {index};
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
        choose_gpu(names[found.front()]);
    } else if (found.size() > 1) {
        app_log::write(app_log::Level::warning, "vml.gpu_choice",
                       "The GPU chosen, " + *chosen +
                           ", is part of more than one GPU's name: no GPU is used for the GPU metrics");
    }
}

}  // namespace vml
