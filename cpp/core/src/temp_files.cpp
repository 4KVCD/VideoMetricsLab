#include "vml/temp_files.h"

#include <charconv>
#include <chrono>
#include <climits>
#include <system_error>

#include "vml/platform/system.h"

namespace vml {

namespace {

// How long a name without a process id is kept untouched before it goes.
constexpr std::chrono::hours kept_without_owner{7 * 24};

// What a temporary file or folder's name says of its owner.
enum class Owner { none, unknown, process };

struct NameOwner {
    Owner owner = Owner::none;
    long long process = -1;  // with Owner::process; -1 for an id too large for any process
};

// The owner of the temporary file or folder `name`: none for a name that is
// not the app's, unknown for one of the app's names without a process id.
NameOwner owner_of(std::string_view name) {
    for (const std::string_view known : temp_prefixes) {
        if (!name.starts_with(known)) {
            continue;
        }
        const std::string_view rest = name.substr(known.size());
        const std::size_t digits = rest.find_first_not_of("0123456789");
        if (digits == 0 || digits == std::string_view::npos || rest[digits] != '-') {
            return {Owner::unknown};
        }
        long long process = -1;
        const auto [end, error] = std::from_chars(rest.data(), rest.data() + digits, process);
        return {Owner::process, error == std::errc() ? process : -1};
    }
    return {Owner::none};
}

// Whether the process that made a temporary file or folder still runs.
bool owner_running(long long process) {
    if (process == platform::process_id()) {
        return true;
    }
    return process >= 0 && process <= INT_MAX && platform::process_running(static_cast<int>(process));
}

}  // namespace

std::string temp_prefix(std::string_view name) {
    return std::string(name) + std::to_string(platform::process_id()) + "-";
}

int remove_leftovers(const std::filesystem::path& folder) {
    std::error_code error;
    std::filesystem::directory_iterator entry(folder, error);
    const auto now = std::filesystem::file_time_type::clock::now();
    int removed = 0;
    for (; !error && entry != std::filesystem::directory_iterator(); entry.increment(error)) {
        const NameOwner name = owner_of(platform::path_text(entry->path().filename()));
        if (name.owner == Owner::none || (name.owner == Owner::process && owner_running(name.process))) {
            continue;
        }
        if (name.owner == Owner::unknown) {
            std::error_code unread;
            const auto changed = std::filesystem::last_write_time(entry->path(), unread);
            if (unread || now - changed < kept_without_owner) {
                continue;
            }
        }
        std::error_code failed;
        std::filesystem::remove_all(entry->path(), failed);
        if (!failed) {
            ++removed;
        }
    }
    return removed;
}

}  // namespace vml
