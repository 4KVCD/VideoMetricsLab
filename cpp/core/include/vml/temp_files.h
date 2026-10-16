// The app's temporary files and folders, and the removal of those that runs
// left behind (the Python app's temp_files).
//
// A run removes its temporary files as it ends, however it ends. A program
// that is itself ended outright (End task, a crash, a power cut) cannot, and
// the pictures of SSIMULACRA2 and Butteraugli on the CPU alone can take more
// than a gigabyte. So each name is one of the app's prefixes, then the id of
// the process that made it (temp_prefix), and the next run removes those
// whose process no longer runs (remove_leftovers). Both apps name them the
// same way, so either removes what the other left. Names without a process
// id were made by earlier versions: they are removed once nothing has
// changed them for a week, as Windows' Disk Cleanup removes temporary files.
#pragma once

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace vml {

// The start of every temporary file or folder name the app uses.
inline constexpr std::array<std::string_view, 9> temp_prefixes{
    "videometricslab-perceptual-",  // the pictures of SSIMULACRA2 and Butteraugli on the CPU
    "vml-isolated-",                // a process of its own's folder
    "vmaf_run_",                    // FFmpeg's runs
    "vmaf_gpu_run_",
    "vmaf_cpu_run_",
    "vmaf_xpsnr_run_",
    "vmaf_resample_",
    "vml-vship-pts-",  // FFmpeg's frame timestamps for Vship
    "vml-vmaf-v1-",    // libvmaf's list of features
};

// `name`, one of temp_prefixes, followed by this process's id and a dash:
// the start of a temporary file or folder name whose owner remove_leftovers
// can find.
std::string temp_prefix(std::string_view name);

// Removes the app's temporary files and folders in `folder` whose process no
// longer runs, and those without a process id that nothing has changed for a
// week. Leaves everything else, and anything it cannot remove (a file still
// open). Returns how many it removed.
int remove_leftovers(const std::filesystem::path& folder);

// A new folder in the system's temporary folder, named temp_prefix(prefix)
// and a random number; nothing if none could be made in 100 tries.
std::optional<std::filesystem::path> make_temp_folder(std::string_view prefix);

// A new, empty file in `folder`, named temp_prefix(prefix), a random number
// and `suffix`; nothing if none could be made in 100 tries.
std::optional<std::filesystem::path> make_temp_file(const std::filesystem::path& folder, std::string_view prefix,
                                                    std::string_view suffix);

// A temporary folder that is deleted, with everything in it, when this is.
class TempFolder {
public:
    // Makes the folder (make_temp_folder), or throws Error(failure).
    template <typename Error>
    static TempFolder make(std::string_view prefix, const std::string& failure) {
        std::optional<std::filesystem::path> made = make_temp_folder(prefix);
        if (!made) {
            throw Error(failure);
        }
        return TempFolder(std::move(*made));
    }
    ~TempFolder();
    TempFolder(const TempFolder&) = delete;
    TempFolder& operator=(const TempFolder&) = delete;
    const std::filesystem::path& path() const { return path_; }

private:
    explicit TempFolder(std::filesystem::path path) : path_(std::move(path)) {}
    std::filesystem::path path_;
};

// A temporary file that is deleted when this is.
class TempFile {
public:
    // Makes the file (make_temp_file), or throws Error(failure).
    template <typename Error>
    static TempFile make(const std::filesystem::path& folder, std::string_view prefix, std::string_view suffix,
                         const std::string& failure) {
        std::optional<std::filesystem::path> made = make_temp_file(folder, prefix, suffix);
        if (!made) {
            throw Error(failure);
        }
        return TempFile(std::move(*made));
    }
    ~TempFile();
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
    const std::filesystem::path& path() const { return path_; }

private:
    explicit TempFile(std::filesystem::path path) : path_(std::move(path)) {}
    std::filesystem::path path_;
};

}  // namespace vml
