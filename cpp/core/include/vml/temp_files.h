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

}  // namespace vml
