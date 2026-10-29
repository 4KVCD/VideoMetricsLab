// Native GPU code (the Vulkan engine, libvmaf, the GPU's driver) run in a
// process of its own, so that a crash in it ends that process and not the
// run: a port of vmaf_app/core/isolated.py for the work the C++ app does
// that way.
//
// The child is this program again, started with `isolated_argument`, a
// task's name and the file that holds the task's request (main() hands
// those to run_isolated_task). While it runs, it sends messages (progress,
// statuses) on its standard output, each a line of JSON after a mark, and
// its result or its error last. A child that ends without either crashed.
#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include "vml/cancel.h"

namespace vml {

// The child ended without a result, most likely because native code
// crashed (exit code 0xC0000005 is an access violation).
class IsolatedCrashError : public std::runtime_error {
public:
    IsolatedCrashError(const std::string& what, int exit_code);
    int exit_code() const { return exit_code_; }

private:
    int exit_code_;
};

// An exit code as it reads best: Windows' status codes in hexadecimal
// ("exit code 0xC0000005"), others in decimal.
std::string describe_exit_code(int exit_code);

// The argument that makes this program an isolated child.
inline constexpr std::string_view isolated_argument = "--vml-isolated";

// The program the children run: this one, set by main(). Without it, the
// work is done in this process instead (as in the tests).
void set_isolation_program(std::optional<std::filesystem::path> program);
std::optional<std::filesystem::path> isolation_program();

// Called with the child's result and the folder it ran in, before the
// folder is deleted: for a result too big for a message, which the child
// leaves in a file there.
using IsolatedFiles = std::function<void(const nlohmann::json& result, const std::filesystem::path& folder)>;

// Runs `task` in a child, with `request`, and returns its result.
// `on_message` gets each message the child sends while it runs, in order,
// on this thread. `what` names the work in IsolatedCrashError's message
// ("libvmaf crashed (exit code 0xC0000005)"). Throws what the child threw,
// as std::runtime_error with its message; IsolatedCrashError; and
// Cancelled when `cancel` is set, after ending the child and every program
// it started.
nlohmann::json run_isolated(std::string_view task, std::string_view what, const nlohmann::json& request,
                            const std::function<void(const nlohmann::json&)>& on_message, const CancelToken& cancel,
                            const IsolatedFiles& on_files = {});

// The token the GPU probes' children run under (Vship's, VMAF on the GPU's,
// VMAF v1's): cancelled once, by stop_gpu_probes() as the window's app ends,
// which ends the probes still running instead of waiting seconds for them
// (Qt waits for its pool's threads as the app ends), and makes later probes
// throw Cancelled. A cancelled probe is no answer: nothing is kept of it.
const CancelToken& gpu_probes_token();
void stop_gpu_probes();

// The child's side, for main(): runs task `task` with the request in
// `request_file` and returns the process's exit code.
int run_isolated_task(std::string_view task, const std::filesystem::path& request_file);

// The child's side: sends a message to the parent. Any thread may call it.
void send_isolated_message(const nlohmann::json& message);

}  // namespace vml
