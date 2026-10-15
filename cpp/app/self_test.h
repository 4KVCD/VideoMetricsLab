// The window's self-test (the Python app's main.self_test): checks the tools
// and libraries the app finds at run time, not at build time, and reports
// what it found. It exists for the packaged build, which has no console:
// when the build fails to start, or quietly falls back to something slower,
// the report is the only thing to look at; and the release script runs it
// (VideoMetricsLab.exe --self-test --quiet).
#pragma once

#include <QApplication>

#include <string>
#include <vector>

namespace vml::app {

// The self-test's lines, and whether any check failed. Each check records
// its failure as it is made: reading "FAIL" back from the text would wrongly
// fail a run whose paths or messages contain that word.
class SelfTestReport {
public:
    explicit SelfTestReport(std::string title) { lines_.push_back(std::move(title)); }
    void ok(const std::string& text) { lines_.push_back("  OK    " + text); }
    void warn(const std::string& text) { lines_.push_back("  WARN  " + text); }
    void fail(const std::string& text) {
        lines_.push_back("  FAIL  " + text);
        failed_ = true;
    }
    void note(const std::string& text) { lines_.push_back("  " + text); }
    bool failed() const { return failed_; }
    std::string text() const;

private:
    std::vector<std::string> lines_;
    bool failed_ = false;
};

SelfTestReport self_test(const QApplication& app);

// Runs the self-test for main(): prints and saves its report (the user's
// .videometricslab/self-test.txt), shows it unless `quiet`, and returns the
// program's exit code (1 when a check failed).
int run_self_test(const QApplication& app, bool quiet);

}  // namespace vml::app
