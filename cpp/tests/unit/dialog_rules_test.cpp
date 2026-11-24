// The window's dialogs come from one place (docs/ARCHITECTURE.md, "Asking and
// telling the user"): every message from show_message (app_dialog.h) in the
// app's look, every dialog an AppDialog, the file dialogs from Dialogs
// (dialogs.cpp), which the window's tests answer. Qt's own message box,
// input, progress and colour dialogs would bring back the system's title bar
// and icons, and a QFileDialog outside Dialogs a dialog no test can answer.
#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Rule {
    std::regex pattern;
    std::set<std::string> allowed_in;  // the files that may: those that wrap it
    const char* why;
};

// `text` without its comments, so that a comment may name what it replaced.
std::string without_comments(const std::string& text) {
    static const std::regex comment(R"(//[^\n]*|/\*[\s\S]*?\*/)");
    return std::regex_replace(text, comment, "");
}

// "<line>: <rule's why>" for each use of a rule outside its files.
std::vector<std::string> broken_rules(const std::string& file, const std::string& text, const std::vector<Rule>& rules) {
    std::vector<std::string> found;
    // Most files name no dialog at all: those are passed without the
    // regular expressions, which are slow.
    if (text.find("Dialog") == std::string::npos && text.find("QMessageBox") == std::string::npos &&
        text.find("QErrorMessage") == std::string::npos) {
        return found;
    }
    const std::string code = without_comments(text);
    for (const Rule& rule : rules) {
        if (rule.allowed_in.contains(file)) {
            continue;
        }
        for (auto match = std::sregex_iterator(code.begin(), code.end(), rule.pattern); match != std::sregex_iterator();
             ++match) {
            const auto line = 1 + std::count(code.begin(), code.begin() + match->position(), '\n');
            found.push_back(std::to_string(line) + ": " + match->str() + ": " + rule.why);
        }
    }
    return found;
}

const std::vector<Rule>& rules() {
    static const std::vector<Rule> all{
        {std::regex(R"(\b(QMessageBox|QInputDialog|QProgressDialog|QColorDialog|QErrorMessage)\b)"),
         {"app_dialog.cpp", "app_dialog.h"},
         "Qt's own dialog: use show_message, ask_line, ProgressDialog or choose_color (app_dialog.h)"},
        {std::regex(R"(\bQFileDialog\b)"), {"dialogs.cpp"}, "a file dialog: use Dialogs (open_file, save_file, ...)"},
        {std::regex(R"(public\s+QDialog\b|\bnew\s+QDialog\b|\bQDialog\s+\w+\s*\()"),
         {"app_dialog.cpp", "app_dialog.h"},
         "a bare QDialog: build it on AppDialog (app_dialog.h)"},
    };
    return all;
}

TEST(DialogRules, TheFinderFindsWhatTheRulesForbid) {
    const std::string sample = "// QMessageBox in a comment is fine\n"
                               "QMessageBox::warning(this, a, b);\n"
                               "class X : public QDialog {\n";
    EXPECT_EQ(broken_rules("x.cpp", sample, rules()).size(), 2u);
    EXPECT_TRUE(broken_rules("app_dialog.cpp", sample, rules()).empty());
}

TEST(DialogRules, EveryDialogIsTheAppsOwn) {
    const std::filesystem::path folder = std::filesystem::path(VML_SOURCE_DIR) / "app";
    std::size_t files = 0;
    for (const auto& entry : std::filesystem::directory_iterator(folder)) {
        const std::filesystem::path& path = entry.path();
        if (!entry.is_regular_file() ||
            (path.extension() != ".cpp" && path.extension() != ".h" && path.extension() != ".mm")) {
            continue;
        }
        std::ifstream file(path, std::ios::binary);
        std::stringstream text;
        text << file.rdbuf();
        ++files;
        for (const std::string& broken : broken_rules(path.filename().string(), text.str(), rules())) {
            ADD_FAILURE() << path.filename().string() << ":" << broken;
        }
    }
    EXPECT_GT(files, 50u);  // the folder was found
}

}  // namespace
