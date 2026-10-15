// The window's text as the Python app gives it, from the same catalogs
// (reference/i18n.json, made by its vmaf_app/i18n.py).
#include "i18n.h"

#include <QCoreApplication>
#include <QDialogButtonBox>
#include <QFile>
#include <QLibraryInfo>
#include <QPushButton>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <string>
#include <vector>

#include "reference.h"

namespace vml::app {
namespace {

using Json = nlohmann::json;

// A text's placeholder values from the reference: text, a whole number or a
// number with a fraction, each by its name.
class Values {
public:
    explicit Values(const Json& values) {
        // Every name first: the values point into them.
        for (const auto& item : values.items()) {
            names_.push_back(item.key());
        }
        std::size_t name = 0;
        for (const auto& item : values.items()) {
            const Json& value = item.value();
            list_.push_back({names_[name++].c_str(),
                             value.is_string()           ? FormatValue(value.get<std::string>())
                             : value.is_number_integer() ? FormatValue(value.get<long long>())
                                                         : FormatValue(value.get<double>())});
        }
    }

    const std::vector<NamedValue>& list() const { return list_; }

private:
    std::vector<std::string> names_;
    std::vector<NamedValue> list_;
};

// What differs from the Python app's text, at most the first few, so that a
// broken rule says what it broke without a thousand lines.
class Differences {
public:
    void check(const std::string& expected, const QString& actual, const std::string& what) {
        if (actual.toStdString() == expected) {
            return;
        }
        if (++count_ <= 10) {
            shown_ += what + "\n  expected: " + expected + "\n  actual:   " + actual.toStdString() + "\n";
        }
    }

    ~Differences() { EXPECT_EQ(count_, 0) << shown_; }

private:
    int count_ = 0;
    std::string shown_;
};

// English again at the end of a test that chose another language.
struct InEnglishAfter {
    ~InEnglishAfter() { set_language("en"); }
};

TEST(I18n, WindowsLanguagesMapToTheTranslations) {
    const Json reference = test::load_reference("i18n.json");
    Differences differences;
    for (const auto& item : reference["windows"].items()) {
        differences.check(item.value().get<std::string>(), language_for(QString::fromStdString(item.key())),
                          "\"" + item.key() + "\"");
    }
}

TEST(I18n, EachLanguagesPluralRules) {
    const Json reference = test::load_reference("i18n.json");
    for (const auto& item : reference["plural"].items()) {
        std::string forms;
        for (const Json& count : reference["counts"]) {
            forms += std::to_string(plural_index(QString::fromStdString(item.key()), count.get<long long>()));
        }
        EXPECT_EQ(forms, item.value().get<std::string>()) << item.key();
    }
}

TEST(I18n, PlaceholdersAreFilledInAsPythonsStrFormatFillsThem) {
    const Json reference = test::load_reference("i18n.json");
    Differences differences;
    for (const Json& item : reference["formats"]) {
        const std::string text = item["text"].get<std::string>();
        differences.check(item["formatted"].get<std::string>(),
                          python_format(QString::fromStdString(text), Values(item["values"]).list()),
                          text + " with " + item["values"].dump());
    }
}

// Every text with placeholders, every counted text, and the engine's
// messages, in each kind of catalog: plural forms of two, three and six,
// right to left, ideographic full stops, no spaces.
TEST(I18n, TheWindowsTextIsThePythonAppsInEachLanguage) {
    const Json reference = test::load_reference("i18n.json");
    const InEnglishAfter english;
    Differences differences;
    ASSERT_FALSE(reference["languages"].empty());
    for (const auto& item : reference["languages"].items()) {
        const std::string& code = item.key();
        const Json& expected = item.value();
        ASSERT_EQ(set_language(QString::fromStdString(code)).toStdString(), code);
        const Json& fielded = reference["fielded"];
        for (std::size_t i = 0; i < fielded.size(); ++i) {
            const std::string text = fielded[i]["text"].get<std::string>();
            differences.check(expected["fielded"][i].get<std::string>(),
                              tr(QString::fromStdString(text), Values(fielded[i]["values"]).list()),
                              code + ": " + text);
        }
        const Json& counted = reference["counted"];
        const Json& counts = reference["counted_counts"];
        for (std::size_t i = 0; i < counted.size(); ++i) {
            const std::string singular = counted[i]["singular"].get<std::string>();
            const std::string plural = counted[i]["plural"].get<std::string>();
            const Values values(counted[i]["values"]);
            for (std::size_t j = 0; j < counts.size(); ++j) {
                const long long count = counts[j].get<long long>();
                differences.check(expected["counted"][i][j].get<std::string>(),
                                  ntr(singular.c_str(), plural.c_str(), count, values.list()),
                                  code + ": " + singular + " for " + std::to_string(count));
            }
        }
        const Json& messages = reference["messages"];
        for (std::size_t i = 0; i < messages.size(); ++i) {
            const std::string message = messages[i].get<std::string>();
            differences.check(expected["messages"][i].get<std::string>(), tr_message(QString::fromStdString(message)),
                              code + ": " + message);
        }
    }
}

// Qt's own buttons, in Qt's translation (qtbase_de.qm), where Qt's folder has
// its translations.
TEST(I18n, QtsOwnButtonsAreInTheChosenLanguage) {
    if (!QFile::exists(QLibraryInfo::path(QLibraryInfo::TranslationsPath) + "/qtbase_de.qm")) {
        GTEST_SKIP() << "this Qt has no translations (qttranslations)";
    }
    QTranslator* german = install_qt_translations("de");
    ASSERT_NE(german, nullptr);
    const QDialogButtonBox buttons(QDialogButtonBox::Cancel);
    const QString cancel = buttons.button(QDialogButtonBox::Cancel)->text();
    QCoreApplication::removeTranslator(german);
    delete german;
    EXPECT_EQ(cancel.toStdString(), "Abbrechen");
}

}  // namespace
}  // namespace vml::app
