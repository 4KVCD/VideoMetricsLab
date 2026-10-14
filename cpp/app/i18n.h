// The window's text in the user's language, as the Python app's
// vmaf_app/i18n.py gives it, from the same catalogs
// (vmaf_app/translations/<code>.json).
//
// The app opens in Windows' display language when it has a translation for
// it, and in English otherwise; Settings > Window > Language can choose
// another. Only what the window shows is translated: the log, saved results,
// exported files and the engine's messages stay in English.
//
// tr("Calculate metrics") translates a text. The English text is the key
// into the catalog, and is what shows when there is no translation.
// Placeholders are Python str.format fields, filled in from named values, as
// in tr("{done} done", {{"done", 3}}); a literal brace is doubled. ntr()
// picks the form for a number by the language's plural rules. N_() marks
// English text kept as data, which tr() translates where it is shown.
// tr_message() translates a message from the engine where it has one of the
// shapes the window knows (message_templates()); anything else, such as
// FFmpeg's own words or a path, stays as it came.
#pragma once

#include <QString>

#include <initializer_list>
#include <string>
#include <variant>
#include <vector>

namespace vml::app {

// A value for a {placeholder}: text, a whole number or a number with a
// fraction, formatted as Python's str.format formats it.
class FormatValue {
public:
    FormatValue(const QString& text) : value_(text) {}
    FormatValue(const char* text) : value_(QString::fromUtf8(text)) {}
    FormatValue(const std::string& text) : value_(QString::fromStdString(text)) {}
    FormatValue(int number) : value_(static_cast<long long>(number)) {}
    FormatValue(long number) : value_(static_cast<long long>(number)) {}
    FormatValue(long long number) : value_(number) {}
    FormatValue(unsigned number) : value_(static_cast<long long>(number)) {}
    FormatValue(unsigned long number) : value_(static_cast<long long>(number)) {}
    FormatValue(unsigned long long number) : value_(static_cast<long long>(number)) {}
    FormatValue(double number) : value_(number) {}

    // The value formatted with a format spec (the text after ":" in a field),
    // as Python formats it: [[fill]align][width][,][.precision][type].
    QString formatted(const QString& spec) const;

private:
    std::variant<QString, long long, double> value_;
};

// A placeholder's name and its value.
struct NamedValue {
    const char* name;
    FormatValue value;
};

// Marks `text` for translation where it is shown; returns it as it is.
constexpr const char* N_(const char* text) { return text; }

// `text` in the window's language, its {placeholders} filled in.
QString tr(const char* text, std::initializer_list<NamedValue> values = {});
QString tr(const QString& text, std::initializer_list<NamedValue> values = {});

// The text for `count` of something, {count} filled in: `singular` for one
// in English, `plural` otherwise, and in another language the form its rules
// give.
QString ntr(const char* singular, const char* plural, long long count, std::initializer_list<NamedValue> values = {});

// A message from the engine in the window's language, line by line, where it
// has a known shape; otherwise as it came.
QString tr_message(const QString& text);

// `text` with its {placeholders} filled in, as Python's str.format(**values)
// does. A field with no value is left as it is.
QString python_format(const QString& text, std::initializer_list<NamedValue> values);

// tr(), ntr() and python_format() with values gathered at run time.
QString tr(const QString& text, const std::vector<NamedValue>& values);
QString ntr(const char* singular, const char* plural, long long count, const std::vector<NamedValue>& values);
QString python_format(const QString& text, const std::vector<NamedValue>& values);

// A language the window can be shown in: its code ("de") and its name in
// itself ("Deutsch"), in the Settings list's order.
struct Language {
    QString code;
    QString name;
};
const std::vector<Language>& languages();

// Whether a language is written right to left: the whole window is mirrored.
bool right_to_left(const QString& code);

// Which of a language's forms a number takes (CLDR's rules for whole
// numbers); the catalogs list the forms in this order.
int plural_index(const QString& code, long long count);

// The translation for a language name as Windows gives it ("de-DE",
// "zh-Hant-TW", "pt-BR"), or "en" when there is none.
QString language_for(const QString& ui_language);

// The translation for Windows' display language, or "en".
QString windows_language();

// Shows the window's text in `code` from now on; English for a code with no
// translation, or one whose catalog cannot be read. Returns the language set.
QString set_language(const QString& code);

// The code of the language the window is shown in, such as "en" or "de".
QString language();

// The engine's message shapes the window translates (the Python app's
// MESSAGE_TEMPLATES).
const std::vector<QString>& message_templates();

}  // namespace vml::app
