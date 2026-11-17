// What a command-line run names its files after (issue #33): --name's
// template for each test video's files, and the same fields in the paths of
// --json, --scores-csv and --log-file, so that a pipeline names a whole
// run's files under its own scheme.
#pragma once

#include <ctime>
#include <optional>
#include <string>

namespace vml::cli {

// A field's text, for one test video, or for the run in a path.
struct NameFields {
    std::optional<std::string> test;       // the test video's file name, without its extension
    std::optional<std::string> reference;  // the reference's
    std::string version;                   // the app's, "2.6"
    std::string date;                      // the run's start, YYYYMMDD
    std::string time;                      // and HHMMSS, local time
    std::optional<std::string> sequence;   // the --batch row's sequence
};

// The run's own fields (version, date, time), at local time `started`.
NameFields run_fields(const std::tm& started);

// Why `name` cannot be --name's template, or nothing: a field that is not
// one of {test}, {reference}, {version}, {date}, {time} and {sequence},
// {sequence} without --batch (`batch`), a folder separator (it names files,
// not folders), or a brace with no partner. "{{" and "}}" are a brace each.
std::optional<std::string> name_template_error(const std::string& name, bool batch);

// The fields of `text` (a path) that are none of the run's own, by name, the
// first found: {test}, {reference} or {sequence}. Nothing when it has none.
std::optional<std::string> video_field_in(const std::string& text);

// `text` with each field replaced by its text in `fields`, and "{{" and "}}"
// by a brace. Braces that make no field are kept as they are (a path's own:
// --name's are refused first, name_template_error), and so is a field
// `fields` has no text for.
std::string expand_names(const std::string& text, const NameFields& fields);

}  // namespace vml::cli
