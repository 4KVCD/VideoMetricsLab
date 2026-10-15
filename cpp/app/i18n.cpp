#include "i18n.h"

#include <QCoreApplication>
#include <QHash>
#include <QLibraryInfo>
#include <QLocale>
#include <QRegularExpression>
#include <QStringList>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <utility>

#include "vml/model_select.h"
#include "vml/number_format.h"
#include "vml/platform/system.h"

namespace vml::app {

namespace {

using Values = std::vector<std::pair<QString, FormatValue>>;

const std::vector<Language> all_languages = {
    {"en", "English"},
    {"zh_CN", "简体中文"},
    {"zh_TW", "繁體中文"},
    {"es", "Español"},
    {"pt_BR", "Português (Brasil)"},
    {"de", "Deutsch"},
    {"fr", "Français"},
    {"ja", "日本語"},
    {"ru", "Русский"},
    {"ko", "한국어"},
    {"it", "Italiano"},
    {"pl", "Polski"},
    {"tr", "Türkçe"},
    {"ar", "العربية"},
    {"id", "Bahasa Indonesia"},
    {"vi", "Tiếng Việt"},
    {"uk", "Українська"},
    {"th", "ไทย"},
    {"cs", "Čeština"},
    {"hu", "Magyar"},
    {"nl", "Nederlands"},
};

// The language shown, and its catalog: each English text's translation, and
// each counted text's forms. Set at the start (set_language), read after.
struct Catalog {
    QString language = "en";
    QHash<QString, QString> strings;
    QHash<QString, QStringList> plurals;
};
Catalog catalog;

// Lines of the engine's messages translated so far (tr_message), from any
// thread.
std::mutex translated_mutex;
QHash<QString, QString> translated_lines;
constexpr qsizetype translated_kept = 4096;

// While an InEnglish lives on this thread.
thread_local bool english_here = false;

// Whether text is translated: a language other than English is shown, and
// not for the log.
bool active() { return catalog.language != "en" && !english_here; }

// A spec's parts: [[fill]align][sign][#][0][width][grouping][.precision][type].
struct Spec {
    QChar fill = ' ';
    QChar align;  // '<', '>', '^', or none (the value's own)
    int width = 0;
    QChar grouping;  // ',' or '_', or none
    int precision = -1;
    QChar type;
};

bool is_align(QChar c) { return c == '<' || c == '>' || c == '^' || c == '='; }

Spec parse_spec(const QString& text) {
    Spec spec;
    qsizetype at = 0;
    if (text.size() >= 2 && is_align(text[1])) {
        spec.fill = text[0];
        spec.align = text[1];
        at = 2;
    } else if (!text.isEmpty() && is_align(text[0])) {
        spec.align = text[0];
        at = 1;
    }
    // A sign, "#" and "0" are taken and dropped: no text of the window uses them.
    while (at < text.size() && (text[at] == '+' || text[at] == '-' || text[at] == ' ' || text[at] == '#')) {
        ++at;
    }
    while (at < text.size() && text[at].isDigit()) {
        spec.width = spec.width * 10 + text[at].digitValue();
        ++at;
    }
    if (at < text.size() && (text[at] == ',' || text[at] == '_')) {
        spec.grouping = text[at++];
    }
    if (at < text.size() && text[at] == '.') {
        ++at;
        spec.precision = 0;
        while (at < text.size() && text[at].isDigit()) {
            spec.precision = spec.precision * 10 + text[at].digitValue();
            ++at;
        }
    }
    if (at < text.size()) {
        spec.type = text[at];
    }
    return spec;
}

// `digits` (a number's whole part, no sign) with `separator` every three
// digits from the right.
QString grouped(const QString& digits, QChar separator) {
    QString out;
    const qsizetype count = digits.size();
    for (qsizetype i = 0; i < count; ++i) {
        if (i > 0 && (count - i) % 3 == 0) {
            out += separator;
        }
        out += digits[i];
    }
    return out;
}

// A number's text with grouping applied to its whole part.
QString with_grouping(const QString& text, QChar separator) {
    if (separator.isNull()) {
        return text;
    }
    const bool negative = text.startsWith('-');
    const QString body = negative ? text.mid(1) : text;
    qsizetype end = 0;
    while (end < body.size() && body[end].isDigit()) {
        ++end;
    }
    return (negative ? QStringLiteral("-") : QString()) + grouped(body.left(end), separator) + body.mid(end);
}

QString format_double(double value, const Spec& spec) {
    if (std::isnan(value)) {
        return "nan";
    }
    if (std::isinf(value)) {
        return value > 0 ? "inf" : "-inf";
    }
    std::string text;
    const QChar type = spec.type;
    if (type == 'f' || type == 'F') {
        text = std::format("{:.{}f}", value, spec.precision < 0 ? 6 : spec.precision);
    } else if (type == 'e' || type == 'E') {
        text = std::format("{:.{}e}", value, spec.precision < 0 ? 6 : spec.precision);
    } else if (type == '%') {
        text = std::format("{:.{}f}%", value * 100.0, spec.precision < 0 ? 6 : spec.precision);
    } else if (type == 'g' || type == 'G' || spec.precision >= 0) {
        text = std::format("{:.{}g}", value, spec.precision < 0 ? 6 : std::max(spec.precision, 1));
    } else {
        text = python_repr(value);  // str(float)
    }
    return with_grouping(QString::fromStdString(text), spec.grouping);
}

// `text` padded with the spec's fill to its width; `numeric` values align
// right unless the spec says otherwise.
QString padded(const QString& text, const Spec& spec, bool numeric) {
    const qsizetype missing = spec.width - text.size();
    if (missing <= 0) {
        return text;
    }
    const QChar align = spec.align.isNull() ? QChar(numeric ? '>' : '<') : spec.align;
    if (align == '<') {
        return text + QString(missing, spec.fill);
    }
    if (align == '^') {
        const qsizetype left = missing / 2;
        return QString(left, spec.fill) + text + QString(missing - left, spec.fill);
    }
    return QString(missing, spec.fill) + text;
}

// `text` formatted as Python's str.format(**values). A field that names no
// value (or a brace that opens none) gives nothing, as Python raises
// KeyError, or with `keep_unknown` is left as it is.
std::optional<QString> format_fields(const QString& text, const Values& values, bool keep_unknown) {
    QString out;
    const qsizetype size = text.size();
    qsizetype at = 0;
    while (at < size) {
        const QChar c = text[at];
        if (c == '{' && at + 1 < size && text[at + 1] == '{') {
            out += '{';
            at += 2;
        } else if (c == '}' && at + 1 < size && text[at + 1] == '}') {
            out += '}';
            at += 2;
        } else if (c == '{') {
            const qsizetype close = text.indexOf('}', at + 1);
            if (close < 0) {
                if (!keep_unknown) {
                    return std::nullopt;
                }
                out += text.mid(at);
                break;
            }
            const QString field = text.mid(at + 1, close - at - 1);
            const qsizetype colon = field.indexOf(':');
            QString name = colon < 0 ? field : field.left(colon);
            if (const qsizetype bang = name.indexOf('!'); bang >= 0) {
                name.truncate(bang);  // !s: the value's text, as it is anyway
            }
            const auto found = std::find_if(values.begin(), values.end(),
                                            [&name](const auto& value) { return value.first == name; });
            if (found != values.end()) {
                out += found->second.formatted(colon < 0 ? QString() : field.mid(colon + 1));
            } else if (keep_unknown) {
                out += text.mid(at, close - at + 1);
            } else {
                return std::nullopt;
            }
            at = close + 1;
        } else {
            out += c;
            ++at;
        }
    }
    return out;
}

std::optional<QString> try_format(const QString& text, const Values& values) {
    return format_fields(text, values, false);
}

Values named(std::span<const NamedValue> values) {
    Values out;
    out.reserve(values.size());
    for (const NamedValue& value : values) {
        out.emplace_back(QString::fromUtf8(value.name), value.value);
    }
    return out;
}

std::span<const NamedValue> all(std::initializer_list<NamedValue> values) { return {values.begin(), values.size()}; }

// `text` with its fields filled in; a field that names no value is left as
// it is.
QString formatted(const QString& text, const Values& values) { return *format_fields(text, values, true); }

// --------------------------------------------------------------- core messages
// The shapes of the engine's statuses and errors that the window shows, as
// templates. "{name}" matches any text, which is itself translated when it
// is a message too. Kept in step with the Python app's MESSAGE_TEMPLATES,
// whose translations the catalogs hold.
const char* const templates[] = {
    // --- a video's steps on the run line ("distorted" shown as "test video")
    "Detecting black bars in source and test video…",
    "Detecting black bars in source…",
    "Running ffmpeg (GPU decode: {plan})…",
    "Running ffmpeg, VMAF on the GPU (GPU decode: {plan})…",
    "Running ffmpeg, {metrics} in the app (GPU decode: {plan})…",
    "Running VMAF on the GPU (GPU decode: {plan})…",
    "Running VMAF on the GPU…",
    "Running {metrics} in the app (GPU decode: {plan})…",
    "Running {metrics} in the app…",
    "{metrics} in the app failed ({error}); calculating them with FFmpeg…",
    "Decoding in the app failed ({error}); decoding through FFmpeg instead…",
    "GPU decoding failed ({error}); decoding through FFmpeg instead…",
    "GPU metric {number}/{count}: {metrics}",
    "VMAF on the GPU failed ({error}); calculating it on the CPU…",
    "GPU decode failed, retrying (GPU decode: {plan})…",
    "GPU decode failed, retrying…",
    "Vship GPU ({device}): calculating {metrics} (GPU decode: {plan})…",
    "Vship GPU ({device}): calculating {metrics}…",
    "GPU decode failed for the source, decoding it in software (GPU decode: {plan})…",
    "GPU decode failed for the test video, decoding it in software (GPU decode: {plan})…",
    "GPU decode failed for the source, decoding it in software…",
    "GPU decode failed for the test video, decoding it in software…",
    "Calculating SSIMULACRA2/Butteraugli on the CPU as frames are extracted (GPU decode: off)…",
    "Calculating SSIMULACRA2/Butteraugli on the CPU as frames are extracted…",
    "Calculating selected perceptual metric(s) on CPU…",
    "{metrics} failed in the shared GPU pass; calculating it in a pass of its own…",
    "{metrics} failed in the shared GPU pass; calculating each in a pass of its own…",
    "{metrics} failed on the GPU; calculating it on the CPU…",
    "Vship GPU unavailable ({reason}); using CPU reference metrics…",
    "Vship GPU compute failed ({error}); using CPU reference metrics…",
    "Waiting for the GPU: another video's GPU pass is running…",
    // --- why a video or a metric failed
    "{metrics} failed: {error}",
    "Cancelled by user",
    "GPU scoring failed ({error}). It was not retried on the CPU, which would take hours to days for a video "
    "over 10 minutes. Choose CPU for {metrics} to calculate it on the CPU anyway.",
    "GPU scoring failed ({error}); the CPU retry failed too: {reason}",
    "{metric} needs a GPU that Vship can use, and could not use one: {reason}",
    "{metric} could not be calculated: {reason}",
    "Vship GPU acceleration is only bundled for Windows.",
    "No GPU that Vship can use was found.",
    "Vship GPU calculation failed: {error}",
    "Vship produced no scores.",
    "FFmpeg produced no frame pairs for Vship.",
    "FFmpeg failed while decoding the {side} for Vship: {error}",
    "FFmpeg failed while decoding the {side} for Vship.",
    "Could not start FFmpeg for Vship: {error}",
    "FFmpeg ended partway through a {side} frame.",
    "FFmpeg did not give the timestamp of a {side} frame.",
    "The source has no frame at the test video's first.",
    "Could not allocate Vship pinned frame memory: {error}",
    "Could not initialize Vship {metric}: {error}",
    "Vship {metric} failed: {error}",
    "Vship {metric} returned a non-finite value.",
    "CVVDP needs the video's frame rate, and none was found (it was read as {fps} fps).",
    "The GPU metrics do not support resolution round-trip tests yet.",
    "CVVDP scores every frame; it cannot be subsampled.",
    "Perceptual metrics in one task must use the same frame coverage.",
    "The GPU metrics cannot read the decoded pixel format {format}.",
    "Vship does not support the {matrix} color matrix.",
    "Vship does not support the {transfer} transfer function.",
    "Vship does not support the {primaries} color primaries.",
    "Vship does not support the {range} range tag.",
    "Vship does not support {siting} chroma siting.",
    "{video} is a raw {codec} stream with no timestamps, so its frame rate is unknown. Put it in a container with "
    "its frame rate first, e.g. mkvmerge -o video.mkv --default-duration 0:120fps <file> (MKVToolNix), with the "
    "video's own frame rate.",
    "Variable-frame-rate video is not supported safely yet. Convert both videos to the same constant frame rate "
    "before comparing them.",
    "Frame rates do not match ({source} vs {test} fps).",
    "Durations do not match ({source} vs {test} seconds). Set a duration limit within both files if comparing "
    "only their common opening segment.",
    "The duration limit extends beyond the end of one of the videos.",
    "The two videos are different shapes after cropping: source {source} versus distorted {test}. Scoring them "
    "would stretch one to fit the other and the result would be meaningless. If one is letterboxed, set "
    "black-bar handling to 'Auto-detect' so the bars are removed before comparison.",
    "FFmpeg could not prepare lossless perceptual-metric frames.",
    "FFmpeg produced no frame pairs for perceptual metrics.",
    "{metric} did not produce a numeric score: {value}",
    "Could not parse {metric} score.",
    "Could not run {metric}: {error}",
    "{metric} did not finish a frame within {seconds} s.",
    "{metric} failed for a frame.",
    "Perceptual CPU metrics do not support resolution round-trip tests yet.",
    "{tool} is not installed. Install the reference {tool2} executable or set {variable}_PATH.",
    "ffmpeg exited with code {code}",
    "ffmpeg finished but no metric log was produced.",
    "No results for requested metrics: {metrics}",
    "Select at least one metric to calculate.",
    "Crop detection cancelled",
    "Crop detection timed out",
    "Could not run crop detection: {error}",
    "Crop detection failed for {path}: {detail}",
    "Could not auto-detect black bars in {path}: FFmpeg found no pictures in the parts of it that were sampled. "
    "The file may be damaged or cut short.",
    "Only {compared} of the {expected} frames expected could be compared: one of the videos ends after {time}, "
    "though its file says it is longer. It may have been cut short: an encode that stopped early, or a copy that "
    "did not finish.",
    "Could not auto-detect black bars in {path}: {detail}. Choose 'None (use full frame)' for this video to "
    "continue without cropping.",
    // --- reading files, settings, tools
    "ffprobe was not found. Make sure ffmpeg is installed and on PATH, or set a custom ffmpeg folder in Settings.",
    "Could not start ffprobe for {path}: {error}",
    "ffprobe timed out while reading {path}",
    "Probe of {path} was cancelled",
    "ffprobe failed for {path}:",
    "Could not parse ffprobe output for {path}",
    "No video stream found in {path}",
    "Video file is missing: {path}",
    "Could not start ffprobe: {error}",
    "Bitrate analysis was cancelled.",
    "ffprobe returned no packets for the first video stream.",
    "This video has no usable frame rate.",
    "Frame extraction was cancelled.",
    "No frame was returned at frame {frame}; it may be beyond the end of the video.",
    "Every display value must be a finite number.",
    "The display resolution must be at least 16 x 16.",
    "The display's screen size must be greater than zero.",
    "The display's viewing distance must be greater than zero.",
    "The display's peak brightness must be greater than zero.",
    "The display's contrast must be greater than zero.",
    "The display's exposure must be greater than zero.",
    "Ambient light cannot be negative and reflectivity must be between 0 and 1.",
    "A preset needs a name.",
    "\"{name}\" is a built-in preset; choose another name.",
    "No custom model file selected.",
    "Built-in VMAF model is missing: {path}",
    "Unsupported analysis result format version: {version}",
    "Could not save settings: {error}",
    "{tool} could not be run ({error}).",
    "not found",
    "exited with code {code}",
    "ffmpeg {version} is too old -- version {minimum} or newer is required.",
    // --- Video Compare's playback, a " · " part at a time
    "Playing",
    "Paused",
    "Playback ended",
    "Opening with GStreamer/D3D11…",
    "Opening with FFmpeg tone-map fallback… ({reason})",
    "Opening with FFmpeg tone-map fallback…",
    "GStreamer could not start native playback: {error}",
    "GStreamer playback failed; trying FFmpeg fallback: {error}",
    "Could not decode comparison with ffmpeg: {error}",
    "Video {number} could not play: {error}",
    "Selected video failed: {error}",
    "Buffering selected comparison; adjacent videos are preparing in the background",
    "Preparing source/current/adjacent videos",
    "GPU tone mapping and RGB",
    "native fallback: {reason}",
    "{count}/{limit} streams",
    "SDR preview (native playback unavailable)",
    "Video playback is unavailable for synthetic resolution tests; their processed frames are displayed "
    "automatically.",
    "Could not create the GStreamer pipeline.",
    "GStreamer could not open the comparison.",
    "GStreamer could not seek to that frame.",
    "Could not open the {side} video file for decoding.",
    "D3D11 device unavailable",
    "Native frame lock currently requires matching frame rates",
    "Audio preroll timed out",
    "Video frame has no usable presentation timestamp",
    "Buffering locked pair",
    "Frame-locked GPU pair",
    "source soundtrack",
    "audio unavailable",
    "Vulkan GPU decode + GPU tone mapping/RGB",
    "hardware decode + GPU tone mapping/RGB (host transfer)",
    "software decode + GPU tone mapping/RGB",
    "CPU fallback (GPU processing unavailable)",
    "retry: {error}",
    "Decoder returned a truncated frame.",
    "Decoder produced no frames.",
    "ffmpeg returned no video frame.",
};

// Without a closing "...", "…" or full stop: the run line shows some
// messages trimmed of theirs.
QString bare(const QString& text) {
    static const QString closing = QStringLiteral(".…。 ");
    qsizetype end = text.size();
    while (end > 0 && closing.contains(text[end - 1])) {
        --end;
    }
    return text.left(end);
}

const QRegularExpression placeholder(QStringLiteral(R"(\{(\w+)\})"));

// A regular expression matching a message of the shape of `template`, with a
// named group for each placeholder.
QRegularExpression pattern_of(const QString& shape) {
    const QString text = bare(shape);
    const QString ellipsis = QRegularExpression::escape(QStringLiteral("…"));
    QString regex;
    qsizetype at = 0;
    QRegularExpressionMatchIterator fields = placeholder.globalMatch(text);
    const auto literal = [&ellipsis](const QString& part) {
        return QRegularExpression::escape(part).replace(ellipsis, QStringLiteral(R"((?:…|\.\.\.))"));
    };
    while (fields.hasNext()) {
        const QRegularExpressionMatch field = fields.next();
        regex += literal(text.mid(at, field.capturedStart() - at));
        regex += QStringLiteral("(?<%1>.+?)").arg(field.captured(1));
        at = field.capturedEnd();
    }
    regex += literal(text.mid(at));
    return QRegularExpression(QRegularExpression::anchoredPattern(regex),
                              QRegularExpression::DotMatchesEverythingOption);
}

// The templates as patterns, the most specific first (the one with the most
// fixed text), so that "Vship {metric} failed: {error}" is tried before
// "{metrics} failed: {error}".
const std::vector<std::pair<QString, QRegularExpression>>& patterns() {
    static const std::vector<std::pair<QString, QRegularExpression>> ordered = [] {
        std::vector<QString> shapes = message_templates();
        std::stable_sort(shapes.begin(), shapes.end(), [](const QString& a, const QString& b) {
            return QString(a).remove(placeholder).size() > QString(b).remove(placeholder).size();
        });
        std::vector<std::pair<QString, QRegularExpression>> out;
        for (const QString& shape : shapes) {
            out.emplace_back(shape, pattern_of(shape));
        }
        return out;
    }();
    return ordered;
}

// One line of an engine message in the window's language, or the line as
// it came.
QString translate_line(const QString& line) {
    if (line.trimmed().isEmpty()) {
        return line;
    }
    {
        const std::lock_guard lock(translated_mutex);
        if (const auto found = translated_lines.constFind(line); found != translated_lines.constEnd()) {
            return *found;
        }
    }
    QString out = line;
    if (const auto found = catalog.strings.constFind(line); found != catalog.strings.constEnd()) {
        out = *found;
    } else {
        // Matched without its closing "..." or full stop. The ending is put
        // back in the translation's own form, or left off if the line had
        // none.
        const QString trimmed = bare(line);
        const QString ending = line.mid(trimmed.size());
        for (const auto& [shape, pattern] : patterns()) {
            const QRegularExpressionMatch match = pattern.match(trimmed);
            const auto translation = catalog.strings.constFind(shape);
            if (!match.hasMatch() || translation == catalog.strings.constEnd()) {
                continue;
            }
            Values values;
            for (const QString& name : pattern.namedCaptureGroups()) {
                if (!name.isEmpty()) {
                    values.emplace_back(name, translate_line(match.captured(name)));
                }
            }
            if (const std::optional<QString> text = try_format(*translation, values)) {
                out = ending.trimmed().isEmpty() ? bare(*text) + ending : *text;
            }
            break;
        }
    }
    const std::lock_guard lock(translated_mutex);
    if (translated_lines.size() >= translated_kept) {
        translated_lines.clear();
    }
    translated_lines.insert(line, out);
    return out;
}

// tr(): with no values, the text is not formatted at all, as Python's
// text.format() is not called then.
QString translated(const QString& text, std::span<const NamedValue> values) {
    QString shown = text;
    if (active()) {
        shown = catalog.strings.value(text, text);
    }
    return values.empty() ? shown : formatted(shown, named(values));
}

// ntr().
QString counted(const char* singular, const char* plural, long long count, std::span<const NamedValue> values) {
    const QString key = QString::fromUtf8(singular);
    QString text = count == 1 ? key : QString::fromUtf8(plural);
    if (active()) {
        if (const auto forms = catalog.plurals.constFind(key);
            forms != catalog.plurals.constEnd() && !forms->isEmpty()) {
            text = forms->at(std::min<qsizetype>(plural_index(catalog.language, count), forms->size() - 1));
        }
    }
    Values filled = named(values);
    filled.emplace_back(QStringLiteral("count"), count);
    return formatted(text, filled);
}

}  // namespace

QString FormatValue::formatted(const QString& spec_text) const {
    const Spec spec = parse_spec(spec_text);
    if (const QString* text = std::get_if<QString>(&value_)) {
        return padded(spec.precision >= 0 ? text->left(spec.precision) : *text, spec, false);
    }
    if (const long long* number = std::get_if<long long>(&value_)) {
        const QChar type = spec.type;
        if (type == 'f' || type == 'F' || type == 'e' || type == 'E' || type == 'g' || type == 'G' || type == '%') {
            return padded(format_double(static_cast<double>(*number), spec), spec, true);
        }
        return padded(with_grouping(QString::number(*number), spec.grouping), spec, true);
    }
    return padded(format_double(std::get<double>(value_), spec), spec, true);
}

QString python_format(const QString& text, std::initializer_list<NamedValue> values) {
    return formatted(text, named(all(values)));
}

QString python_format(const QString& text, const std::vector<NamedValue>& values) {
    return formatted(text, named(values));
}

QString tr(const char* text, std::initializer_list<NamedValue> values) {
    return tr(QString::fromUtf8(text), values);
}

QString tr(const QString& text, std::initializer_list<NamedValue> values) { return translated(text, all(values)); }

QString tr(const QString& text, const std::vector<NamedValue>& values) { return translated(text, values); }

QString ntr(const char* singular, const char* plural, long long count, std::initializer_list<NamedValue> values) {
    return counted(singular, plural, count, all(values));
}

QString ntr(const char* singular, const char* plural, long long count, const std::vector<NamedValue>& values) {
    return counted(singular, plural, count, values);
}

QString tr_message(const QString& text) {
    if (!active() || text.isEmpty()) {
        return text;
    }
    QStringList lines = text.split('\n');
    for (QString& line : lines) {
        line = translate_line(line);
    }
    return lines.join('\n');
}

const std::vector<Language>& languages() { return all_languages; }

bool right_to_left(const QString& code) { return code == "ar"; }

int plural_index(const QString& code, long long count) {
    const long long n = count < 0 ? -count : count;
    if (code == "zh_CN" || code == "zh_TW" || code == "ja" || code == "ko" || code == "th" || code == "vi" ||
        code == "id") {
        return 0;
    }
    if (code == "fr") {
        return n == 0 || n == 1 ? 0 : 1;
    }
    if (code == "ru" || code == "uk") {
        if (n % 10 == 1 && n % 100 != 11) {
            return 0;
        }
        return 2 <= n % 10 && n % 10 <= 4 && !(12 <= n % 100 && n % 100 <= 14) ? 1 : 2;
    }
    if (code == "pl") {
        if (n == 1) {
            return 0;
        }
        return 2 <= n % 10 && n % 10 <= 4 && !(12 <= n % 100 && n % 100 <= 14) ? 1 : 2;
    }
    if (code == "cs") {
        return n == 1 ? 0 : 2 <= n && n <= 4 ? 1 : 2;
    }
    if (code == "ar") {
        if (n <= 2) {
            return static_cast<int>(n);  // zero, one, two
        }
        return 3 <= n % 100 && n % 100 <= 10 ? 3 : 11 <= n % 100 && n % 100 <= 99 ? 4 : 5;
    }
    return n == 1 ? 0 : 1;
}

QString language_for(const QString& ui_language) {
    static const QRegularExpression separators(QStringLiteral("[-_]"));
    QStringList parts = ui_language.trimmed().split(separators, Qt::SkipEmptyParts);
    if (parts.isEmpty()) {
        return "en";
    }
    const QString base = parts.takeFirst().toLower();
    if (base == "zh") {
        for (const QString& part : parts) {
            const QString lower = part.toLower();
            if (lower == "hant" || lower == "tw" || lower == "hk" || lower == "mo") {
                return "zh_TW";
            }
        }
        return "zh_CN";
    }
    if (base == "pt") {
        return "pt_BR";
    }
    const bool known = std::any_of(all_languages.begin(), all_languages.end(),
                                   [&base](const Language& language) { return language.code == base; });
    return known ? base : QStringLiteral("en");
}

QString windows_language() {
    const QStringList languages = QLocale::system().uiLanguages();
    return languages.isEmpty() ? QStringLiteral("en") : language_for(languages.first());
}

QString set_language(const QString& code) {
    Catalog chosen;
    const bool known = std::any_of(all_languages.begin(), all_languages.end(),
                                   [&code](const Language& language) { return language.code == code; });
    if (code != "en" && known) {
        chosen.language = code;
        const std::filesystem::path file =
            resource_folder() / "translations" / platform::path_from_text((code + ".json").toStdString());
        try {
            const std::optional<std::string> text = platform::read_file(file);
            if (!text) {
                throw std::runtime_error("unreadable");
            }
            const nlohmann::json data = nlohmann::json::parse(*text);
            if (!data.is_object()) {
                throw std::runtime_error("not an object");
            }
            if (const auto strings = data.find("strings"); strings != data.end() && strings->is_object()) {
                for (auto item = strings->begin(); item != strings->end(); ++item) {
                    if (item->is_string() && !item->get_ref<const std::string&>().empty()) {
                        chosen.strings.insert(QString::fromStdString(item.key()),
                                              QString::fromStdString(item->get<std::string>()));
                    }
                }
            }
            if (const auto plurals = data.find("plurals"); plurals != data.end() && plurals->is_object()) {
                for (auto item = plurals->begin(); item != plurals->end(); ++item) {
                    if (!item->is_array() || item->empty()) {
                        continue;
                    }
                    QStringList forms;
                    for (const nlohmann::json& form : *item) {
                        forms.append(form.is_string() ? QString::fromStdString(form.get<std::string>()) : QString());
                    }
                    chosen.plurals.insert(QString::fromStdString(item.key()), forms);
                }
            }
        } catch (const std::exception&) {
            chosen = Catalog();  // English
        }
    }
    catalog = std::move(chosen);
    const std::lock_guard lock(translated_mutex);
    translated_lines.clear();
    return catalog.language;
}

QString language() { return catalog.language; }

QTranslator* install_qt_translations(const QString& code) {
    if (code == "en") {
        return nullptr;
    }
    auto translator = std::make_unique<QTranslator>(QCoreApplication::instance());
    if (!translator->load("qtbase_" + code, QLibraryInfo::path(QLibraryInfo::TranslationsPath))) {
        return nullptr;
    }
    QCoreApplication::installTranslator(translator.get());
    return translator.release();
}

InEnglish::InEnglish() : before_(english_here) { english_here = true; }

InEnglish::~InEnglish() { english_here = before_; }

const std::vector<QString>& message_templates() {
    static const std::vector<QString> all = [] {
        std::vector<QString> out;
        for (const char* shape : templates) {
            out.push_back(QString::fromUtf8(shape));
        }
        return out;
    }();
    return all;
}

}  // namespace vml::app
