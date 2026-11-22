#include "video_names.h"

#include <algorithm>

#include "vml/settings.h"

namespace vml::app {

namespace {

const QChar ellipsis(0x2026);

// Where a name's words meet: what a name is cut at, so a word is left out
// whole or not at all.
bool separates(QChar c) { return c.isSpace() || QStringLiteral("-_.,+()[]{}").contains(c); }

// Fewer characters than this left out is not worth the "…".
constexpr qsizetype least_left_out = 4;

std::vector<QString> differing(const std::vector<QString>& names) {
    const QString& first = names.empty() ? QString() : names.front();
    if (std::all_of(names.begin(), names.end(), [&first](const QString& name) { return name == first; })) {
        return names;  // one video, or nothing tells them apart
    }
    qsizetype shortest = first.size();
    for (const QString& name : names) {
        shortest = std::min(shortest, name.size());
    }
    qsizetype start = shortest;  // the start they share
    qsizetype end = shortest;    // the end they share
    for (const QString& name : names) {
        qsizetype same = 0;
        while (same < start && name[same] == first[same]) {
            ++same;
        }
        start = same;
        same = 0;
        while (same < end && name[name.size() - 1 - same] == first[first.size() - 1 - same]) {
            ++same;
        }
        end = same;
    }
    end = std::min(end, shortest - start);
    // Back to the start of the word the first difference is in, then one
    // word more: "…CRF 20", not "…20".
    while (start > 0 && !separates(first[start - 1])) {
        --start;
    }
    while (start > 0 && separates(first[start - 1])) {
        --start;
    }
    while (start > 0 && !separates(first[start - 1])) {
        --start;
    }
    // The shared end from a word's edge.
    while (end > 0 && !separates(first[first.size() - end])) {
        --end;
    }
    if (start < least_left_out) {
        start = 0;
    }
    if (end < least_left_out) {
        end = 0;
    }
    if (start == 0 && end == 0) {
        return names;
    }
    std::vector<QString> shown;
    shown.reserve(names.size());
    for (const QString& name : names) {
        const QString middle = name.mid(start, name.size() - start - end).trimmed();
        if (middle.isEmpty()) {
            return names;  // a name of only separators past the start shared
        }
        shown.push_back((start > 0 ? QString(ellipsis) : QString()) + middle +
                        (end > 0 ? QString(ellipsis) : QString()));
    }
    return shown;
}

QString ends(const QString& name, int start, int end) {
    if (name.size() <= start + end + 1) {
        return name;
    }
    qsizetype head = start;
    if (head > 0 && name[head - 1].isHighSurrogate()) {
        ++head;  // the whole character
    }
    qsizetype tail = name.size() - end;
    if (tail < name.size() && name[tail].isLowSurrogate()) {
        --tail;
    }
    return name.left(head).trimmed() + ellipsis + name.mid(tail).trimmed();
}

}  // namespace

VideoNameRule video_name_rule(const Settings& settings) {
    VideoNameRule rule;
    rule.kind = settings.video_names == "full"   ? VideoNameRule::Kind::full
                : settings.video_names == "ends" ? VideoNameRule::Kind::ends
                                                 : VideoNameRule::Kind::differs;
    rule.start = std::clamp(settings.video_name_start, 0, 200);
    rule.end = std::clamp(settings.video_name_end, 0, 200);
    if (rule.kind == VideoNameRule::Kind::ends && rule.start == 0 && rule.end == 0) {
        rule.kind = VideoNameRule::Kind::full;
    }
    return rule;
}

std::vector<QString> shown_video_names(const std::vector<QString>& names, const VideoNameRule& rule) {
    switch (rule.kind) {
    case VideoNameRule::Kind::differs:
        return differing(names);
    case VideoNameRule::Kind::ends: {
        std::vector<QString> shown;
        shown.reserve(names.size());
        for (const QString& name : names) {
            shown.push_back(ends(name, rule.start, rule.end));
        }
        return shown;
    }
    case VideoNameRule::Kind::full:
        break;
    }
    return names;
}

}  // namespace vml::app
