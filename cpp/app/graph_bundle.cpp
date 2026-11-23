#include "graph_bundle.h"

#include <QBuffer>
#include <QByteArray>
#include <QPixmap>

#include <algorithm>
#include <map>
#include <optional>

#include "bitrate_panel.h"
#include "graph_panel.h"
#include "i18n.h"
#include "vml/metrics.h"
#include "vml/python_text.h"
#include "window_helpers.h"

namespace vml::app {

namespace {

// A pixmap as a PNG file's bytes.
std::string png_bytes(const QPixmap& image) {
    QByteArray bytes;
    QBuffer buffer(&bytes);
    buffer.open(QIODevice::WriteOnly);
    image.save(&buffer, "PNG");
    return bytes.toStdString();
}

// A file name in the archive: `stem` and " - `what`.png", as UTF-8.
std::string png_name(const QString& stem, const QString& what) {
    return file_safe(stem + QStringLiteral(" - ") + what).toStdString() + ".png";
}

// The name every video's files share in an archive of several.
QString all_videos() { return app::tr("All test videos"); }

}  // namespace

QString file_safe(const QString& text) {
    QString safe = text;
    for (const QChar c : QStringLiteral("\\/:*?\"<>|")) {
        safe.replace(c, QLatin1Char('_'));
    }
    return safe.trimmed();
}

std::vector<QString> bundle_names(const std::vector<BundleVideo>& videos) {
    std::vector<QString> names;
    std::map<QString, int> uses;
    for (const BundleVideo& video : videos) {
        names.push_back(video.label.isEmpty() ? text_of(video.path.stem()) : video.label);
        ++uses[names.back()];
    }
    // Where several share a name: each with its folder's.
    for (std::size_t i = 0; i < names.size(); ++i) {
        if (uses[names[i]] > 1) {
            names[i] += QStringLiteral(" (") + name_of(videos[i].path.parent_path()) + QStringLiteral(")");
        }
    }
    // Where those are shared too: numbered from the second on.
    std::map<QString, int> seen;
    for (QString& name : names) {
        const int count = ++seen[name];
        if (count > 1) {
            name += QStringLiteral(" ") + QString::number(count);
        }
    }
    return names;
}

std::vector<BundleFile> metric_graph_files(const std::vector<BundleVideo>& videos, const BundleStyle& style,
                                           bool together) {
    std::vector<BundleFile> files;
    if (videos.empty()) {
        return files;
    }
    const std::vector<QString> names = bundle_names(videos);
    // A panel of its own, never shown: a line per video, under an identity
    // of the export's own.
    GraphPanel panel(style.colors);
    panel.export_colours = [colours = style.colours] { return colours; };
    std::vector<GraphPanel::Run> runs;
    for (std::size_t i = 0; i < videos.size(); ++i) {
        runs.push_back({videos[i].result, names[i], "bundle:" + std::to_string(i), true});
    }
    panel.add_runs(runs);
    const std::vector<std::string> keys = panel.scored_metrics();
    for (const std::string& key : keys) {
        const QString metric = qt_text(metric_definition(key).label);
        for (std::size_t i = 0; i < videos.size(); ++i) {
            if (videos[i].result->has_metric(key)) {
                files.emplace_back(png_name(names[i], metric),
                                   png_bytes(panel.render_export_image(key, {runs[i].identity}, style.size)));
            }
        }
    }
    if (together && videos.size() > 1) {
        for (const std::string& key : keys) {
            files.emplace_back(png_name(all_videos(), qt_text(metric_definition(key).label)),
                               png_bytes(panel.render_export_image(key, {}, style.size)));
        }
    }
    return files;
}

SavedBitrates saved_bitrates(std::vector<BundleVideo>& videos, const std::vector<std::filesystem::path>& folders,
                             const std::filesystem::path& reference, const OwnFile& own,
                             const LoadSavedBitrate& load) {
    SavedBitrates saved;
    if (videos.empty() || folders.size() != videos.size()) {
        return saved;
    }
    if (!own(folders.front(), true, reference)) {
        saved[reference] = load(folders.front(), true, reference);
    }
    for (std::size_t i = 0; i < videos.size(); ++i) {
        BundleVideo& video = videos[i];
        if (video.result->resample_target || own(folders[i], false, video.path)) {
            continue;  // a resolution test has no file of its own; the file there is this one's
        }
        // Its path and its folder's name, after a character no file name
        // has: no file is read by it, and none is drawn in its place.
        video.bitrate_key = video.path;
        video.bitrate_key += "|";
        video.bitrate_key += folders[i].filename();
        saved[video.bitrate_key] = load(folders[i], false, video.path);
    }
    return saved;
}

std::vector<VideoInfo> bitrate_files(const std::filesystem::path& reference, const std::vector<BundleVideo>& videos,
                                     const SavedBitrates& saved) {
    std::vector<VideoInfo> files;
    if (videos.empty()) {
        return files;
    }
    const auto add = [&files, &saved](const VideoInfo& info, const std::filesystem::path& path) {
        if (!saved.contains(path)) {
            files.push_back(info);
            files.back().path = path;
        }
    };
    add(videos.front().result->source_info, reference);
    for (const BundleVideo& video : videos) {
        if (!video.result->resample_target) {
            add(video.result->distorted_info, video.bitrate_path());
        }
    }
    return files;
}

std::vector<BundleFile> bitrate_graph_files(const BitratePanel& read, const std::filesystem::path& reference,
                                            const std::vector<BundleVideo>& videos, const BundleStyle& style,
                                            std::vector<QString>& missing, const SavedBitrates& saved) {
    std::vector<BundleFile> files;
    if (videos.empty()) {
        return files;
    }
    const std::vector<QString> names = bundle_names(videos);
    const auto is_read = [&read, &saved](const std::filesystem::path& path) {
        if (const auto found = saved.find(path); found != saved.end()) {
            return found->second.has_value();
        }
        return read.state_of(path) == BitrateState::complete;
    };
    if (!is_read(reference)) {
        missing.push_back(name_of(reference));
    }
    for (const BundleVideo& video : videos) {
        if (!is_read(video.bitrate_path())) {
            missing.push_back(name_of(video.path));
        }
    }
    // Drawn from `paths` (the reference first), in a panel of its own.
    const auto drawn = [&](const std::vector<std::filesystem::path>& paths) -> std::optional<std::string> {
        BitratePanel panel([] { return std::string(); }, style.colors);
        panel.export_colours = [colours = style.colours] { return colours; };
        panel.adopt_read(read, paths, saved);
        if (std::none_of(paths.begin(), paths.end(), is_read)) {
            return std::nullopt;
        }
        return png_bytes(panel.render_export_image(style.size));
    };
    const QString bitrate = app::tr("Bitrate");
    if (videos.size() > 1) {
        std::vector<std::filesystem::path> all{reference};
        for (const BundleVideo& video : videos) {
            all.push_back(video.bitrate_path());
        }
        if (const std::optional<std::string> png = drawn(all)) {
            files.emplace_back(png_name(all_videos(), bitrate), *png);
        }
    }
    for (std::size_t i = 0; i < videos.size(); ++i) {
        if (!is_read(videos[i].bitrate_path())) {
            continue;  // the reference alone says nothing of this video
        }
        if (const std::optional<std::string> png = drawn({reference, videos[i].bitrate_path()})) {
            files.emplace_back(png_name(names[i], bitrate), *png);
        }
    }
    return files;
}

}  // namespace vml::app
