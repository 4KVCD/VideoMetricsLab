#include "saved_deletion.h"

#include <QDateTime>
#include <QLocale>

#include <algorithm>

#include "i18n.h"
#include "window_helpers.h"
#include "vml/number_format.h"

namespace vml::app {

SavedDeletion saved_deletion(const std::vector<SavedComparison>& comparisons, const std::filesystem::path& folder) {
    SavedDeletion deletion;
    deletion.folder = folder;
    for (const SavedComparison& saved : comparisons) {
        ++deletion.comparisons;
        deletion.bytes += saved.bytes;
        deletion.first = std::min(deletion.first.value_or(saved.saved), saved.saved);
        deletion.last = std::max(deletion.last.value_or(saved.saved), saved.saved);
        deletion.without_reference += saved.reference_found ? 0 : 1;
    }
    return deletion;
}

QString saved_size_text(std::uintmax_t bytes) {
    // A comparison's scores are often under a megabyte (shown_size).
    const ShownSize size = shown_size(bytes);
    return size.megabytes ? app::tr("{size:.1f} MB", {{"size", size.value}})
                          : app::tr("{size:.0f} KB", {{"size", size.value}});
}

QString saved_date_text(std::chrono::system_clock::time_point when) {
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(when.time_since_epoch()).count();
    return QLocale().toString(QDateTime::fromSecsSinceEpoch(seconds).date(), QLocale::ShortFormat);
}

Message saved_deletion_message(const SavedDeletion& deletion) {
    Message message;
    message.kind = MessageKind::danger;
    message.title = app::tr("Delete saved results");
    message.headline = ntr("The saved scores of {count} comparison are deleted. This can't be undone.",
                           "The saved scores of {count} comparisons are deleted. This can't be undone.",
                           deletion.comparisons);
    QStringList text;
    const QString first = deletion.first ? saved_date_text(*deletion.first) : QString();
    const QString last = deletion.last ? saved_date_text(*deletion.last) : QString();
    text << (first == last ? app::tr("{size}, last saved {date}, in {directory}.",
                                     {{"size", saved_size_text(deletion.bytes)},
                                      {"date", first},
                                      {"directory", text_of(deletion.folder)}})
                           : app::tr("{size}, last saved {first} to {last}, in {directory}.",
                                     {{"size", saved_size_text(deletion.bytes)},
                                      {"first", first},
                                      {"last", last},
                                      {"directory", text_of(deletion.folder)}}));
    text << app::tr("Their videos are scored again the next time they are compared.");
    if (deletion.without_reference > 0) {
        text << ntr("{count} of them can't be scored again: its reference video is no longer where it was scored "
                    "from (marked \"not found\" in the Saved Results tab).",
                    "{count} of them can't be scored again: their reference videos are no longer where they were "
                    "scored from (marked \"not found\" in the Saved Results tab).",
                    deletion.without_reference);
    }
    if (deletion.everything) {
        text << app::tr("The bitrate readings and black-bar answers kept with them are deleted too.");
    }
    message.text = text.join(QStringLiteral("\n\n"));
    message.buttons = {{app::tr("Delete saved results"), MessageButton::Role::danger},
                       {app::tr("Cancel"), MessageButton::Role::plain}};
    message.default_button = 1;
    message.escape_button = 1;
    return message;
}

}  // namespace vml::app
