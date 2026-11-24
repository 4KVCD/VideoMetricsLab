#include "quick_save.h"

#include <QContextMenuEvent>
#include <QEvent>
#include <QImage>
#include <QMenu>
#include <QPainter>
#include <QPainterPath>
#include <QStyle>
#include <QStyleOptionButton>

#include <algorithm>
#include <memory>

#include "file_writes.h"
#include "i18n.h"
#include "theme.h"
#include "vml/export_csv.h"
#include "vml/settings.h"
#include "window_helpers.h"

namespace vml::app {

std::vector<std::string> quick_formats(const Settings& settings) {
    std::vector<std::string> formats;
    for (const std::string& key : export_format_keys()) {
        if (std::find(settings.quick_save_formats.begin(), settings.quick_save_formats.end(), key) !=
            settings.quick_save_formats.end()) {
            formats.push_back(key);
        }
    }
    return formats;
}

QString formats_text(const std::vector<std::string>& formats) {
    QStringList names;
    for (const std::string& format : formats) {
        names.append(QString::fromStdString(format).toUpper());
    }
    if (names.size() == 2) {
        return app::tr("{first} and {second}", {{"first", names[0]}, {"second", names[1]}});
    }
    return names.join(QStringLiteral(", "));
}

QString no_format_text() {
    return app::tr("Nothing to save: no format is ticked in Settings > Storage > Quick saves.");
}

QString nothing_shown_text() { return app::tr("Nothing to save yet: this card shows nothing."); }

void not_exported(QWidget* view) { view->setProperty("vmlNotExported", true); }

bool is_not_exported(const QWidget* view) { return view->property("vmlNotExported").toBool(); }

std::vector<QuickFile> data_files(const QString& stem, const std::vector<std::string>& formats,
                                  const std::function<std::string(const std::string& format)>& text) {
    std::vector<QuickFile> files;
    for (const std::string& format : formats) {
        files.push_back({suggested_file_name(stem, "." + QString::fromStdString(format)),
                         [made = text(format)](const QString& path) { return write_text_file(path, made); }});
    }
    return files;
}

QuickFile image_file(const QString& name, const std::function<QPixmap()>& image) {
    // A QImage, which another thread may save; a QPixmap is the window's own.
    return {name, [picture = image().toImage()](const QString& path) {
                return picture.save(path) ? QString() : could_not_write(path);
            }};
}

QuickSaveRunner written_by(FileWrites& writes) {
    return [&writes](std::function<ExportOutcome()> save, ExportReport done) {
        auto outcome = std::make_shared<ExportOutcome>();
        writes.submit(
            QStringLiteral("quick save"), [save = std::move(save), outcome] { *outcome = save(); },
            [outcome, done = std::move(done)](const QString& error) {
                if (!error.isEmpty()) {
                    outcome->error = tr_message(error);
                    outcome->details = outcome->error;
                }
                done(*outcome);
            });
    };
}

void QuickSaves::save(const QString& into, std::vector<QuickFile> files, const ExportReport& done) const {
    if (files.empty()) {
        return;
    }
    std::function<ExportOutcome()> write = [into, files = std::move(files)] { return save_quick(into, files); };
    const ExportReport report = [done](const ExportOutcome& outcome) { report_to(done, outcome); };
    if (background) {
        background(std::move(write), report);
    } else {
        report(write());
    }
}

QString quick_save_tooltip(const QString& what, const std::vector<std::string>& formats) {
    const QString saves =
        formats.empty()
            ? app::tr("Save {what} in the export folder.", {{"what", what}})
            : app::tr("Save {what}, as {formats}, in the export folder.", {{"what", what}, {"formats", formats_text(formats)}});
    return saves + QStringLiteral("\n") + app::tr("Right-click: Save as...");
}

std::vector<QString> quick_save_paths(const QString& folder, const std::vector<QString>& names,
                                      std::set<std::filesystem::path>& reserved) {
    std::vector<std::string> wanted;
    for (const QString& name : names) {
        wanted.push_back(name.toStdString());
    }
    std::vector<QString> paths;
    for (const std::filesystem::path& path : unique_output_names(path_of(folder), wanted, reserved)) {
        paths.push_back(text_of(path));
    }
    return paths;
}

ExportOutcome save_quick(const QString& folder, const std::vector<QuickFile>& files) {
    ExportOutcome outcome;
    if (files.empty()) {
        return outcome;
    }
    std::error_code ignored;
    std::filesystem::create_directories(path_of(folder), ignored);  // a folder chosen since deleted: made again
    std::vector<QString> names;
    for (const QuickFile& file : files) {
        names.push_back(file.name);
    }
    std::set<std::filesystem::path> reserved;
    const std::vector<QString> paths = quick_save_paths(folder, names, reserved);
    for (std::size_t index = 0; index < files.size(); ++index) {
        const QString error = files[index].write(paths[index]);
        if (!error.isEmpty()) {
            outcome.error = error;
            outcome.details = error;
            return outcome;
        }
        outcome.files.append(paths[index]);
    }
    return outcome;
}

// ------------------------------------------------------------ the save icon

SaveButton::SaveButton(QWidget* parent) : QPushButton(parent) {
    setFocusPolicy(Qt::TabFocus);
    theme::set_role(this, "icon");
    setAccessibleName(app::tr("Save"));
    connect(this, &QPushButton::clicked, this, [this] {
        refresh();  // what it saves may have changed since it was last looked at
        if (isEnabled() && save) {
            save();
        }
    });
}

void SaveButton::refresh() {
    const QString why = why_not ? why_not() : QString();
    setEnabled(why.isEmpty());
    setToolTip(!why.isEmpty() ? why : describe ? describe() : QString());
}

QString SaveButton::summary() const {
    const QString why = why_not ? why_not() : QString();
    const QString text = !why.isEmpty() ? why : describe ? describe() : QString();
    return text.section(QLatin1Char('\n'), 0, 0);
}

QSize SaveButton::sizeHint() const {
    // VML Modern: as its style sheet makes it (the "icon" role); Classic: a
    // small square, not a push button's usual width.
    if (theme::current_interface() == theme::Interface::modern) {
        return QPushButton::sizeHint();
    }
    return {28, 26};
}

QSize SaveButton::minimumSizeHint() const { return sizeHint(); }

bool SaveButton::event(QEvent* event) {
    // A disabled widget is still told of the pointer and asked its tooltip.
    if (event->type() == QEvent::Enter || event->type() == QEvent::ToolTip || event->type() == QEvent::Show) {
        refresh();
    }
    return QPushButton::event(event);
}

void SaveButton::contextMenuEvent(QContextMenuEvent* event) {
    if (!save_as || !isEnabled()) {
        return;
    }
    QMenu menu(this);
    menu.setToolTipsVisible(true);
    save_as(menu);
    if (!menu.isEmpty()) {
        menu.exec(event->globalPos());
    }
}

void SaveButton::paintEvent(QPaintEvent* event) {
    QPushButton::paintEvent(event);  // the button, without icon or text
    // A disk, drawn in the text's color (greyed when disabled), as the
    // title bar draws its glyphs: no icon file to keep in two themes.
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    const QColor ink = palette().color(isEnabled() ? QPalette::Active : QPalette::Disabled, QPalette::ButtonText);
    const qreal side = 14;
    // Centered in the button as drawn: VML Modern's style sheet gives it a
    // margin at the left (its gap from the button beside it).
    QStyleOptionButton option;
    initStyleOption(&option);
    const QRectF drawn = style()->subElementRect(QStyle::SE_PushButtonContents, &option, this);
    const QRectF box(drawn.center().x() - side / 2.0, drawn.center().y() - side / 2.0, side, side);
    QPainterPath outline;
    const qreal cut = 3.5;
    outline.moveTo(box.left() + 1.5, box.top());
    outline.lineTo(box.right() - cut, box.top());
    outline.lineTo(box.right(), box.top() + cut);
    outline.lineTo(box.right(), box.bottom() - 1.5);
    outline.quadTo(box.right(), box.bottom(), box.right() - 1.5, box.bottom());
    outline.lineTo(box.left() + 1.5, box.bottom());
    outline.quadTo(box.left(), box.bottom(), box.left(), box.bottom() - 1.5);
    outline.lineTo(box.left(), box.top() + 1.5);
    outline.quadTo(box.left(), box.top(), box.left() + 1.5, box.top());
    QPen pen(ink, 1.3);
    pen.setJoinStyle(Qt::RoundJoin);
    painter.setPen(pen);
    painter.setBrush(Qt::NoBrush);
    painter.drawPath(outline);
    // The shutter at the top, the label at the bottom.
    painter.drawRect(QRectF(box.left() + 3.5, box.top(), side - 8, 4.5));
    painter.drawRect(QRectF(box.left() + 3, box.top() + side * 0.55, side - 6, side * 0.45));
}

}  // namespace vml::app
