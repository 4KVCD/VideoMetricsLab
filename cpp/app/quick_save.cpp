#include "quick_save.h"

#include <QContextMenuEvent>
#include <QEvent>
#include <QIconEngine>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QStylePainter>
#include <QImage>
#include <QPointer>
#include <QMenu>
#include <QPainter>
#include <QPainterPath>
#include <QStyle>
#include <QStyleOptionButton>

#include <algorithm>
#include <memory>

#include "file_writes.h"
#include "i18n.h"
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
    return saves + QStringLiteral("\n") + app::tr("Arrow: Save as...");
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

// ------------------------------------------------------------ the Export button's icon

namespace {

class ExportIconEngine : public QIconEngine {
public:
    explicit ExportIconEngine(QWidget* button) : button_(button) {}

    void paint(QPainter* painter, const QRect& rect, QIcon::Mode mode, QIcon::State) override {
        const QPalette colors = button_ != nullptr ? button_->palette() : QPalette();
        const QColor ink =
            colors.color(mode == QIcon::Disabled ? QPalette::Disabled : QPalette::Active, QPalette::ButtonText);
        painter->save();
        painter->setRenderHint(QPainter::Antialiasing);
        QPen pen(ink, std::max<qreal>(1.3, rect.height() / 11.0));
        pen.setCapStyle(Qt::RoundCap);
        pen.setJoinStyle(Qt::RoundJoin);
        painter->setPen(pen);
        painter->setBrush(Qt::NoBrush);
        // A square at the left of `rect`: the rest is room before the text.
        const QRectF box = QRectF(rect.left(), rect.top(), rect.height(), rect.height()).adjusted(1.5, 1.0, -1.5, -1.5);
        // The tray: open at the top, its sides a third of the height.
        const qreal tray = box.top() + box.height() * 0.62;
        QPainterPath outline;
        outline.moveTo(box.left(), tray);
        outline.lineTo(box.left(), box.bottom());
        outline.lineTo(box.right(), box.bottom());
        outline.lineTo(box.right(), tray);
        painter->drawPath(outline);
        // The arrow, rising out of it.
        const qreal middle = box.center().x();
        const qreal head = box.width() * 0.3;
        painter->drawLine(QPointF(middle, box.top()), QPointF(middle, box.top() + box.height() * 0.76));
        QPainterPath arrow;
        arrow.moveTo(middle - head, box.top() + head);
        arrow.lineTo(middle, box.top());
        arrow.lineTo(middle + head, box.top() + head);
        painter->drawPath(arrow);
        painter->restore();
    }

    QPixmap pixmap(const QSize& size, QIcon::Mode mode, QIcon::State state) override {
        return scaledPixmap(size, mode, state, 1.0);
    }

    QPixmap scaledPixmap(const QSize& size, QIcon::Mode mode, QIcon::State state, qreal scale) override {
        // Drawn at the screen's own pixels, sharp at any scaling.
        QPixmap drawn(size * scale);
        drawn.setDevicePixelRatio(scale);
        drawn.fill(Qt::transparent);
        QPainter painter(&drawn);
        paint(&painter, QRect(QPoint(0, 0), size), mode, state);
        return drawn;
    }

    QIconEngine* clone() const override { return new ExportIconEngine(button_); }

private:
    QPointer<QWidget> button_;
};

}  // namespace

QIcon export_icon(QWidget* button) { return QIcon(new ExportIconEngine(button)); }

// ------------------------------------------------------------ the save button

namespace {

// A save button's glyph (issue #261), drawn as the Export button's is: in
// the button's text color, greyed when disabled, a square at the left of
// the icon's size. A picture: a frame with hills and a sun. Data: a small
// table, its heading row and a column line.
class SaveGlyphEngine : public QIconEngine {
public:
    SaveGlyphEngine(QWidget* button, SaveKind kind) : button_(button), kind_(kind) {}

    void paint(QPainter* painter, const QRect& rect, QIcon::Mode mode, QIcon::State) override {
        const QPalette colors = button_ != nullptr ? button_->palette() : QPalette();
        const QColor ink =
            colors.color(mode == QIcon::Disabled ? QPalette::Disabled : QPalette::Active, QPalette::ButtonText);
        painter->save();
        painter->setRenderHint(QPainter::Antialiasing);
        QPen pen(ink, std::max<qreal>(1.3, rect.height() / 11.0));
        pen.setCapStyle(Qt::RoundCap);
        pen.setJoinStyle(Qt::RoundJoin);
        painter->setPen(pen);
        painter->setBrush(Qt::NoBrush);
        const QRectF box = QRectF(rect.left(), rect.top(), rect.height(), rect.height()).adjusted(1.5, 2.0, -1.5, -2.0);
        painter->drawRoundedRect(box, 1.5, 1.5);
        if (kind_ == SaveKind::picture) {
            // Solid hills and a sun, as a photo's icon: an outline hill read
            // as a line chart at 16 px.
            QPainterPath hills;
            hills.moveTo(box.left(), box.bottom());
            hills.lineTo(box.left(), box.bottom() - box.height() * 0.12);
            hills.lineTo(box.left() + box.width() * 0.36, box.top() + box.height() * 0.45);
            hills.lineTo(box.left() + box.width() * 0.6, box.top() + box.height() * 0.72);
            hills.lineTo(box.left() + box.width() * 0.74, box.top() + box.height() * 0.58);
            hills.lineTo(box.right(), box.bottom() - box.height() * 0.15);
            hills.lineTo(box.right(), box.bottom());
            hills.closeSubpath();
            painter->setBrush(ink);
            painter->drawPath(hills);
            painter->setPen(Qt::NoPen);
            painter->drawEllipse(QPointF(box.left() + box.width() * 0.7, box.top() + box.height() * 0.28),
                                 box.width() * 0.13, box.width() * 0.13);
        } else {
            const qreal heading = box.top() + box.height() * 0.33;
            painter->drawLine(QPointF(box.left(), heading), QPointF(box.right(), heading));
            const qreal row = box.top() + box.height() * 0.66;
            painter->drawLine(QPointF(box.left(), row), QPointF(box.right(), row));
            const qreal column = box.left() + box.width() * 0.4;
            painter->drawLine(QPointF(column, heading), QPointF(column, box.bottom()));
        }
        painter->restore();
    }

    QPixmap pixmap(const QSize& size, QIcon::Mode mode, QIcon::State state) override {
        return scaledPixmap(size, mode, state, 1.0);
    }

    QPixmap scaledPixmap(const QSize& size, QIcon::Mode mode, QIcon::State state, qreal scale) override {
        QPixmap drawn(size * scale);
        drawn.setDevicePixelRatio(scale);
        drawn.fill(Qt::transparent);
        QPainter painter(&drawn);
        paint(&painter, QRect(QPoint(0, 0), size), mode, state);
        return drawn;
    }

    QIconEngine* clone() const override { return new SaveGlyphEngine(button_, kind_); }

private:
    QPointer<QWidget> button_;
    SaveKind kind_;
};

}  // namespace

SaveButton::SaveButton(const QString& label, SaveKind kind, QWidget* parent) : QPushButton(parent), kind_(kind) {
    setFocusPolicy(Qt::TabFocus);
    setText(label);
    setIcon(QIcon(new SaveGlyphEngine(this, kind)));
    setIconSize(QSize(20, 16));  // the glyph is square at the left: the rest is room before the words
    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);  // down to its glyph and arrow where room is short
    setAccessibleName(label);
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
    setAccessibleDescription(summary());
}

QString SaveButton::summary() const {
    const QString why = why_not ? why_not() : QString();
    const QString text = !why.isEmpty() ? why : describe ? describe() : QString();
    return text.section(QLatin1Char('\n'), 0, 0);
}

int SaveButton::arrow_width() const { return fontMetrics().height() + 4; }

QRect SaveButton::arrow_rect() const { return {width() - arrow_width(), 0, arrow_width(), height()}; }

QSize SaveButton::sizeHint() const {
    // The words and glyph as a push button sizes them, and the arrow.
    const QSize words = QPushButton::sizeHint();
    return {words.width() + arrow_width(), words.height()};
}

QSize SaveButton::minimumSizeHint() const {
    // The glyph with a little room each side, and the arrow: no wider than
    // the icon-only button it replaces needs, with its arrow.
    return {glyph_only_width + arrow_width(), sizeHint().height()};
}

bool SaveButton::shows_words() const {
    // Its side padding gives way before its words do: a row a few pixels
    // short keeps them.
    return width() >= sizeHint().width() - padding_given_up;
}

bool SaveButton::event(QEvent* event) {
    // A disabled widget is still told of the pointer and asked its tooltip.
    if (event->type() == QEvent::Enter || event->type() == QEvent::ToolTip || event->type() == QEvent::Show) {
        refresh();
    }
    return QPushButton::event(event);
}

void SaveButton::open_menu() {
    if (!save_as || !isEnabled()) {
        return;
    }
    QMenu menu(this);
    menu.setToolTipsVisible(true);
    save_as(menu);
    if (!menu.isEmpty()) {
        setDown(true);
        menu.exec(mapToGlobal(QPoint(0, height())));
        setDown(false);
    }
}

void SaveButton::contextMenuEvent(QContextMenuEvent* event) {
    event->accept();
    open_menu();
}

void SaveButton::mousePressEvent(QMouseEvent* event) {
    // The arrow opens Save as... on the press, as a menu button does.
    if (event->button() == Qt::LeftButton && arrow_rect().contains(event->position().toPoint())) {
        event->accept();
        open_menu();
        return;
    }
    QPushButton::mousePressEvent(event);
}

void SaveButton::keyPressEvent(QKeyEvent* event) {
    // Alt+Down, as a drop-down opens its list; the menu key and Shift+F10
    // come as a context menu (contextMenuEvent).
    if (event->key() == Qt::Key_Down && (event->modifiers() & Qt::AltModifier)) {
        open_menu();
        return;
    }
    QPushButton::keyPressEvent(event);
}

void SaveButton::paintEvent(QPaintEvent* /*event*/) {
    // The button, its glyph and words left of the arrow; then the divider
    // and the arrow, in the text's color (greyed when disabled).
    QStylePainter painter(this);
    QStyleOptionButton option;
    initStyleOption(&option);
    painter.drawControl(QStyle::CE_PushButtonBevel, option);
    QStyleOptionButton words = option;
    words.rect = style()->subElementRect(QStyle::SE_PushButtonContents, &option, this);
    words.rect.setRight(arrow_rect().left() - 1);
    if (shows_words() && width() < sizeHint().width()) {
        // Short of its full width: the words take the padding's room.
        words.rect = QRect(4, 0, arrow_rect().left() - 4, height());
    }
    if (!shows_words()) {
        // The glyph alone, centered left of the arrow; the words stay its
        // tooltip's and accessible name.
        words.text.clear();
        words.iconSize = QSize(16, 16);
        words.rect = QRect(0, 0, arrow_rect().left(), height());
    }
    painter.drawControl(QStyle::CE_PushButtonLabel, words);
    const QColor ink = palette().color(isEnabled() ? QPalette::Active : QPalette::Disabled, QPalette::ButtonText);
    painter.setRenderHint(QPainter::Antialiasing);
    const QRect arrow = arrow_rect();
    QColor line = ink;
    line.setAlphaF(0.35F);
    painter.setPen(QPen(line, 1));
    painter.drawLine(QPointF(arrow.left() + 0.5, arrow.top() + 6), QPointF(arrow.left() + 0.5, arrow.bottom() - 5));
    QPen chevron(ink, 1.4);
    chevron.setCapStyle(Qt::RoundCap);
    chevron.setJoinStyle(Qt::RoundJoin);
    painter.setPen(chevron);
    const QPointF middle(arrow.center().x() + 1.0, arrow.center().y() + 1.0);
    QPainterPath path;
    path.moveTo(middle + QPointF(-3.5, -1.75));
    path.lineTo(middle + QPointF(0, 1.75));
    path.lineTo(middle + QPointF(3.5, -1.75));
    painter.drawPath(path);
}

}  // namespace vml::app
