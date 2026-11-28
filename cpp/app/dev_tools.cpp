#include "dev_tools.h"

#include <QAbstractButton>
#include <QAbstractItemModel>
#include <QAbstractItemView>
#include <QAbstractSpinBox>
#include <QApplication>
#include <QComboBox>
#include <QGroupBox>
#include <QHeaderView>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QPainter>
#include <QPlainTextEdit>
#include <QTabBar>
#include <QTextEdit>

#include <algorithm>
#include <utility>

#include "i18n.h"
#include "theme.h"
#include "vml/app_log.h"

namespace vml::app::dev {

namespace {

constexpr int rows_listed = 50;  // of a table or list, and items of a combo box

std::string text_of(const QString& text) { return text.toStdString(); }

nlohmann::json rect_json(const QRect& rect) { return {rect.x(), rect.y(), rect.width(), rect.height()}; }

// A model's headings and first rows, as shown.
void describe_view(const QAbstractItemView& view, nlohmann::json& out) {
    const QAbstractItemModel* model = view.model();
    if (model == nullptr) {
        return;
    }
    const int rows = model->rowCount();
    const int columns = model->columnCount();
    out["rows"] = rows;
    out["columns"] = columns;
    nlohmann::json headings = nlohmann::json::array();
    for (int column = 0; column < columns; ++column) {
        headings.push_back(text_of(model->headerData(column, Qt::Horizontal).toString()));
    }
    out["headings"] = std::move(headings);
    nlohmann::json cells = nlohmann::json::array();
    for (int row = 0; row < std::min(rows, rows_listed); ++row) {
        nlohmann::json line = nlohmann::json::array();
        for (int column = 0; column < columns; ++column) {
            line.push_back(text_of(model->index(row, column).data().toString()));
        }
        cells.push_back(std::move(line));
    }
    out["cells"] = std::move(cells);
    if (const QModelIndex current = view.currentIndex(); current.isValid()) {
        out["current"] = {current.row(), current.column()};
    }
}

// A control's own state: its text, choice or tick, whichever it has.
void describe_state(const QWidget& widget, nlohmann::json& out) {
    if (const auto* button = qobject_cast<const QAbstractButton*>(&widget)) {
        out["text"] = text_of(button->text());
        if (button->isCheckable()) {
            out["checked"] = button->isChecked();
        }
    } else if (const auto* label = qobject_cast<const QLabel*>(&widget)) {
        out["text"] = text_of(label->text());
    } else if (const auto* line = qobject_cast<const QLineEdit*>(&widget)) {
        out["text"] = text_of(line->text());
        if (!line->placeholderText().isEmpty()) {
            out["placeholder"] = text_of(line->placeholderText());
        }
    } else if (const auto* combo = qobject_cast<const QComboBox*>(&widget)) {
        out["text"] = text_of(combo->currentText());
        nlohmann::json items = nlohmann::json::array();
        for (int index = 0; index < std::min(combo->count(), rows_listed); ++index) {
            items.push_back(text_of(combo->itemText(index)));
        }
        out["items"] = std::move(items);
    } else if (const auto* spin = qobject_cast<const QAbstractSpinBox*>(&widget)) {
        out["text"] = text_of(spin->text());
    } else if (const auto* tabs = qobject_cast<const QTabBar*>(&widget)) {
        nlohmann::json names = nlohmann::json::array();
        for (int index = 0; index < tabs->count(); ++index) {
            names.push_back(text_of(tabs->tabText(index)));
        }
        out["tabs"] = std::move(names);
        out["current"] = tabs->currentIndex();
    } else if (const auto* box = qobject_cast<const QGroupBox*>(&widget)) {
        out["text"] = text_of(box->title());
        if (box->isCheckable()) {
            out["checked"] = box->isChecked();
        }
    } else if (const auto* plain = qobject_cast<const QPlainTextEdit*>(&widget)) {
        out["text"] = text_of(plain->toPlainText().left(4000));
    } else if (const auto* rich = qobject_cast<const QTextEdit*>(&widget)) {
        out["text"] = text_of(rich->toPlainText().left(4000));
    }
    if (const auto* view = qobject_cast<const QAbstractItemView*>(&widget)) {
        describe_view(*view, out);
    }
}

// Whether a window is one to list: shown, and not the tooltip.
bool listed_window(const QWidget* window) {
    return window->isVisible() && !window->inherits("QTipLabel") && window->windowType() != Qt::ToolTip;
}

}  // namespace

bool enabled() {
    static const bool on = qEnvironmentVariable("VML_DEV") == QStringLiteral("1");
    return on;
}

const std::vector<Key>& keys() {
    static const std::vector<Key> all = [] {
        const auto with = [](int key) { return QKeySequence(Qt::CTRL | Qt::ALT | Qt::SHIFT | key); };
        return std::vector<Key>{
            {Command::dump_state, with(Qt::Key_F1), "Write the window's state to ui-state.json"},
            {Command::save_picture, with(Qt::Key_F2), "Save the window alone as window.png"},
            {Command::next_look, with(Qt::Key_F3), "Next look (nine, light and dark)"},
            {Command::tab_order, with(Qt::Key_F4), "Show or hide the Tab order"},
            {Command::names, with(Qt::Key_F5), "Show or hide each control's object name"},
            {Command::next_language, with(Qt::Key_F6), "Next language, then long text (restarts)"},
            {Command::forced_state, with(Qt::Key_F7),
             "Next faked run state: running, paused, cancelled, "
             "failed, FFmpeg not usable"},
        };
    }();
    return all;
}

std::filesystem::path output_folder() { return app_log::log_dir(); }

nlohmann::json describe_widget(const QWidget* widget, const QWidget* window) {
    nlohmann::json out;
    out["class"] = widget->metaObject()->className();
    if (!widget->objectName().isEmpty()) {
        out["name"] = text_of(widget->objectName());
    }
    out["geometry"] = rect_json(QRect(widget->mapTo(window, QPoint(0, 0)), widget->size()));
    if (!widget->isEnabled()) {
        out["enabled"] = false;
    }
    if (widget == QApplication::focusWidget()) {  // not a control whose focus proxy has it
        out["focus"] = true;
    }
    if (const QVariant role = widget->property("vmlRole"); role.isValid()) {
        out["role"] = text_of(role.toString());
    }
    if (!widget->toolTip().isEmpty()) {
        out["tooltip"] = text_of(widget->toolTip());
    }
    describe_state(*widget, out);
    nlohmann::json children = nlohmann::json::array();
    for (const QObject* child : widget->children()) {
        const auto* shown = qobject_cast<const QWidget*>(child);
        if (shown != nullptr && !shown->isWindow() && shown->isVisible()) {
            children.push_back(describe_widget(shown, window));
        }
    }
    if (!children.empty()) {
        out["children"] = std::move(children);
    }
    return out;
}

nlohmann::json describe(QWidget* main_window) {
    nlohmann::json out;
    out["interface"] = theme::current_interface() == theme::Interface::modern ? "modern" : "classic";
    out["look"] = text_of(theme::look());
    out["dark"] = theme::is_dark();
    out["language"] = text_of(language());
    // Geometry is in the window's own (logical) pixels: a capture's are
    // this many times as many.
    out["device_pixel_ratio"] = main_window->devicePixelRatioF();
    if (const QWidget* focus = QApplication::focusWidget()) {
        nlohmann::json path = nlohmann::json::array();  // from the window down
        for (const QWidget* at = focus; at != nullptr; at = at->isWindow() ? nullptr : at->parentWidget()) {
            path.insert(path.begin(), at->objectName().isEmpty() ? std::string(at->metaObject()->className())
                                                                 : text_of(at->objectName()));
        }
        out["focus"] = std::move(path);
    }
    std::vector<QWidget*> windows{main_window};
    for (QWidget* window : QApplication::topLevelWidgets()) {
        if (window != main_window && listed_window(window)) {
            windows.push_back(window);
        }
    }
    nlohmann::json listed = nlohmann::json::array();
    for (QWidget* window : windows) {
        nlohmann::json entry = describe_widget(window, window);
        entry["title"] = text_of(window->windowTitle());
        entry["screen_geometry"] = rect_json(window->frameGeometry());
        entry["active"] = window->isActiveWindow();
        listed.push_back(std::move(entry));
    }
    out["windows"] = std::move(listed);
    return out;
}

std::vector<QWidget*> tab_stops(QWidget* window) {
    // As QWidget::focusNextPrevChild walks the chain: a control with a
    // focus proxy is reached as its proxy.
    std::vector<QWidget*> stops;
    for (QWidget* widget = window->nextInFocusChain(); widget != nullptr && widget != window;
         widget = widget->nextInFocusChain()) {
        if ((widget->focusPolicy() & Qt::TabFocus) != 0 && widget->focusProxy() == nullptr && widget->isVisible() &&
            widget->isEnabled() && widget->window() == window) {
            stops.push_back(widget);
        }
    }
    return stops;
}

Overlay::Overlay(QWidget* window) : QWidget(window) {
    setAttribute(Qt::WA_TransparentForMouseEvents);
    setAttribute(Qt::WA_NoSystemBackground);
    setFocusPolicy(Qt::NoFocus);
    setObjectName(QStringLiteral("devOverlay"));
    hide();
    refresh_.setInterval(300);
    connect(&refresh_, &QTimer::timeout, this, [this] { follow_window(); });
}

void Overlay::set_mode(Mode mode) {
    mode_ = mode;
    if (mode == Mode::none) {
        refresh_.stop();
        hide();
        return;
    }
    follow_window();
    show();
    refresh_.start();
}

void Overlay::follow_window() {
    setGeometry(parentWidget()->rect());
    raise();
    update();
}

void Overlay::paintEvent(QPaintEvent* /*event*/) {
    QPainter painter(this);
    QWidget* window = parentWidget();
    QFont small = font();
    small.setPixelSize(11);
    small.setBold(true);
    painter.setFont(small);
    const QFontMetrics metrics(small);
    const auto badge = [&](const QPoint& at, const QString& text, const QColor& fill) {
        const QRect box(at, QSize(metrics.horizontalAdvance(text) + 6, metrics.height() + 2));
        painter.fillRect(box, fill);
        painter.setPen(Qt::white);
        painter.drawText(box, Qt::AlignCenter, text);
    };
    if (mode_ == Mode::tab_order) {
        int number = 1;
        const QColor fill(200, 0, 120);
        for (QWidget* stop : tab_stops(window)) {
            const QRect area(stop->mapTo(window, QPoint(0, 0)), stop->size());
            painter.setPen(QPen(fill, 1));
            painter.drawRect(area.adjusted(0, 0, -1, -1));
            badge(area.topLeft(), QString::number(number++), stop->hasFocus() ? QColor(0, 140, 60) : fill);
        }
    } else if (mode_ == Mode::names) {
        const std::vector<QColor> colours{{0, 120, 215}, {200, 80, 0}, {0, 150, 90}, {150, 0, 170}};
        std::function<void(QWidget*, int)> walk = [&](QWidget* widget, int depth) {
            for (QObject* child : widget->children()) {
                auto* shown = qobject_cast<QWidget*>(child);
                if (shown == nullptr || shown == this || shown->isWindow() || !shown->isVisible()) {
                    continue;
                }
                const QString name = shown->objectName();
                if (!name.isEmpty() && !name.startsWith(QStringLiteral("qt_"))) {
                    const QColor colour = colours[static_cast<std::size_t>(depth) % colours.size()];
                    const QRect area(shown->mapTo(window, QPoint(0, 0)), shown->size());
                    painter.setPen(QPen(colour, 1));
                    painter.drawRect(area.adjusted(0, 0, -1, -1));
                    badge(area.topLeft(), name, colour);
                }
                walk(shown, depth + 1);
            }
        };
        walk(window, 0);
    }
}

namespace {

// keys()' presses, from anywhere in the application.
class KeyCatcher : public QObject {
public:
    KeyCatcher(QObject* owner, std::function<void(Command)> run) : QObject(owner), run_(std::move(run)) {}

protected:
    bool eventFilter(QObject* /*watched*/, QEvent* event) override {
        if (event->type() != QEvent::ShortcutOverride && event->type() != QEvent::KeyPress) {
            return false;
        }
        auto* key = static_cast<QKeyEvent*>(event);
        const QKeySequence pressed(key->keyCombination());
        for (const Key& entry : keys()) {
            if (entry.keys == pressed) {
                if (event->type() == QEvent::ShortcutOverride) {
                    key->accept();  // to no shortcut: the press comes here next
                    return true;
                }
                // Taken where it is first seen (the window, then the focused
                // control): kept from every other, so run once.
                app_log::write(app_log::Level::info, "vml.dev", std::string("Developer key: ") + entry.what);
                QTimer::singleShot(0, parent(), [run = run_, command = entry.command] { run(command); });
                return true;
            }
        }
        return false;
    }

private:
    std::function<void(Command)> run_;
};

}  // namespace

void install(QObject* owner, std::function<void(Command)> run) {
    static bool installed = false;
    if (!enabled() || installed) {
        return;
    }
    installed = true;
    qApp->installEventFilter(new KeyCatcher(owner, std::move(run)));
}

}  // namespace vml::app::dev
