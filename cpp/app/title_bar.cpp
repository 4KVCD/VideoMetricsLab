#include "title_bar.h"

#include "i18n.h"
#include "theme.h"

#include <QAbstractButton>
#include <QActionGroup>
#include <QApplication>
#include <QCursor>
#include <QEvent>
#include <QFontDatabase>
#include <QHBoxLayout>
#include <QLabel>
#include <QMainWindow>
#include <QMenu>
#include <QPainter>
#include <QPainterPath>
#include <QStyleOptionToolButton>
#include <QStylePainter>
#include <QToolButton>
#include <QToolTip>
#include <QWindow>

#ifdef _WIN32
#include <windows.h>
#endif

#include <algorithm>
#include <functional>
#include <optional>

namespace vml::app {

namespace {

// The font Windows draws its title bar's glyphs in: Segoe Fluent Icons on
// Windows 11, Segoe MDL2 Assets on 10. Empty elsewhere.
QString glyph_font() {
#ifdef _WIN32
    static const QString family = QFontDatabase::hasFamily(QStringLiteral("Segoe Fluent Icons"))
                                      ? QStringLiteral("Segoe Fluent Icons")
                                      : QStringLiteral("Segoe MDL2 Assets");
    return family;
#else
    return {};
#endif
}

// `colour` at `alpha` (0 to 1): a hover's or a press's wash.
QColor faded(QColor colour, double alpha) {
    colour.setAlphaF(static_cast<float>(alpha));
    return colour;
}

// The progress button: its text, and under it a thin bar of how far the
// run has got, in the accent, or both in the error colour or muted. Its
// tooltip is asked for as it is about to show.
class ProgressButton : public QToolButton {
public:
    double fraction = 0.0;
    TitleButtons::Progress::Tone tone = TitleButtons::Progress::Tone::normal;
    std::function<void()> tooltip_wanted;

protected:
    bool event(QEvent* event) override {
        if (event->type() == QEvent::ToolTip && tooltip_wanted) {
            tooltip_wanted();
        }
        return QToolButton::event(event);
    }
    void paintEvent(QPaintEvent*) override {
        // QToolButton's own, its text in the tone's colour.
        const std::optional<QColor> colour = tone_colour();
        QStylePainter painter(this);
        QStyleOptionToolButton option;
        initStyleOption(&option);
        if (colour) {
            for (const QPalette::ColorRole role : {QPalette::ButtonText, QPalette::WindowText, QPalette::Text}) {
                option.palette.setColor(role, *colour);
            }
        }
        painter.drawComplexControl(QStyle::CC_ToolButton, option);
        const QRect track(6, height() - 4, width() - 12, 2);
        painter.fillRect(track, faded(palette().windowText().color(), 0.15));
        painter.fillRect(QRect(track.left(), track.top(), static_cast<int>(track.width() * fraction), 2),
                         colour ? QBrush(*colour) : palette().highlight());
    }

private:
    std::optional<QColor> tone_colour() const {
        switch (tone) {
        case TitleButtons::Progress::Tone::error:
            return theme::color("failed");
        case TitleButtons::Progress::Tone::muted:
            return theme::color("muted");
        case TitleButtons::Progress::Tone::normal:
            break;
        }
        return std::nullopt;
    }
};

// The update's badge: its text on a pill in the accent.
class BadgeButton : public QToolButton {
protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        QColor fill = palette().highlight().color();
        if (underMouse()) {
            fill = fill.lighter(112);
        }
        const QRectF pill = QRectF(rect()).adjusted(2, 5, -2, -5);
        QPainterPath path;
        path.addRoundedRect(pill, pill.height() / 2, pill.height() / 2);
        painter.fillPath(path, fill);
        painter.setPen(theme::text_on(fill));
        painter.drawText(pill, Qt::AlignCenter, text());
    }
    QSize sizeHint() const override {
        return QSize(fontMetrics().horizontalAdvance(text()) + 24, 30);
    }
};

}  // namespace

// ================================================================== the buttons

TitleButtons::TitleButtons(QWidget* parent) : QWidget(parent) {
    auto* row = new QHBoxLayout(this);
    row->setContentsMargins(0, 0, 0, 0);
    row->setSpacing(4);

    auto* progress = new ProgressButton;
    progress_ = progress;
    progress_->setAutoRaise(true);
    progress_->setCursor(Qt::PointingHandCursor);
    progress->tooltip_wanted = [this] { emit progress_tooltip_wanted(); };
    connect(progress_, &QToolButton::clicked, this, &TitleButtons::progress_clicked);
    row->addWidget(progress_);

    update_ = new BadgeButton;
    update_->setCursor(Qt::PointingHandCursor);
    update_->setVisible(false);
    connect(update_, &QToolButton::clicked, this, &TitleButtons::update_clicked);
    row->addWidget(update_);

    theme_ = new QToolButton;
    theme_->setAutoRaise(true);
    theme_->setPopupMode(QToolButton::InstantPopup);
    theme_->setToolTip(app::tr("Theme"));
    theme_->setAccessibleName(app::tr("Theme"));
    if (glyph_font().isEmpty()) {
        theme_->setText(QStringLiteral("\u25D0"));  // a circle half dark
    } else {
        QFont font(glyph_font());
        font.setPixelSize(14);
        theme_->setFont(font);
        theme_->setText(QStringLiteral("\uE790"));  // Color
    }
    // No arrow: the button is its glyph alone.
    theme_->setStyleSheet(QStringLiteral("QToolButton::menu-indicator { image: none; width: 0; }"));
    theme_menu_ = new QMenu(theme_);
    modes_ = new QActionGroup(theme_menu_);
    for (const auto& [name, value] : theme::theme_choices()) {  // Settings > Appearance's
        QAction* action = theme_menu_->addAction(name);
        action->setCheckable(true);
        action->setData(value);
        modes_->addAction(action);
        connect(action, &QAction::triggered, this, [this, value] {
            if (value != theme_value_) {
                emit theme_chosen(value);
            }
        });
    }
    // Below them, Settings > Appearance's looks, ticked as the themes are (no
    // swatch: a checkable item's icon takes the place of its tick).
    theme_menu_->addSeparator();
    for (const bool dark : {false, true}) {
        QMenu* looks = theme_menu_->addMenu(dark ? app::tr("Dark look") : app::tr("Light look"));
        QActionGroup*& group = dark ? dark_looks_ : light_looks_;
        group = new QActionGroup(looks);
        for (const auto& [name, value] : theme::look_choices(dark)) {
            QAction* action = looks->addAction(name);
            action->setCheckable(true);
            action->setData(value);
            group->addAction(action);
            connect(action, &QAction::triggered, this, [this, dark, value] { emit look_chosen(dark, value); });
        }
    }
    // The theme ticked is the one saved when the menu opens: Settings may
    // have changed it meanwhile without changing the colours (the system's
    // and the one it is now), which no palette change would tell.
    connect(theme_menu_, &QMenu::aboutToShow, this, &TitleButtons::theme_menu_opening);
    theme_->setMenu(theme_menu_);
    row->addWidget(theme_);
}

void TitleButtons::set_progress(const Progress& progress) {
    auto* button = static_cast<ProgressButton*>(progress_);
    if (progress_->text() != progress.text) {
        progress_->setText(progress.text);
    }
    const QString name = app::tr("Progress of the metrics being calculated: {state}", {{"state", progress.state}});
    if (progress_->accessibleName() != name) {
        progress_->setAccessibleName(name);  // a change is announced: only when there is one
    }
    // A tooltip shown is the one set when it showed: shown again in place,
    // so that it follows the run while it is read.
    if (progress_->toolTip() != progress.tooltip) {
        const bool shown = QToolTip::isVisible() && progress_->underMouse() && QToolTip::text() == progress_->toolTip();
        progress_->setToolTip(progress.tooltip);
        if (shown) {
            QToolTip::showText(QCursor::pos(), progress.tooltip, progress_);
        }
    }
    const double fraction = std::clamp(progress.fraction, 0.0, 1.0);
    if (button->fraction != fraction || button->tone != progress.tone) {
        button->fraction = fraction;
        button->tone = progress.tone;
        button->update();
    }
}

void TitleButtons::set_modern(bool modern) {
    modern_ = modern;
    // Classic's tab bar draws its tabs to its edges: a little room at its
    // right end and above the pane's frame.
    layout()->setContentsMargins(0, 0, modern ? 0 : 6, modern ? 0 : 3);
    theme_->setVisible(modern);
    update_->setVisible(modern && !update_version_.isEmpty());
}

void TitleButtons::set_update(const QString& version) {
    update_version_ = version;
    update_->setVisible(modern_ && !version.isEmpty());
    if (version.isEmpty()) {
        return;
    }
    update_->setText(app::tr("Update to {version}", {{"version", version}}));
    update_->setToolTip(app::tr("{app} {version} is available. Click for what's new and the download.",
                                {{"app", "VideoMetricsLab"}, {"version", version}}));
    update_->updateGeometry();
}

void TitleButtons::set_theme(const QString& theme, const QString& light_look, const QString& dark_look) {
    theme_value_ = theme;
    // Each group's match ticked; an unknown setting, its first (the
    // system's, Light's or Dark's own look).
    const auto tick = [](QActionGroup* group, const QString& value) {
        const QList<QAction*> actions = group->actions();
        const auto match = std::find_if(actions.begin(), actions.end(),
                                        [&value](const QAction* action) { return action->data().toString() == value; });
        (match != actions.end() ? *match : actions.front())->setChecked(true);
    };
    tick(modes_, theme);
    tick(light_looks_, light_look);
    tick(dark_looks_, dark_look);
}

QWidget* TitleButtons::progress_button() const { return progress_; }
QWidget* TitleButtons::update_button() const { return update_; }

#ifdef _WIN32

// ======================================================= the minimize, maximize
// and close buttons

// One of the bar's three buttons on its right, drawn as Windows 11 draws
// them: a glyph, a light wash on hover (red for close). The maximize button
// is a maximize button to Windows (HTMAXBUTTON, for its snap layouts), whose
// pointer then comes as the window's non-client messages: TitleBar sets its
// hover and press from them (set_hot, set_pressed).
class CaptionButton : public QAbstractButton {
public:
    CaptionButton(QString glyph, bool close, QWidget* parent) : QAbstractButton(parent), close_(close) {
        setText(std::move(glyph));
        setFocusPolicy(Qt::NoFocus);
        setAttribute(Qt::WA_Hover);
        QFont font(glyph_font());
        font.setPixelSize(10);
        setFont(font);
    }
    void set_hot(bool hot) {
        if (hot_ != hot) {
            hot_ = hot;
            update();
        }
    }
    void set_pressed(bool pressed) {
        if (pressed_ != pressed) {
            pressed_ = pressed;
            update();
        }
    }
    bool pressed() const { return pressed_; }
    QSize sizeHint() const override { return QSize(46, 32); }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        const bool hot = hot_ || underMouse();
        const bool down = pressed_ || isDown();
        const QColor ink = palette().windowText().color();
        QColor glyph = window()->isActiveWindow() ? ink : faded(ink, 0.45);
        if (close_ && (hot || down)) {
            painter.fillRect(rect(), down ? QColor(0xc4, 0x2b, 0x1c).lighter(115) : QColor(0xc4, 0x2b, 0x1c));
            glyph = Qt::white;
        } else if (down) {
            painter.fillRect(rect(), faded(ink, 0.04));
            glyph = faded(ink, 0.7);
        } else if (hot) {
            painter.fillRect(rect(), faded(ink, 0.07));
            glyph = ink;
        }
        painter.setPen(glyph);
        painter.drawText(rect(), Qt::AlignCenter, text());
    }
    void leaveEvent(QEvent* event) override {
        set_hot(false);
        QAbstractButton::leaveEvent(event);
    }

private:
    bool close_;
    bool hot_ = false;
    bool pressed_ = false;
};

// =================================================================== the bar

namespace {

constexpr int bar_height = 32;

const QString minimize_glyph = QStringLiteral("\uE921");
const QString maximize_glyph = QStringLiteral("\uE922");
const QString restore_glyph = QStringLiteral("\uE923");
const QString close_glyph = QStringLiteral("\uE8BB");

// The window's native window, or none while it is being made: winId()
// would make it, inside the messages of its making.
HWND hwnd_of(const QWidget* window) { return reinterpret_cast<HWND>(window->internalWinId()); }

// The window's icon at 16 pixels: its own, or else the program's (IDI_ICON1,
// VideoMetricsLab.rc), which Qt gives Windows but not the window.
QPixmap icon_pixmap(const QWidget* window) {
    const double ratio = window->devicePixelRatioF();
    const int size = qRound(16 * ratio);
    QPixmap pixmap;
    if (!window->windowIcon().isNull()) {
        pixmap = window->windowIcon().pixmap(QSize(16, 16), ratio);
    } else if (HICON icon = static_cast<HICON>(
                   LoadImageW(GetModuleHandleW(nullptr), L"IDI_ICON1", IMAGE_ICON, size, size, LR_DEFAULTCOLOR))) {
        pixmap = QPixmap::fromImage(QImage::fromHICON(icon));
        pixmap.setDevicePixelRatio(ratio);
        DestroyIcon(icon);
    }
    return pixmap;
}

}  // namespace

bool TitleBar::wanted() {
    // Windows' own windows: not offscreen (the tests), where there are none.
    if (theme::current_interface() != theme::Interface::modern ||
        QGuiApplication::platformName() != QLatin1String("windows")) {
        return false;
    }
    return !theme::system_contrast();  // in a contrast theme, the system's bar in its colours
}

TitleBar::TitleBar(QWidget* window, bool dialog) : QWidget(window), window_(window), dialog_(dialog) {
    setFixedHeight(bar_height);
    setAutoFillBackground(false);
    auto* row = new QHBoxLayout(this);
    row->setContentsMargins(10, 0, 0, 0);
    row->setSpacing(8);

    icon_ = new QLabel;
    icon_->setFixedSize(16, 16);
    icon_->setToolTip(app::tr("Window menu (Alt+Space)"));
    icon_->setAccessibleName(icon_->toolTip());  // a click opens the menu: named for a screen reader too
    icon_->installEventFilter(this);  // a click opens the system menu
    row->addWidget(icon_);
    title_ = new QLabel;
    title_->setAttribute(Qt::WA_TransparentForMouseEvents);
    row->addWidget(title_);
    row->addStretch(1);

    auto* captions = new QHBoxLayout;
    captions->setSpacing(0);
    if (!dialog_) {  // a dialog's bar: its title and close alone
        buttons_ = new TitleButtons;
        row->addWidget(buttons_);
        row->addSpacing(6);
        minimize_ = new CaptionButton(minimize_glyph, false, this);
        minimize_->setToolTip(app::tr("Minimize"));
        minimize_->setAccessibleName(app::tr("Minimize"));
        connect(minimize_, &QAbstractButton::clicked, window_, &QWidget::showMinimized);
        maximize_ = new CaptionButton(maximize_glyph, false, this);
        // A click on it with the pointer comes as the window's non-client
        // messages (native_event: it is HTMAXBUTTON to Windows) and never
        // reaches Qt as a click; this is for everything else that presses it:
        // Narrator, Voice Access and other UI Automation clients
        // (InvokePattern), which click it as they do Minimize and Close.
        connect(maximize_, &QAbstractButton::clicked, window_,
                [this] { window_->isMaximized() ? window_->showNormal() : window_->showMaximized(); });
        captions->addWidget(minimize_);
        captions->addWidget(maximize_);
    }
    close_ = new CaptionButton(close_glyph, true, this);
    close_->setToolTip(app::tr("Close"));
    close_->setAccessibleName(app::tr("Close"));
    connect(close_, &QAbstractButton::clicked, window_, &QWidget::close);
    captions->addWidget(close_);
    row->addLayout(captions);

    window_->installEventFilter(this);
    title_->setText(window_->windowTitle());
    icon_->setPixmap(icon_pixmap(window_));
    show_window_state();
}

void TitleBar::prepare() {
    // No title, icon or buttons of Qt's own: the bar is the app's. Asked for
    // before the native window is made, which these choose the kind of.
    window_->setWindowFlags((dialog_ ? Qt::Dialog : Qt::Window) | Qt::CustomizeWindowHint |
                            Qt::ExpandedClientAreaHint | Qt::NoTitleBarBackgroundHint);
    // The bar goes where Qt's own would have been: the window's contents
    // are not kept below it (Qt's safe area, for a bar of its own).
    window_->setAttribute(Qt::WA_ContentsMarginsRespectsSafeArea, false);
}

void TitleBar::attach() {
    if (attached_) {
        return;
    }
    attached_ = true;
    window_->winId();  // made by now (the first showEvent); made here if not
    // Without a title or buttons asked for, Qt makes the window without the
    // styles Windows' own behaviours look for: given back, double-clicking
    // the bar maximizes, the window snaps and the taskbar minimizes it. Qt
    // draws nothing for them: its own bar is not asked for.
    const HWND hwnd = hwnd_of(window_);
    const LONG_PTR style = GetWindowLongPtrW(hwnd, GWL_STYLE);
    SetWindowLongPtrW(hwnd, GWL_STYLE,
                      style | WS_CAPTION | WS_SYSMENU | (dialog_ ? 0 : WS_MINIMIZEBOX | WS_MAXIMIZEBOX));
    SetWindowPos(hwnd, nullptr, 0, 0, 0, 0,
                 SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_NOOWNERZORDER);
}

int resize_edge(QPoint point, QSize client, int edge_x, int edge_y) {
    const bool near_left = point.x() < edge_x;
    const bool near_right = point.x() >= client.width() - edge_x;
    if (point.y() < edge_y) {  // the top: a band inside, nothing above the bar
        return near_left ? HTTOPLEFT : near_right ? HTTOPRIGHT : HTTOP;
    }
    const bool near_bottom = point.y() >= client.height() - edge_y;
    if (point.x() < 0) {  // the rest: Windows' border, outside the client area
        return near_bottom ? HTBOTTOMLEFT : HTLEFT;
    }
    if (point.x() >= client.width()) {
        return near_bottom ? HTBOTTOMRIGHT : HTRIGHT;
    }
    if (point.y() >= client.height()) {
        return near_left ? HTBOTTOMLEFT : near_right ? HTBOTTOMRIGHT : HTBOTTOM;
    }
    return HTNOWHERE;
}

int TitleBar::hit_test(QPoint native) const {
    // Every point of the window is answered here: left to Qt, a window with
    // Qt::ExpandedClientAreaHint and CustomizeWindowHint has its hit test
    // answered by QWindowsWindow::handleNonClientHitTest, which makes a
    // mouse press or release of its own wherever the buttons' state has
    // changed since it last looked (GetAsyncKeyState), besides the one
    // Windows sends: every click came twice (Qt 6.12, QTBUG-152532;
    // AetherSDR's PR #6328 found the same).
    const HWND hwnd = hwnd_of(window_);
    POINT point{native.x(), native.y()};
    ScreenToClient(hwnd, &point);
    RECT client;
    GetClientRect(hwnd, &client);
    // The edges resize, unless maximized or full screen (resize_edge), or a
    // dialog, which keeps its size.
    if (!dialog_ && !window_->isMaximized() && !window_->isFullScreen()) {
        const UINT dpi = GetDpiForWindow(hwnd);
        const int padding = GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi);
        const int edge = resize_edge(QPoint(point.x, point.y), QSize(client.right, client.bottom),
                                     GetSystemMetricsForDpi(SM_CXSIZEFRAME, dpi) + padding,
                                     GetSystemMetricsForDpi(SM_CYSIZEFRAME, dpi) + padding);
        if (edge != HTNOWHERE) {
            return edge;
        }
    }
    const double ratio = window_->devicePixelRatioF();
    const QPoint local = mapFrom(window_, QPoint(qRound(point.x / ratio), qRound(point.y / ratio)));
    if (!rect().contains(local)) {
        return HTCLIENT;  // the window's contents, below the bar
    }
    const QWidget* child = childAt(local);
    if (child != nullptr && child == maximize_) {
        return HTMAXBUTTON;
    }
    for (const QWidget* widget = child; widget != nullptr && widget != this; widget = widget->parentWidget()) {
        if (qobject_cast<const QAbstractButton*>(widget) != nullptr || widget == icon_) {
            return HTCLIENT;  // the app's: its buttons, the icon's menu
        }
    }
    return HTCAPTION;  // the rest drags the window
}

void TitleBar::show_system_menu(QPoint at) {
    const HWND hwnd = hwnd_of(window_);
    const HMENU menu = GetSystemMenu(hwnd, FALSE);
    if (menu == nullptr) {
        return;
    }
    const bool maximized = IsZoomed(hwnd) != 0;
    const auto enable = [menu](UINT item, bool on) { EnableMenuItem(menu, item, MF_BYCOMMAND | (on ? MF_ENABLED : MF_GRAYED)); };
    enable(SC_RESTORE, maximized);
    enable(SC_MOVE, !maximized);
    enable(SC_SIZE, !maximized && !dialog_);
    enable(SC_MINIMIZE, !dialog_);
    enable(SC_MAXIMIZE, !maximized && !dialog_);
    enable(SC_CLOSE, true);
    SetMenuDefaultItem(menu, SC_CLOSE, FALSE);
    const UINT flags = TPM_RETURNCMD | TPM_RIGHTBUTTON | (QApplication::isRightToLeft() ? TPM_LAYOUTRTL : 0);
    const int command = TrackPopupMenu(menu, flags, at.x(), at.y(), 0, hwnd, nullptr);
    if (command != 0) {
        PostMessageW(hwnd, WM_SYSCOMMAND, static_cast<WPARAM>(command), 0);
    }
}

bool TitleBar::native_event(void* message, qintptr* result) {
    const MSG* msg = static_cast<const MSG*>(message);
    if (msg->hwnd == nullptr || msg->hwnd != hwnd_of(window_)) {
        return false;
    }
    switch (msg->message) {
    case WM_NCHITTEST:
        *result = hit_test(QPoint(static_cast<short>(LOWORD(msg->lParam)), static_cast<short>(HIWORD(msg->lParam))));
        return true;
    case WM_NCMOUSEMOVE:
        // The maximize button's pointer, which Windows has while it is a
        // maximize button to it.
        if (maximize_ == nullptr) {
            return false;
        }
        maximize_->set_hot(msg->wParam == HTMAXBUTTON);
        if (msg->wParam == HTMAXBUTTON) {
            TRACKMOUSEEVENT track{sizeof(TRACKMOUSEEVENT), TME_LEAVE | TME_NONCLIENT, msg->hwnd, 0};
            TrackMouseEvent(&track);
        } else {
            maximize_->set_pressed(false);
        }
        return false;
    case WM_NCMOUSELEAVE:
        if (maximize_ == nullptr) {
            return false;
        }
        maximize_->set_hot(false);
        maximize_->set_pressed(false);
        return false;
    case WM_NCLBUTTONDOWN:
    case WM_NCLBUTTONDBLCLK:
        if (msg->wParam == HTMAXBUTTON) {
            maximize_->set_pressed(true);  // Windows' own would draw a button of its own over the bar
            *result = 0;
            return true;
        }
        return false;
    case WM_NCLBUTTONUP:
        if (msg->wParam == HTMAXBUTTON) {
            if (maximize_->pressed()) {
                maximize_->set_pressed(false);
                window_->isMaximized() ? window_->showNormal() : window_->showMaximized();
            }
            *result = 0;
            return true;
        }
        return false;
    case WM_NCRBUTTONUP:
        if (msg->wParam == HTCAPTION) {
            show_system_menu(QPoint(static_cast<short>(LOWORD(msg->lParam)), static_cast<short>(HIWORD(msg->lParam))));
            *result = 0;
            return true;
        }
        return false;
    case WM_SYSKEYDOWN:
        // Alt+Space. Qt shows a menu of its own for it, with Minimize and
        // Maximize off, as the window asked Qt for no buttons.
        if (msg->wParam == VK_SPACE) {
            const double ratio = window_->devicePixelRatioF();
            const QPoint corner = mapTo(window_, QPoint(0, height()));
            POINT point{qRound(corner.x() * ratio), qRound(corner.y() * ratio)};
            ClientToScreen(msg->hwnd, &point);
            show_system_menu(QPoint(point.x, point.y));
            *result = 0;
            return true;
        }
        return false;
    default:
        return false;
    }
}

bool TitleBar::eventFilter(QObject* watched, QEvent* event) {
    if (watched == window_) {
        switch (event->type()) {
        case QEvent::WindowTitleChange:
            title_->setText(window_->windowTitle());
            break;
        case QEvent::WindowIconChange:
            icon_->setPixmap(icon_pixmap(window_));
            break;
        case QEvent::WindowStateChange:
            show_window_state();
            break;
        case QEvent::ActivationChange:
            show_active();
            for (QWidget* button : {static_cast<QWidget*>(minimize_), static_cast<QWidget*>(maximize_),
                                    static_cast<QWidget*>(close_)}) {
                if (button != nullptr) {
                    button->update();
                }
            }
            break;
        default:
            break;
        }
    } else if (watched == icon_ && event->type() == QEvent::MouseButtonPress) {
        const double ratio = window_->devicePixelRatioF();
        const QPoint corner = mapTo(window_, QPoint(0, height()));
        POINT point{qRound(corner.x() * ratio), qRound(corner.y() * ratio)};
        ClientToScreen(hwnd_of(window_), &point);
        show_system_menu(QPoint(point.x, point.y));
        return true;
    }
    return QWidget::eventFilter(watched, event);
}

void TitleBar::show_window_state() {
    if (maximize_ == nullptr) {
        return;
    }
    const bool maximized = window_->isMaximized();
    maximize_->setText(maximized ? restore_glyph : maximize_glyph);
    maximize_->setToolTip(maximized ? app::tr("Restore Down") : app::tr("Maximize"));
    maximize_->setAccessibleName(maximize_->toolTip());
}

void TitleBar::show_active() {
    // An inactive window's title and glyphs dimmed, as Windows' own are.
    const QColor text = palette().windowText().color();
    QPalette colours = title_->palette();
    colours.setColor(QPalette::WindowText, window_->isActiveWindow() ? text : faded(text, 0.45));
    title_->setPalette(colours);
    update();
}

void TitleBar::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.fillRect(rect(), palette().window());
}

void TitleBar::changeEvent(QEvent* event) {
    if (event->type() == QEvent::PaletteChange) {
        show_active();
    }
    QWidget::changeEvent(event);
}

#endif

}  // namespace vml::app
