// VML Modern's title bar and its quiet buttons.
//
// TitleButtons is three small buttons any tab can use: a run's progress
// (shown only during a run; a click opens the Videos tab), the theme (a menu
// of the themes), and an update's badge (shown only when one is available; a
// click shows it). They are VML Modern's alone. On Windows they sit in the
// title bar; elsewhere, and on Windows where the title bar stays the
// system's, at the right end of the tab bar.
//
// TitleBar (Windows) is the window's title bar drawn by the app, in the
// theme's colours: the icon, the title, the TitleButtons and the minimize,
// maximize and close buttons. Qt's Qt::ExpandedClientAreaHint gives the
// window's area to the app up to its top edge, the system's frame staying
// (its shadow, rounded corners, resize edges and snapping). Asked for
// without a title or buttons, Qt draws none of its own, and the window's
// styles that Windows' own behaviours look for (double-click to maximize,
// snapping, the taskbar's minimize) are given back (TitleBar::attach). The
// window's nativeEvent passes its messages to TitleBar::native_event, which
// tells Windows which parts are the bar, the edges and the maximize button
// (so that hovering it opens Windows 11's snap layouts), draws that button's
// hover and press, and shows the system menu for Alt+Space, a click on the
// icon or a right-click on the bar, Minimize and Maximize included.
#pragma once

#include <QPointF>
#include <QWidget>

#include <functional>
#include <optional>

class QAbstractButton;
class QLabel;
class QMainWindow;
class QMenu;
class QProgressBar;
class QToolButton;

namespace vml::app {

class CaptionButton;

class TitleButtons : public QWidget {
    Q_OBJECT

public:
    explicit TitleButtons(QWidget* parent = nullptr);

    // A run's progress: `text` ("2 of 5 · 43 %") on the button, `fraction`
    // (0 to 1) as its thin bar, `tooltip` the status line. None: no run, the
    // button hidden.
    struct Progress {
        QString text;
        double fraction = 0.0;
        QString tooltip;
    };
    void set_progress(const std::optional<Progress>& progress);
    // An update's version on its badge; empty: none, the badge hidden.
    void set_update(const QString& version);
    // The theme ticked in the theme menu ("", "light", "dark" or "mocha").
    void set_theme(const QString& theme);

    QWidget* progress_button() const;
    QWidget* update_button() const;

signals:
    void progress_clicked();
    void update_clicked();
    void theme_chosen(const QString& theme);
    // The theme menu is about to open: set_theme, with the theme saved now.
    void theme_menu_opening();

private:
    QToolButton* progress_ = nullptr;
    QProgressBar* progress_bar_ = nullptr;
    QToolButton* theme_ = nullptr;
    QMenu* theme_menu_ = nullptr;
    QToolButton* update_ = nullptr;
    QString theme_value_;
};

// A window with Qt::ExpandedClientAreaHint gets each of its mouse presses and
// releases twice on Windows (Qt 6.12: press, press, release, release, once
// with Qt's own timestamp and once with Windows'; Windows itself sends one
// of each). A menu a button opens on its press then closed at once, the
// second press landing outside it; whatever acts on a press acted twice; and
// Qt, timing double clicks across both clocks, made none.
//
// This filter drops a press or release that repeats the one just before it:
// the same kind and button at the same place within 100 ms, nothing between.
// A real click cannot repeat itself so: a release comes between two presses.
// On `window` (the one with the flag) it also makes the double clicks: Qt's
// are dropped, and a kept press that follows the last within the system's
// double-click time and distance is delivered, then a double click, as Qt
// delivers them. TitleBar::prepare installs it on the application, on the
// windows' own (QWindow) events.
class RepeatedMouseFilter : public QObject {
public:
    // `now`: milliseconds, from any start (tests give their own).
    RepeatedMouseFilter(QWidget* window, std::function<qint64()> now, QObject* parent = nullptr);
    explicit RepeatedMouseFilter(QWidget* window, QObject* parent = nullptr);

    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    QWidget* window_;
    std::function<qint64()> now_;
    bool delivering_ = false;  // sending a press and its double click itself
    int last_type_ = 0;  // QEvent::Type of the last press or release; 0: none
    int last_button_ = 0;
    QPointF last_at_;
    qint64 last_time_ = 0;
    // The window's last kept press, for double clicks: counted, if it can
    // be a double click's first.
    bool press_counted_ = false;
    int press_button_ = 0;
    QPointF press_at_;
    qint64 press_time_ = 0;
};

#ifdef _WIN32

class TitleBar : public QWidget {
public:
    // Whether the window can have the app's title bar: Windows, VML Modern
    // and not high contrast (whose colours are the system's).
    static bool wanted();

    explicit TitleBar(QMainWindow* window);

    // Asks Qt for the window's area up to its top edge, with no title or
    // buttons of its own: before the window is first shown.
    void prepare();
    // Gives the window back the styles Windows' behaviours look for: once its
    // native window is made (the window's first showEvent).
    void attach();
    // The window's native messages: true if handled, with `result` for it.
    bool native_event(void* message, qintptr* result);

    TitleButtons* buttons() const { return buttons_; }

protected:
    void paintEvent(QPaintEvent* event) override;
    void changeEvent(QEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    // The system menu at `at` (global, native pixels), Minimize, Maximize,
    // Restore, Move and Size enabled as the window's state allows.
    void show_system_menu(QPoint at);
    // Where in the bar `native` (a screen point in native pixels) is, as
    // WM_NCHITTEST's answer, or none outside it.
    std::optional<int> hit_test(QPoint native) const;
    void show_window_state();
    void show_active();

    QMainWindow* window_;
    QLabel* icon_ = nullptr;
    QLabel* title_ = nullptr;
    TitleButtons* buttons_ = nullptr;
    CaptionButton* minimize_ = nullptr;
    CaptionButton* maximize_ = nullptr;
    CaptionButton* close_ = nullptr;
    bool attached_ = false;
};

#endif

}  // namespace vml::app
