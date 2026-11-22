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
// answers Windows' hit test for every point of the window: the edges, the
// bar, the maximize button (so that hovering it opens Windows 11's snap
// layouts) and the rest. Left to Qt, that hit test made a mouse press or
// release of its own beside Windows' (Qt 6.12), so that every click came
// twice. native_event also draws the maximize button's
// hover and press, and shows the system menu for Alt+Space, a click on the
// icon or a right-click on the bar, Minimize and Maximize included.
#pragma once

#include <QWidget>

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
    // WM_NCHITTEST's answer for `native` (a screen point in native
    // pixels), anywhere in the window: the edges, the bar's parts, the rest
    // client.
    int hit_test(QPoint native) const;
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
