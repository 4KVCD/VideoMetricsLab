// The window in each interface (theme.h): every role a widget is given is one
// VML Modern's style sheet styles, and the window is drawn in Classic, Modern
// dark and Modern light without a style sheet Qt cannot read.
//
// Applying an interface changes the whole application's style, and Classic
// cannot be put back once Modern is shown (theme.h), so these tests are a
// program of their own (vml_appearance_tests), run in the order written:
// Classic first. They share one window, built once.
//
// Modern dark needs a platform that takes the app's light or dark (Windows',
// macOS'): Qt's offscreen one, the tests' own, does not, so there it is
// skipped. VML_TEST_ON_SCREEN=1 runs it, on the display (tests/app/main.cpp).
//
// With VML_UI_SNAPSHOTS set to a folder, each test also saves the window's
// tabs there as PNGs (classic-01-videos.png, modern-dark-01-videos.png, ...),
// for a review to look at both interfaces without starting the app.
#include <QApplication>
#include <QComboBox>
#include <QDir>
#include <QBoxLayout>
#include <QFrame>
#include <QAbstractItemView>
#include <QGroupBox>
#include <QHeaderView>
#include <QIcon>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QMargins>
#include <QPushButton>
#include <QRegularExpression>
#include <QMouseEvent>
#include <QScreen>
#include <QSet>
#include <QSlider>
#include <QSpinBox>
#include <QStyle>
#include <QStyleOption>
#include <QTabWidget>
#include <QTableWidget>
#include <QWidget>
#include <QWindow>

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "chart.h"
#include "keyboard.h"
#include "main_window.h"
#include "quick_save.h"
#include "theme.h"
#include "widgets.h"

namespace vml::app {
namespace {

const QColor accent("#4f8fe8");

// The roles VML Modern's style sheet has a rule for.
QSet<QString> styled_roles() {
    QSet<QString> roles;
    const QRegularExpression rule(R"re(\[vmlRole="([a-z]+)"\])re");
    for (const bool dark : {false, true}) {
        auto it = rule.globalMatch(theme::modern_style_sheet(dark, accent));
        while (it.hasNext()) {
            roles.insert(it.next().captured(1));
        }
    }
    return roles;
}

// Qt's warnings while `draw` runs: a style sheet it cannot read says so here,
// and is otherwise dropped without a word.
std::vector<std::string> warnings_while(const std::function<void()>& draw) {
    static std::vector<std::string>* caught = nullptr;
    std::vector<std::string> warnings;
    caught = &warnings;
    const QtMessageHandler previous = qInstallMessageHandler([](QtMsgType type, const QMessageLogContext&, const QString& text) {
        // Qt's offscreen platform remarks on any window's size hints: not a sheet's.
        if (type != QtDebugMsg && type != QtInfoMsg && caught != nullptr &&
            !text.contains("propagateSizeHints")) {
            caught->push_back(text.toStdString());
        }
    });
    draw();
    qInstallMessageHandler(previous);
    caught = nullptr;
    return warnings;
}

// Draws each tab of `window`, and saves it when VML_UI_SNAPSHOTS names a folder.
void draw_tabs(QWidget& window, const QString& name) {
    auto* tabs = window.findChild<QTabWidget*>();
    ASSERT_NE(tabs, nullptr);
    const QString folder = qEnvironmentVariable("VML_UI_SNAPSHOTS");
    for (int index = 0; index < tabs->count(); ++index) {
        tabs->setCurrentIndex(index);
        QApplication::processEvents();
        const QImage image = window.grab().toImage();
        EXPECT_FALSE(image.isNull()) << tabs->tabText(index).toStdString();
        if (!folder.isEmpty()) {
            const QString tab = tabs->tabText(index).toLower().replace(' ', '-').remove('&');
            image.save(QDir(folder).filePath(QStringLiteral("%1-%2-%3.png").arg(name).arg(index + 1, 2, 10, QChar('0')).arg(tab)));
        }
    }
    tabs->setCurrentIndex(0);
}

// The colour a widget is drawn in at (x, y), drawn alone.
QColor pixel(QWidget& widget, int x, int y) {
    return widget.grab().toImage().pixelColor(x, y);
}

class Appearance : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Layout that is Modern's alone (theme::container, modern_only,
        // classic_only, classic_box), built in Classic, checked in each.
        layouts_ = new QWidget;
        auto* column = new QVBoxLayout(layouts_);
        QBoxLayout* inside = nullptr;
        padded_ = theme::container("card", true, &inside);
        padded_child_ = new QLabel("inside");
        inside->addWidget(padded_child_);
        column->addWidget(padded_);
        modern_label_ = new QLabel("Modern's");
        theme::modern_only(modern_label_);
        column->addWidget(modern_label_);
        classic_label_ = new QLabel("Classic's");
        theme::classic_only(classic_label_);
        column->addWidget(classic_label_);
        box_ = new QGroupBox;
        theme::classic_box(box_, "Framed in Classic");
        column->addWidget(box_);
        layouts_->resize(400, 300);
        layouts_->show();
        window_ = new MainWindow;
        window_->resize(1600, 1000);
        window_->show();
        QApplication::processEvents();
    }
    static void TearDownTestSuite() {
        delete layouts_;
        delete window_;
        window_ = nullptr;
    }
    static MainWindow* window_;
    static QWidget* layouts_;
    static QWidget* padded_;
    static QLabel* padded_child_;
    static QLabel* modern_label_;
    static QLabel* classic_label_;
    static QGroupBox* box_;

    // Where padded_child_ sits in its container, laid out now.
    static QPoint child_place() {
        padded_->layout()->activate();
        QApplication::processEvents();
        return padded_child_->pos();
    }

    // Modern in dark, light or Mocha (dark, in its own colours): each role
    // drawn as its rule says, a style sheet Qt reads, and the window's tabs
    // drawn in it.
    static void check_modern(bool dark, bool mocha = false) {
        theme::apply_theme(dark || mocha ? "dark" : "light", "light", mocha ? "mocha" : "dark");
        std::vector<std::string> warnings = warnings_while([] {
            theme::apply_appearance(theme::Interface::modern, accent);
        });
        ASSERT_EQ(theme::current_interface(), theme::Interface::modern);
        const char* name = mocha ? "modern-mocha" : dark ? "modern-dark" : "modern-light";
        // Modern's layout: the container padded, its labels shown and
        // Classic's hidden, the box untitled.
        EXPECT_GE(child_place().x(), 12) << name;
        EXPECT_GE(child_place().y(), 10) << name;
        EXPECT_TRUE(modern_label_->isVisibleTo(layouts_)) << name;
        EXPECT_FALSE(classic_label_->isVisibleTo(layouts_)) << name;
        EXPECT_TRUE(box_->title().isEmpty()) << name;
        // The Options' middle column, built in Classic: Modern's margins,
        // none, after the switch, so that its cards line up (#182).
        const auto* metric_box = window_->findChild<QGroupBox*>("metricSettings");
        ASSERT_NE(metric_box, nullptr);
        EXPECT_EQ(metric_box->layout()->contentsMargins(), QMargins()) << name;
        // And no padding of the sheet's either, whether the box was untitled
        // when the sheet came (built in Modern) or not (built in Classic):
        // the cards' tops level with the side boxes'.
        EXPECT_EQ(metric_box->contentsRect().top(), 0) << name;

        QWidget probes;
        auto* primary = new QPushButton("Run", &probes);
        theme::set_role(primary, "primary");
        auto* card = new QWidget(&probes);
        theme::set_role(card, "card");
        auto* toolbar = new QWidget(&probes);
        theme::set_role(toolbar, "toolbar");
        for (QWidget* probe : {static_cast<QWidget*>(primary), card, toolbar}) {
            probe->resize(120, 40);
        }
        probes.show();
        QApplication::processEvents();
        EXPECT_EQ(pixel(*primary, 6, 20), accent) << name;
        EXPECT_EQ(pixel(*card, 60, 20), theme::modern_colour("card", dark)) << name;
        EXPECT_EQ(pixel(*toolbar, 60, 20), theme::modern_colour("raised", dark)) << name;

        const std::vector<std::string> drawing = warnings_while([&] { draw_tabs(*window_, name); });
        warnings.insert(warnings.end(), drawing.begin(), drawing.end());
        for (const std::string& warning : warnings) ADD_FAILURE() << name << ": " << warning;

        // Hints upright; a sheet only Classic draws, none.
        QLabel hint;
        theme::style(&hint, "color: {muted}; font-style: {hint_style};");
        EXPECT_TRUE(hint.styleSheet().contains("font-style: normal;")) << name;
        QLabel classic_only;
        theme::style_classic(&classic_only, "QLabel {{ color: {muted}; }}");
        EXPECT_TRUE(classic_only.styleSheet().isEmpty()) << name;

        // Video Compare's Zoom: its text box as wide as "Fit to window" and
        // the text box's own margins, so the text is not scrolled, its first
        // letter cut off (its tab was shown by draw_tabs).
        for (QComboBox* combo : window_->findChildren<QComboBox*>()) {
            if (combo->isEditable() && combo->itemText(0) == "Fit to window") {
                QLineEdit* edit = combo->lineEdit();
                EXPECT_GE(edit->width(), edit->fontMetrics().horizontalAdvance("Fit to window") + 4) << name;
            }
        }
    }
};

MainWindow* Appearance::window_ = nullptr;
QWidget* Appearance::layouts_ = nullptr;
QWidget* Appearance::padded_ = nullptr;
QLabel* Appearance::padded_child_ = nullptr;
QLabel* Appearance::modern_label_ = nullptr;
QLabel* Appearance::classic_label_ = nullptr;
QGroupBox* Appearance::box_ = nullptr;

TEST_F(Appearance, ClassicHasNoStyleSheetOfItsOwn) {
    ASSERT_EQ(theme::current_interface(), theme::Interface::classic);  // run first: see the top
    theme::apply_appearance(theme::Interface::classic, accent);
    EXPECT_TRUE(qApp->styleSheet().isEmpty());
    EXPECT_TRUE(warnings_while([] { draw_tabs(*window_, "classic"); }).empty());
    // Modern's layout leaves Classic's as it was: no margins, Modern's
    // labels hidden, Classic's shown, the box framed and titled.
    EXPECT_EQ(child_place(), QPoint(0, 0));
    EXPECT_FALSE(modern_label_->isVisibleTo(layouts_));
    EXPECT_TRUE(classic_label_->isVisibleTo(layouts_));
    EXPECT_EQ(box_->title(), "Framed in Classic");
}

QPushButton* window_button(QWidget& window, const QString& text) {
    for (QPushButton* button : window.findChildren<QPushButton*>()) {
        if (button->text() == text && button->isVisible()) {
            return button;
        }
    }
    return nullptr;
}

void settle() {
    QApplication::processEvents();  // the focus's change, then the ring's move queued after it
    QApplication::processEvents();
}

TEST_F(Appearance, NoFocusRingInClassic) {
    ASSERT_EQ(theme::current_interface(), theme::Interface::classic);  // run before Modern
    window_->activateWindow();
    QApplication::processEvents();
    ASSERT_EQ(QApplication::activeWindow(), window_);
    QPushButton* add = window_button(*window_, "Add files");
    ASSERT_NE(add, nullptr);
    add->clearFocus();  // focused anew
    add->setFocus(Qt::TabFocusReason);
    settle();
    EXPECT_TRUE(keyboard::focus_visible());
    const keyboard::FocusRing* ring = keyboard::ring_in(window_);
    EXPECT_TRUE(ring == nullptr || !ring->isVisible());
    keyboard::set_focus_visible(false);
}

// Every widget given a role is given one the style sheet styles: a role
// misspelt, or one added without its rule, would be drawn as no role at all,
// and only in Modern, where a review of Classic alone would not see it.
TEST_F(Appearance, EveryRoleInTheWindowIsOneModernStyles) {
    const QSet<QString> styled = styled_roles();
    EXPECT_TRUE(styled.contains("primary"));
    EXPECT_TRUE(styled.contains("card"));
    int given = 0;
    for (const QWidget* widget : window_->findChildren<QWidget*>()) {
        const QString role = widget->property("vmlRole").toString();
        if (!role.isEmpty()) {
            ++given;
            EXPECT_TRUE(styled.contains(role)) << role.toStdString() << " on " << widget->metaObject()->className()
                                               << " " << widget->objectName().toStdString();
        }
    }
    EXPECT_GT(given, 0);  // the window's roles were found at all

    // The Videos tab's: the test videos a card, its buttons a footer, the
    // removing ones quiet.
    const auto* card = window_->findChild<QWidget*>("testsCard");
    ASSERT_NE(card, nullptr);
    EXPECT_EQ(card->property("vmlRole").toString(), "card");
    const auto* footer = window_->findChild<QWidget*>("testsFooter");
    ASSERT_NE(footer, nullptr);
    EXPECT_EQ(footer->property("vmlRole").toString(), "footer");
    int removing = 0;
    for (const QPushButton* button : footer->findChildren<QPushButton*>()) {
        if (button->text() == "Remove all" || button->text() == "Remove selected") {
            ++removing;
            EXPECT_EQ(button->property("vmlRole").toString(), "quiet");
        }
    }
    EXPECT_EQ(removing, 2);

    // The Bitrate Viewer's: the files a card, the graph a card.
    for (const char* name : {"filesCard", "bitrateGraphCard"}) {
        const auto* bitrate_card = window_->findChild<QWidget*>(name);
        ASSERT_NE(bitrate_card, nullptr) << name;
        EXPECT_EQ(bitrate_card->property("vmlRole").toString(), "card") << name;
    }

    // Metric Graphs': the metrics' pills, chart and readout one card.
    const auto* graph_card = window_->findChild<QWidget*>("graphCard");
    ASSERT_NE(graph_card, nullptr);
    EXPECT_EQ(graph_card->property("vmlRole").toString(), "card");
}

// Every table, list or chart on a tab that shows data has a save icon on its
// card (docs/ARCHITECTURE.md, "Exports"): the nearest card, framed box or, with
// neither, the tab's page around it holds a SaveButton. A view that is not data
// to save (a list of files to read, one to pick from) is marked so
// (not_exported), with why where it is made. A new card without a save icon
// fails here.
TEST_F(Appearance, EveryCardThatShowsDataHasASaveIcon) {
    const QTabWidget* tabs = nullptr;
    for (const QTabWidget* each : window_->findChildren<QTabWidget*>()) {
        if (each->count() > 0 && each->tabText(0) == "Videos") {
            tabs = each;
        }
    }
    ASSERT_NE(tabs, nullptr);
    int views = 0;
    for (int index = 0; index < tabs->count(); ++index) {
        QWidget* page = tabs->widget(index);
        if (tabs->tabText(index) == "Settings") {
            continue;  // its lists are settings, not data
        }
        for (QWidget* view : page->findChildren<QWidget*>()) {
            const bool shows = (qobject_cast<QAbstractItemView*>(view) != nullptr &&
                                qobject_cast<QHeaderView*>(view) == nullptr) ||
                               qobject_cast<ChartWidget*>(view) != nullptr ||
                               QString(view->metaObject()->className()).endsWith("RateCurveChart");
            if (!shows || is_not_exported(view)) {
                continue;
            }
            bool in_a_combo = false;  // a combo box's list, not a view of its own
            QWidget* holder = view->parentWidget();
            for (QWidget* up = view->parentWidget(); up != nullptr && up != page; up = up->parentWidget()) {
                in_a_combo = in_a_combo || qobject_cast<QComboBox*>(up) != nullptr;
            }
            if (in_a_combo) {
                continue;
            }
            while (holder != page && holder->property("vmlRole").toString() != "card" &&
                   qobject_cast<QGroupBox*>(holder) == nullptr) {
                holder = holder->parentWidget();
            }
            ++views;
            EXPECT_FALSE(holder->findChildren<SaveButton*>().isEmpty())
                << tabs->tabText(index).toStdString() << ": " << view->metaObject()->className() << " "
                << view->objectName().toStdString() << " in " << holder->metaObject()->className() << " "
                << holder->objectName().toStdString() << " has no save icon on its card";
        }
    }
    EXPECT_GE(views, 8);  // the tabs' tables, lists and charts were found at all
}

TEST_F(Appearance, ModernLight) {
    check_modern(false);
}

// The window's tabs have nothing drawn above them in Modern. Qt's base line
// for a tab bar ran along their top, broken under the selected tab: a
// thinner edge there in the light looks, none in the dark ones.
TEST_F(Appearance, NoLineAboveTheWindowsTabsInModern) {
    theme::apply_theme("light", "light", "dark");
    theme::apply_appearance(theme::Interface::modern, accent);
    auto* tabs = window_->findChild<QTabWidget*>();
    ASSERT_NE(tabs, nullptr);
    tabs->setCurrentIndex(0);
    QApplication::processEvents();
    const QImage bar = tabs->tabBar()->grab().toImage();
    const QRect selected = tabs->tabBar()->tabRect(0);
    const QRect other = tabs->tabBar()->tabRect(1);
    const QColor ground = bar.pixelColor(selected.left() + 2, selected.center().y());  // the padding, not the text
    for (int y = 0; y < 3; ++y) {
        EXPECT_EQ(bar.pixelColor(selected.center().x(), y), ground) << "above the selected tab, row " << y;
        EXPECT_EQ(bar.pixelColor(other.center().x(), y), ground) << "above another tab, row " << y;
    }
}

// The control the keys moved to has a ring of the accent around it in VML
// Modern, whose style sheet draws the buttons' own focus too faintly to see
// (keyboard.h); a click takes the ring away, as on Windows. Classic draws
// none: the system's style shows the focus.
TEST_F(Appearance, AFocusRingAroundTheControlTheKeysMovedToInModern) {
    ASSERT_EQ(theme::current_interface(), theme::Interface::modern);  // run after ModernLight
    window_->activateWindow();
    QApplication::processEvents();
    ASSERT_EQ(QApplication::activeWindow(), window_);
    QPushButton* add = window_button(*window_, "Add files");
    ASSERT_NE(add, nullptr);
    add->clearFocus();  // focused anew
    add->setFocus(Qt::TabFocusReason);
    settle();
    ASSERT_TRUE(keyboard::focus_visible());
    keyboard::FocusRing* ring = keyboard::ring_in(window_);
    ASSERT_NE(ring, nullptr);
    ASSERT_TRUE(ring->isVisible());
    const QRect button(add->mapTo(window_, QPoint()), add->size());
    const int out = keyboard::FocusRing::gap + keyboard::FocusRing::width;
    EXPECT_EQ(ring->geometry(), button.adjusted(-out, -out, out, out));
    // Drawn in the accent, along the middle of its top edge, over what is under it.
    const QImage drawn = window_->grab().toImage();
    EXPECT_EQ(drawn.pixelColor(button.center().x(), button.top() - out), accent);
    EXPECT_EQ(drawn.pixelColor(button.center().x(), button.top() - out + 1), accent);

    const QPointF at(4, 4);
    QMouseEvent press(QEvent::MouseButtonPress, at, add->mapToGlobal(at), Qt::LeftButton, Qt::LeftButton,
                      Qt::NoModifier);
    QApplication::sendEvent(add, &press);
    QMouseEvent release(QEvent::MouseButtonRelease, QPointF(-50, -50), add->mapToGlobal(QPointF(-50, -50)),
                        Qt::LeftButton, Qt::NoButton, Qt::NoModifier);  // released off it: not a click
    QApplication::sendEvent(add, &release);
    settle();
    EXPECT_FALSE(keyboard::focus_visible());
    EXPECT_FALSE(ring->isVisible());
}

// Issue #203: a left click on a slider jumps to the click in Modern, as in
// Classic. Fusion, which Modern's style is drawn over, paged toward it by
// the page step instead (10 frames a click on Video Compare's timeline).
TEST_F(Appearance, ALeftClickOnASliderJumpsThereInModern) {
    ASSERT_EQ(theme::current_interface(), theme::Interface::modern);  // run after ModernLight
    QSlider slider(Qt::Horizontal);
    slider.setRange(0, 1000);
    slider.resize(400, 24);
    slider.show();
    QApplication::processEvents();
    const QPointF at(300, 12);  // three quarters along
    QMouseEvent press(QEvent::MouseButtonPress, at, slider.mapToGlobal(at), Qt::LeftButton, Qt::LeftButton,
                      Qt::NoModifier);
    QApplication::sendEvent(&slider, &press);
    QMouseEvent release(QEvent::MouseButtonRelease, at, slider.mapToGlobal(at), Qt::LeftButton, Qt::NoButton,
                        Qt::NoModifier);
    QApplication::sendEvent(&slider, &release);
    EXPECT_NEAR(slider.value(), 750, 30);  // not 10, one page step
}

// The leftmost column of `image` inside `area` with a pixel far from
// `background`: where the text or a drawing starts. -1: none.
int first_ink(const QImage& image, const QRect& area, const QColor& background) {
    for (int x = area.left(); x <= area.right(); ++x) {
        for (int y = area.top(); y <= area.bottom(); ++y) {
            const QColor c = image.pixelColor(x, y);
            if (std::abs(c.red() - background.red()) + std::abs(c.green() - background.green()) +
                    std::abs(c.blue() - background.blue()) > 90) {
                return x;
            }
        }
    }
    return -1;
}

// Issue #250: Modern's drop-downs and spin boxes. The arrow is a chevron
// inside the rounded field, not a separate box; the list opens below the
// field, at least as wide as its longest entry; a spin box's steps sit
// inside its field, and its text is inset as a drop-down's.
TEST_F(Appearance, DropDownsInModern) {
    ASSERT_EQ(theme::current_interface(), theme::Interface::modern);  // run after ModernLight
    QWidget host;
    auto* combo = new QComboBox(&host);
    const QString longest = QStringLiteral("An entry much longer than the field it is chosen in");
    combo->addItems({QStringLiteral("0 short"), longest});
    combo->setGeometry(10, 10, 140, 28);
    auto* spin = new QSpinBox(&host);
    spin->setGeometry(10, 50, 140, 28);
    host.resize(400, 300);
    host.show();
    QApplication::processEvents();
    QStyle* style = combo->style();
    const QRect field = combo->rect();

    QStyleOptionComboBox option;
    option.initFrom(combo);
    option.subControls = QStyle::SC_All;
    EXPECT_EQ(style->styleHint(QStyle::SH_ComboBox_Popup, &option, combo), 0);
    // The arrow's area inside the border (1 px) and as tall as the text's, at
    // the right; the text's area ends where it starts.
    const QRect arrow = style->subControlRect(QStyle::CC_ComboBox, &option, QStyle::SC_ComboBoxArrow, combo);
    EXPECT_TRUE(field.adjusted(1, 1, -1, -1).contains(arrow)) << arrow.x() << "," << arrow.y();
    EXPECT_EQ(arrow.right(), field.right() - 1);
    const QRect text = style->subControlRect(QStyle::CC_ComboBox, &option, QStyle::SC_ComboBoxEditField, combo);
    EXPECT_LT(text.right(), arrow.left());
    // The chevron drawn in it, in the muted text colour: no bevelled box.
    const QImage drawn = combo->grab().toImage();
    const QColor background = theme::modern_colour("field", false);
    const int chevron = first_ink(drawn, arrow, background);
    EXPECT_GE(chevron, arrow.left() + 2);
    EXPECT_EQ(drawn.pixelColor(arrow.center().x(), arrow.top() + 1), background);
    // Its icon: painted, sharp at any scale, in the colour its name gives.
    const QPixmap red = QIcon(QStringLiteral(":/chevron-down-ff0000.vmlchevron")).pixmap(QSize(20, 12), 1.5);
    ASSERT_FALSE(red.isNull());
    const QImage red_image = red.toImage();
    EXPECT_EQ(red_image.pixelColor(red_image.width() / 2, red_image.height() * 3 / 4).red(), 255);
    EXPECT_EQ(red_image.pixelColor(0, red_image.height() - 1).alpha(), 0);

    // The list below the field, the field still in view, as wide as the
    // longest entry; its window see-through for its rounded corners.
    combo->showPopup();
    QApplication::processEvents();
    QWidget* list = combo->view()->window();
    ASSERT_NE(list, combo->window());
    EXPECT_TRUE(list->isVisible());
    const QPoint below = combo->mapToGlobal(QPoint(0, combo->height()));
    EXPECT_GE(list->geometry().top(), below.y());
    EXPECT_EQ(list->geometry().left(), below.x());
    EXPECT_GE(list->width(), combo->view()->fontMetrics().horizontalAdvance(longest));
    EXPECT_TRUE(list->testAttribute(Qt::WA_TranslucentBackground));
    combo->hidePopup();
    // With no room below, above it, the same distance away.
    const QRect screen = combo->screen()->availableGeometry();
    const QPoint home = host.pos();
    host.move(screen.left(), screen.bottom() - 45);
    QApplication::processEvents();
    combo->showPopup();
    QApplication::processEvents();
    const int field_top = combo->mapToGlobal(QPoint(0, 0)).y();
    if (combo->mapToGlobal(QPoint(0, combo->height())).y() + list->height() > screen.bottom()) {
        EXPECT_LE(list->geometry().bottom(), field_top - 1);
        EXPECT_GE(list->geometry().bottom(), field_top - 4);
    } else {
        ADD_FAILURE() << "the field is not near enough the screen's bottom";
    }
    combo->hidePopup();
    host.move(home);

    // The spin box's steps inside its field, at the right; its text inset
    // as the drop-down's.
    QStyleOptionSpinBox spin_option;
    spin_option.initFrom(spin);
    spin_option.subControls = QStyle::SC_All;
    spin_option.buttonSymbols = QAbstractSpinBox::UpDownArrows;
    spin_option.stepEnabled = QAbstractSpinBox::StepUpEnabled;
    const QRect spin_field = spin->rect().adjusted(1, 1, -1, -1);
    for (const auto step : {QStyle::SC_SpinBoxUp, QStyle::SC_SpinBoxDown}) {
        const QRect rect = style->subControlRect(QStyle::CC_SpinBox, &spin_option, step, spin);
        EXPECT_TRUE(spin_field.contains(rect)) << rect.x() << "," << rect.y() << " " << rect.width() << "x" << rect.height();
    }
    const int combo_text = first_ink(drawn, QRect(2, 4, 30, field.height() - 8), background);
    const int spin_text = first_ink(spin->grab().toImage(), QRect(2, 4, 30, spin->height() - 8), background);
    EXPECT_GE(combo_text, 6);
    EXPECT_NEAR(spin_text, combo_text, 1);
}

TEST_F(Appearance, ModernDark) {
    theme::apply_theme("dark");
    if (!theme::is_dark()) {
        GTEST_SKIP() << "This platform does not take the app's dark (Qt's offscreen one): run with VML_TEST_ON_SCREEN=1.";
    }
    check_modern(true);
}

// Mocha: Modern's dark, in chocolate browns; amber while no accent is
// chosen, the one chosen otherwise; and in Classic, Dark.
TEST_F(Appearance, ModernMocha) {
    theme::apply_theme("dark");
    if (!theme::is_dark()) {
        GTEST_SKIP() << "This platform does not take the app's dark (Qt's offscreen one): run with VML_TEST_ON_SCREEN=1.";
    }
    check_modern(true, true);
    EXPECT_TRUE(theme::is_mocha());
    EXPECT_EQ(theme::modern_colour("card", true), QColor("#251c16"));
    EXPECT_NE(theme::modern_colour("card", true), theme::modern_colour("card", false));
    EXPECT_EQ(QApplication::palette().color(QPalette::Accent), accent);  // the one chosen
    theme::apply_appearance(theme::Interface::modern, std::nullopt);
    EXPECT_EQ(QApplication::palette().color(QPalette::Accent), theme::mocha_accent());
    // Back to Dark's own look: Modern's own dark colours, at once.
    theme::apply_theme("dark", "light", "dark");
    EXPECT_FALSE(theme::is_mocha());
    EXPECT_EQ(QApplication::palette().color(QPalette::Base), theme::modern_colour("card", true));
    EXPECT_NE(theme::modern_colour("card", true), QColor("#251c16"));
    EXPECT_NE(QApplication::palette().color(QPalette::Accent), theme::mocha_accent());  // the system's again
}

// The Videos table's metric headings are on two lines where that makes
// them narrower, and every metric column is as wide as its heading needs,
// hidden ones too: Qt sizes a hidden section as 0 wide, so a column shown
// later kept 78 px and its heading was cut at both ends.
TEST_F(Appearance, MetricHeadingsFitTheirColumns) {
    QTableWidget* table = nullptr;
    for (QTableWidget* candidate : window_->findChildren<QTableWidget*>()) {
        if (candidate->columnCount() == column_count) {
            table = candidate;
        }
    }
    ASSERT_NE(table, nullptr);
    window_->fit_to_style();
    const auto* header = dynamic_cast<const CheckableHeaderView*>(table->horizontalHeader());
    ASSERT_NE(header, nullptr);
    int tallest = 0;
    for (const MetricColumn& item : metric_columns()) {
        const QSize needed = header->content_size(item.column);
        // Shown, a hidden column has the width it was given while hidden
        // (Qt says 0 for it until then).
        const bool hidden = table->isColumnHidden(item.column);
        table->setColumnHidden(item.column, false);
        EXPECT_GE(table->columnWidth(item.column), needed.width())
            << table->horizontalHeaderItem(item.column)->text().toStdString();
        table->setColumnHidden(item.column, hidden);
        tallest = std::max(tallest, needed.height());
    }
    EXPECT_GE(header->minimumHeight(), tallest);  // as tall as the tallest heading, shown or not
    QString long_heading;  // "PSNR YUV (6:1:1)", on two lines
    for (const MetricColumn& item : metric_columns()) {
        if (std::string_view(item.key) == "psnr_yuv_mse") {
            long_heading = table->horizontalHeaderItem(item.column)->text();
        }
    }
    EXPECT_EQ(long_heading.count(QLatin1Char('\n')), 1) << long_heading.toStdString();
}

// Light's looks: Dim's and Desert's colours in place of Light's, Desert's
// burnt orange while no accent is chosen; Light's own again, at once, when
// its look is Light; and Classic has none.
TEST_F(Appearance, ModernLightLooks) {
    theme::apply_theme("light", "dim", "dark");
    theme::apply_appearance(theme::Interface::modern, std::nullopt);
    if (theme::is_dark()) {
        GTEST_SKIP() << "This platform does not take the app's light.";
    }
    EXPECT_EQ(theme::look(), "dim");
    EXPECT_EQ(QApplication::palette().color(QPalette::Base), QColor("#e4e4e4"));
    EXPECT_EQ(theme::chart_colours().background, QColor("#e4e4e4"));
    EXPECT_EQ(theme::chart_colours(false).background, QColor("white"));  // exported light graphs keep theirs
    // A graph is drawn in the look too, and again when the look changes.
    ChartWidget chart;
    chart.resize(200, 120);
    EXPECT_EQ(chart.grab().toImage().pixelColor(2, 2), QColor("#e4e4e4"));
    theme::apply_theme("light", "desert", "dark");
    EXPECT_EQ(chart.grab().toImage().pixelColor(2, 2), QColor("#fffaef"));
    EXPECT_EQ(theme::look(), "desert");
    EXPECT_EQ(QApplication::palette().color(QPalette::Base), QColor("#fffaef"));
    EXPECT_EQ(QApplication::palette().color(QPalette::Accent), QColor("#903909"));
    EXPECT_EQ(theme::accent_shown(), QColor("#903909"));  // Settings' accent box shows it
    EXPECT_EQ(theme::color("link"), QColor("#1c5e75"));
    theme::apply_appearance(theme::Interface::modern, accent);
    EXPECT_EQ(QApplication::palette().color(QPalette::Accent), accent);  // the one chosen
    theme::apply_appearance(theme::Interface::modern, std::nullopt);
    theme::apply_theme("light", "mocha", "desert");  // each side's own looks only
    EXPECT_EQ(theme::look(), "light");
    theme::apply_theme("light", "light", "dark");
    EXPECT_EQ(theme::look(), "light");
    EXPECT_EQ(QApplication::palette().color(QPalette::Base), theme::modern_colour("card", false));
    EXPECT_EQ(theme::modern_colour("card", false), QColor("#ffffff"));
    EXPECT_NE(QApplication::palette().color(QPalette::Accent), QColor("#903909"));  // the system's again
    EXPECT_NE(theme::accent_shown(), QColor("#903909"));
}

}  // namespace
}  // namespace vml::app
