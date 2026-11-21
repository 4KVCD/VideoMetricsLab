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
#include <QGroupBox>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QRegularExpression>
#include <QSet>
#include <QTabWidget>
#include <QWidget>

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "main_window.h"
#include "theme.h"

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

    // Modern in dark or light: each role drawn as its rule says, a style
    // sheet Qt reads, and the window's tabs drawn in it.
    static void check_modern(bool dark) {
        theme::apply_theme(dark ? "dark" : "light");
        std::vector<std::string> warnings = warnings_while([] {
            theme::apply_appearance(theme::Interface::modern, accent);
        });
        ASSERT_EQ(theme::current_interface(), theme::Interface::modern);
        const char* name = dark ? "modern-dark" : "modern-light";
        // Modern's layout: the container padded, its labels shown and
        // Classic's hidden, the box untitled.
        EXPECT_GE(child_place().x(), 12) << name;
        EXPECT_GE(child_place().y(), 10) << name;
        EXPECT_TRUE(modern_label_->isVisibleTo(layouts_)) << name;
        EXPECT_FALSE(classic_label_->isVisibleTo(layouts_)) << name;
        EXPECT_TRUE(box_->title().isEmpty()) << name;

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
}

TEST_F(Appearance, ModernLight) {
    check_modern(false);
}

TEST_F(Appearance, ModernDark) {
    theme::apply_theme("dark");
    if (!theme::is_dark()) {
        GTEST_SKIP() << "This platform does not take the app's dark (Qt's offscreen one): run with VML_TEST_ON_SCREEN=1.";
    }
    check_modern(true);
}

}  // namespace
}  // namespace vml::app
