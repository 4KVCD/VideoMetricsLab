#include "text_panel.h"

#include <QApplication>
#include <QClipboard>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QScreen>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>

#include "i18n.h"
#include "theme.h"

namespace vml::app {

namespace {
constexpr int copied_ms = 1500;  // how long Copy says "Copied"
}  // namespace

TextPanel::TextPanel(const QString& title, QWidget* parent) : PopupPanel(parent) {
    setAccessibleName(title);
    auto* layout = new QVBoxLayout(this);
    auto* head = new QHBoxLayout;
    auto* heading = new QLabel(title);
    theme::set_role(heading, "title");
    head->addWidget(heading, 1);
    copy_ = new QPushButton(app::tr("Copy"));
    copy_->setToolTip(app::tr("Copy all of it"));
    head->addWidget(copy_);
    layout->addLayout(head);
    text_ = new QLabel;
    text_->setObjectName(QStringLiteral("panelText"));
    text_->setTextFormat(Qt::PlainText);
    text_->setTextInteractionFlags(Qt::TextSelectableByMouse | Qt::TextSelectableByKeyboard);
    text_->setWordWrap(true);
    text_->setFocusPolicy(Qt::StrongFocus);  // Tab reaches it, for keyboard selection
    layout->addWidget(text_);
    setMaximumWidth(fontMetrics().averageCharWidth() * 90);  // long lines wrapped, not one wide row
    copied_ = new QTimer(this);
    copied_->setSingleShot(true);
    connect(copied_, &QTimer::timeout, this, [this] { copy_->setText(app::tr("Copy")); });
    connect(copy_, &QPushButton::clicked, this, [this] {
        QApplication::clipboard()->setText(text());
        copy_->setText(app::tr("Copied"));
        copied_->start(copied_ms);
    });
}

void TextPanel::set_text(const QString& text) {
    if (text_->text() == text) {
        return;  // a selection kept while nothing changed
    }
    text_->setText(text);
    if (isVisible()) {
        adjustSize();
    }
}

QString TextPanel::text() const { return text_->text(); }

void TextPanel::open_under(QWidget* anchor) {
    set_anchor(anchor);
    adjustSize();
    QPoint at = anchor->mapToGlobal(QPoint(anchor->width() - width(), anchor->height() + 2));
    if (const QScreen* screen = anchor->screen()) {
        const QRect room = screen->availableGeometry();
        at.setX(std::clamp(at.x(), room.left(), std::max(room.left(), room.right() - width())));
        if (at.y() + height() > room.bottom()) {
            at.setY(anchor->mapToGlobal(QPoint(0, 0)).y() - height() - 2);  // above it where there is no room below
        }
    }
    move(at);
    show();
}

}  // namespace vml::app
