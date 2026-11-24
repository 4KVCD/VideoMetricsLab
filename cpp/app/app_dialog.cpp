#include "app_dialog.h"

#include <QCheckBox>
#include <QColorDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPainter>
#include <QPainterPath>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QVBoxLayout>

#include "i18n.h"
#include "theme.h"
#include "title_bar.h"

namespace vml::app {

// =============================================================== AppDialog

AppDialog::AppDialog(QWidget* parent, const QString& title) : QDialog(parent) {
    setWindowTitle(title);
    auto* outer = new QVBoxLayout(this);
    outer->setContentsMargins(0, 0, 0, 0);
    outer->setSpacing(0);
#ifdef _WIN32
    if (TitleBar::wanted()) {
        bar_ = new TitleBar(this, true);
        bar_->prepare();  // before the native window is made
        outer->addWidget(bar_);
    }
#endif
    body_ = new QWidget(this);
    outer->addWidget(body_, 1);
}

void AppDialog::showEvent(QShowEvent* event) {
#ifdef _WIN32
    if (bar_ != nullptr) {
        bar_->attach();  // the native window is made now
    }
#endif
    QDialog::showEvent(event);
}

bool AppDialog::nativeEvent(const QByteArray& type, void* message, qintptr* result) {
#ifdef _WIN32
    if (bar_ != nullptr && bar_->native_event(message, result)) {
        return true;
    }
#endif
    return QDialog::nativeEvent(type, message, result);
}

QPushButton* add_dialog_buttons(AppDialog& dialog, QBoxLayout* layout, const QString& ok_text) {
    auto* row = new QHBoxLayout;
    row->addStretch(1);
    auto* ok = new QPushButton(ok_text.isEmpty() ? app::tr("OK") : ok_text);
    theme::set_role(ok, "primary");
    ok->setDefault(true);
    auto* cancel = new QPushButton(app::tr("Cancel"));
    cancel->setAutoDefault(false);
    QObject::connect(ok, &QPushButton::clicked, &dialog, &QDialog::accept);
    QObject::connect(cancel, &QPushButton::clicked, &dialog, &QDialog::reject);
    row->addWidget(ok);
    row->addWidget(cancel);
    layout->addSpacing(6);
    layout->addLayout(row);
    return ok;
}

// =================================================================== icons

namespace {

QColor ink_of(MessageKind kind) {
    switch (kind) {
    case MessageKind::warning:
        return theme::color("stale");  // amber
    case MessageKind::failure:
    case MessageKind::danger:
        return theme::color("failed");
    case MessageKind::information:
    case MessageKind::question:
        break;
    }
    return theme::accent_shown();
}

}  // namespace

QPixmap message_icon(MessageKind kind, int size, qreal ratio) {
    QPixmap pixmap(QSize(size, size) * ratio);
    pixmap.setDevicePixelRatio(ratio);
    pixmap.fill(Qt::transparent);
    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing);
    const QColor ink = ink_of(kind);
    const bool solid = kind == MessageKind::danger;
    const bool triangle = kind == MessageKind::warning || kind == MessageKind::danger;
    const qreal line = std::max(1.5, size / 16.0);

    // The shape: a circle, or a triangle with round corners for a warning;
    // a light wash of the colour with a ring of it, or filled for danger.
    QPainterPath shape;
    QRectF glyph_area;
    if (triangle) {
        const qreal inset = line * 1.5;
        shape.moveTo(size / 2.0, inset);
        shape.lineTo(size - inset, size - inset * 1.4);
        shape.lineTo(inset, size - inset * 1.4);
        shape.closeSubpath();
        glyph_area = QRectF(0, size * 0.28, size, size * 0.66);
    } else {
        shape.addEllipse(QRectF(line, line, size - 2 * line, size - 2 * line));
        glyph_area = QRectF(0, 0, size, size);
    }
    QColor wash = ink;
    wash.setAlphaF(0.16f);
    painter.setPen(QPen(ink, line, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    painter.setBrush(solid ? QBrush(ink) : QBrush(wash));
    painter.drawPath(shape);

    // The glyph: i, ?, ! or a cross.
    const QColor glyph_ink = solid ? theme::text_on(ink) : ink;
    if (kind == MessageKind::failure) {
        const qreal arm = size * 0.17;
        const QPointF centre(size / 2.0, size / 2.0);
        painter.setPen(QPen(glyph_ink, line * 1.25, Qt::SolidLine, Qt::RoundCap));
        painter.drawLine(centre + QPointF(-arm, -arm), centre + QPointF(arm, arm));
        painter.drawLine(centre + QPointF(-arm, arm), centre + QPointF(arm, -arm));
        return pixmap;
    }
    QFont font = painter.font();
    font.setBold(true);
    font.setPixelSize(qRound(size * (triangle ? 0.46 : 0.56)));
    painter.setFont(font);
    painter.setPen(glyph_ink);
    const QString glyph = kind == MessageKind::question      ? QStringLiteral("?")
                          : kind == MessageKind::information ? QStringLiteral("i")
                                                             : QStringLiteral("!");
    painter.drawText(glyph_area, Qt::AlignCenter, glyph);
    return pixmap;
}

// ================================================================ messages

namespace {

void set_button_role(QPushButton* button, MessageButton::Role role) {
    if (role == MessageButton::Role::primary) {
        theme::set_role(button, "primary");
    } else if (role == MessageButton::Role::danger) {
        theme::set_role(button, "danger");
    }
}

// The message's dialog, its answer kept in `answer` as it closes.
AppDialog* make_message(QWidget* parent, const Message& message, MessageAnswer& answer, QCheckBox*& check) {
    auto* dialog = new AppDialog(parent, message.title);
    dialog->layout()->setSizeConstraint(QLayout::SetFixedSize);  // as its text needs, not resized
    auto* layout = new QVBoxLayout(dialog->body());
    layout->setContentsMargins(20, 18, 20, 16);
    layout->setSpacing(14);

    auto* row = new QHBoxLayout;
    row->setSpacing(14);
    auto* icon = new QLabel;
    const int size = 32;
    icon->setPixmap(message_icon(message.kind, size, dialog->devicePixelRatioF()));
    icon->setFixedSize(size, size);
    row->addWidget(icon, 0, Qt::AlignTop);
    auto* texts = new QVBoxLayout;
    texts->setSpacing(6);
    if (!message.headline.isEmpty()) {
        auto* headline = new QLabel(message.headline);
        headline->setObjectName("messageHeadline");
        headline->setWordWrap(true);
        QFont font = headline->font();
        font.setBold(true);
        headline->setFont(font);
        if (message.kind == MessageKind::danger) {
            QPalette colours = headline->palette();
            colours.setColor(QPalette::WindowText, ink_of(MessageKind::danger));
            headline->setPalette(colours);
        }
        texts->addWidget(headline);
    }
    auto* text = new QLabel(message.text);
    text->setObjectName("messageText");
    text->setWordWrap(true);
    text->setMinimumWidth(420);
    text->setMaximumWidth(560);
    text->setTextInteractionFlags(Qt::TextSelectableByMouse | Qt::LinksAccessibleByMouse);
    text->setOpenExternalLinks(true);
    texts->addWidget(text);
    row->addLayout(texts, 1);
    layout->addLayout(row);

    check = nullptr;
    if (!message.check.isEmpty()) {
        check = new QCheckBox(message.check);
        layout->addWidget(check);
    }

    std::vector<MessageButton> buttons = message.buttons;
    if (buttons.empty()) {
        buttons.push_back({app::tr("OK"), MessageButton::Role::primary});
    }
    const int escape = message.escape_button >= 0 ? message.escape_button : static_cast<int>(buttons.size()) - 1;
    auto* bar = new QHBoxLayout;
    if (!message.details.isEmpty()) {
        // Shown on asking, above the buttons, as QMessageBox's details were.
        auto* details = new QPlainTextEdit(message.details);
        details->setReadOnly(true);
        details->setFixedHeight(180);
        details->setVisible(false);
        layout->addWidget(details);
        auto* toggle = new QPushButton(app::tr("Show details"));
        toggle->setAutoDefault(false);
        theme::set_role(toggle, "quiet");
        QObject::connect(toggle, &QPushButton::clicked, details, [details, toggle] {
            details->setVisible(!details->isVisible());
            toggle->setText(details->isVisible() ? app::tr("Hide details") : app::tr("Show details"));
        });
        bar->addWidget(toggle);
    }
    bar->addStretch(1);
    for (int i = 0; i < static_cast<int>(buttons.size()); ++i) {
        auto* button = new QPushButton(buttons[static_cast<std::size_t>(i)].text);
        set_button_role(button, buttons[static_cast<std::size_t>(i)].role);
        button->setAutoDefault(false);
        button->setDefault(i == message.default_button);
        QObject::connect(button, &QPushButton::clicked, dialog, [dialog, &answer, i] {
            answer.button = i;
            dialog->done(QDialog::Accepted);
        });
        bar->addWidget(button);
        if (i == message.default_button) {
            button->setFocus();
        }
    }
    layout->addLayout(bar);
    // Esc and the close button: Esc's answer.
    QObject::connect(dialog, &QDialog::rejected, dialog, [&answer, escape] { answer.button = escape; });
    return dialog;
}

}  // namespace

MessageAnswer show_message(QWidget* parent, const Message& message) {
    MessageAnswer answer;
    QCheckBox* check = nullptr;
    AppDialog* dialog = make_message(parent, message, answer, check);
    dialog->exec();
    answer.checked = check != nullptr && check->isChecked();
    delete dialog;
    return answer;
}

AppDialog* message_dialog(QWidget* parent, const Message& message, std::function<void(MessageAnswer)> answered) {
    // The answer outlives this call: kept by the dialog, which deletes it
    // with itself.
    auto* answer = new MessageAnswer;
    QCheckBox* check = nullptr;
    AppDialog* dialog = make_message(parent, message, *answer, check);
    dialog->setAttribute(Qt::WA_DeleteOnClose);
    QObject::connect(dialog, &QDialog::finished, dialog, [answer, check, answered = std::move(answered)] {
        answer->checked = check != nullptr && check->isChecked();
        if (answered) {
            answered(*answer);
        }
    });
    QObject::connect(dialog, &QObject::destroyed, [answer] { delete answer; });
    return dialog;
}

// ============================================================ the others

ColorChoice choose_color(QWidget* parent, const QString& title, const QColor& initial, const QString& reset) {
    AppDialog dialog(parent, title);
    auto* layout = new QVBoxLayout(dialog.body());
    // Qt's picker as a part of the dialog, its own buttons left out.
    auto* picker = new QColorDialog(initial);
    picker->setWindowFlags(Qt::Widget);
    picker->setOptions(QColorDialog::DontUseNativeDialog | QColorDialog::NoButtons);
    layout->addWidget(picker);
    // The picker is a dialog too: Esc or Enter inside it closes it alone,
    // which would leave this one open and empty. Its answer is this one's.
    QObject::connect(picker, &QDialog::accepted, &dialog, &QDialog::accept);
    QObject::connect(picker, &QDialog::rejected, &dialog, &QDialog::reject);
    auto* bar = new QHBoxLayout;
    constexpr int reset_code = 2;  // done()'s, beside Accepted and Rejected
    if (!reset.isEmpty()) {
        auto* back = new QPushButton(reset);
        back->setAutoDefault(false);
        QObject::connect(back, &QPushButton::clicked, &dialog, [&dialog] { dialog.done(reset_code); });
        bar->addWidget(back);
    }
    bar->addStretch(1);
    auto* ok = new QPushButton(app::tr("OK"));
    theme::set_role(ok, "primary");
    ok->setDefault(true);
    auto* cancel = new QPushButton(app::tr("Cancel"));
    cancel->setAutoDefault(false);
    QObject::connect(ok, &QPushButton::clicked, &dialog, &QDialog::accept);
    QObject::connect(cancel, &QPushButton::clicked, &dialog, &QDialog::reject);
    bar->addWidget(ok);
    bar->addWidget(cancel);
    layout->addLayout(bar);
    const int result = dialog.exec();
    if (result == reset_code) {
        return {ColorChoice::Outcome::reset, {}};
    }
    if (result != QDialog::Accepted) {
        return {};
    }
    return {ColorChoice::Outcome::chosen, picker->currentColor()};
}

std::optional<QString> ask_line(QWidget* parent, const QString& title, const QString& label, const QString& text) {
    AppDialog dialog(parent, title);
    auto* layout = new QVBoxLayout(dialog.body());
    layout->setContentsMargins(20, 18, 20, 16);
    layout->addWidget(new QLabel(label));
    auto* entry = new QLineEdit(text);
    entry->setMinimumWidth(300);
    entry->selectAll();
    layout->addWidget(entry);
    add_dialog_buttons(dialog, layout);
    if (dialog.exec() != QDialog::Accepted) {
        return std::nullopt;
    }
    return entry->text();
}

ProgressDialog::ProgressDialog(QWidget* parent, const QString& title, const QString& label) : AppDialog(parent, title) {
    auto* layout = new QVBoxLayout(body());
    layout->setContentsMargins(20, 18, 20, 16);
    label_ = new QLabel(label);
    label_->setMinimumWidth(360);
    label_->setWordWrap(true);
    layout->addWidget(label_);
    bar_ = new QProgressBar;
    bar_->setRange(0, 0);
    bar_->setTextVisible(false);
    layout->addWidget(bar_);
    auto* row = new QHBoxLayout;
    row->addStretch(1);
    auto* cancel = new QPushButton(app::tr("Cancel"));
    connect(cancel, &QPushButton::clicked, this, &QDialog::reject);
    row->addWidget(cancel);
    layout->addLayout(row);
}

void ProgressDialog::set_label(const QString& text) { label_->setText(text); }
void ProgressDialog::set_range(int minimum, int maximum) { bar_->setRange(minimum, maximum); }
void ProgressDialog::set_value(int value) { bar_->setValue(value); }

}  // namespace vml::app
