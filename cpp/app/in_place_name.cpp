#include "in_place_name.h"

#include <QComboBox>
#include <QEvent>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QSizePolicy>

#include <string>
#include <utility>

#include "theme.h"
#include "vml/python_text.h"

namespace vml::app {

InPlaceName::InPlaceName(QComboBox* list, QWidget* parent) : QStackedWidget(parent), list_(list) {
    addWidget(list_);
    edit_ = new QLineEdit;
    edit_->setObjectName(QStringLiteral("profileNameEdit"));
    edit_->installEventFilter(this);
    addWidget(edit_);
    setCurrentWidget(list_);
    // As wide as the list, whichever is shown: the field doesn't move or
    // grow as it turns into a line to type in.
    setSizePolicy(list_->sizePolicy());
    connect(edit_, &QLineEdit::returnPressed, this, &InPlaceName::commit);
    connect(edit_, &QLineEdit::textEdited, this, [this] { problem_->clear(); });  // being put right
    problem_ = new QLabel;
    problem_->setObjectName(QStringLiteral("profileNameProblem"));
    problem_->setTextFormat(Qt::PlainText);
    problem_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);  // never widens the box
    problem_->setMinimumHeight(problem_->fontMetrics().height());
    theme::style(problem_, "color: {failed}; font-weight: 600;");
}

void InPlaceName::ask(const QString& start, Check check, std::function<void(const QString& name)> done) {
    check_ = std::move(check);
    done_ = std::move(done);
    problem_->clear();
    edit_->setText(start);
    setCurrentWidget(edit_);
    edit_->setFocus();
    edit_->selectAll();
}

bool InPlaceName::asking() const { return currentWidget() == edit_; }

void InPlaceName::commit() {
    if (!asking()) {
        return;
    }
    const QString name = QString::fromStdString(std::string(python_strip(edit_->text().toStdString())));
    if (check_) {
        if (const std::optional<QString> problem = check_(name)) {
            problem_->setText(*problem);
            edit_->setFocus();
            edit_->selectAll();
            return;
        }
    }
    // Back to the list before `done`, which fills it again.
    const std::function<void(const QString&)> done = std::move(done_);
    check_ = nullptr;
    done_ = nullptr;
    setCurrentWidget(list_);
    if (done) {
        done(name);
    }
}

void InPlaceName::cancel() {
    if (!asking()) {
        return;
    }
    check_ = nullptr;
    done_ = nullptr;
    problem_->clear();
    const bool had_focus = edit_->hasFocus();
    setCurrentWidget(list_);
    if (had_focus) {
        list_->setFocus();
    }
}

bool InPlaceName::eventFilter(QObject* watched, QEvent* event) {
    if (watched == edit_) {
        if (event->type() == QEvent::KeyPress && static_cast<QKeyEvent*>(event)->key() == Qt::Key_Escape) {
            cancel();
            return true;
        }
        // A click elsewhere (or the window left): nothing kept. Not a popup
        // opening (the line's own context menu).
        if (event->type() == QEvent::FocusOut &&
            static_cast<QFocusEvent*>(event)->reason() != Qt::PopupFocusReason) {
            cancel();
        }
    }
    return QStackedWidget::eventFilter(watched, event);
}

}  // namespace vml::app
