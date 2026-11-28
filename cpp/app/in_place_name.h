// A name typed where the list of names is (issue #239): a new or renamed
// weak segment profile's name in the profile field itself, in place of a
// window asking for it. The field's list gives way to a line to type in,
// the name selected; Enter keeps it, Esc or a click elsewhere leaves the
// list as it was. A name that can't be used is said on the field's own line
// (problem_line, placed by the owner under the field), and the typing goes
// on.
#pragma once

#include <QStackedWidget>
#include <QString>

#include <functional>
#include <optional>

class QComboBox;
class QLabel;
class QLineEdit;

namespace vml::app {

class InPlaceName : public QStackedWidget {
    Q_OBJECT

public:
    // Holds `list`, shown until a name is asked for.
    explicit InPlaceName(QComboBox* list, QWidget* parent = nullptr);

    // Asks for a name in the field, starting at `start`, selected: `check`
    // says why a name typed can't be used (nothing: it can, the spaces
    // around it already taken off), and `done` takes it.
    using Check = std::function<std::optional<QString>(const QString& name)>;
    void ask(const QString& start, Check check, std::function<void(const QString& name)> done);
    // What Enter does: keeps the name typed, or says why it can't be kept.
    void commit();
    // What Esc and a click elsewhere do: the list back, nothing kept.
    void cancel();
    bool asking() const;

    QLineEdit* line_edit() const { return edit_; }
    // Why the last name typed was refused, in the failure colour; empty
    // otherwise. The owner places it under the field: its line is kept
    // from the start, so nothing moves as it comes.
    QLabel* problem_line() const { return problem_; }

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    QComboBox* list_ = nullptr;
    QLineEdit* edit_ = nullptr;
    QLabel* problem_ = nullptr;
    Check check_;
    std::function<void(const QString&)> done_;
};

}  // namespace vml::app
