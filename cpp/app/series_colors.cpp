#include "series_colors.h"

#include <QAbstractButton>
#include <QColorDialog>
#include <QDialogButtonBox>
#include <QMouseEvent>
#include <QPushButton>
#include <QStyle>
#include <QStyleOptionViewItem>

#include <system_error>

#include "i18n.h"
#include "vml/platform/system.h"

namespace vml::app {

VideoColors::VideoColors(QObject* parent) : QObject(parent) {}

QString VideoColors::path_key(const std::filesystem::path& path) {
    std::error_code error;
    std::filesystem::path full = std::filesystem::weakly_canonical(std::filesystem::absolute(path, error), error);
    if (error) {
        full = std::filesystem::absolute(path, error);
    }
    QString key = QString::fromStdString(platform::path_text(full));
#ifdef _WIN32
    // As os.path.normcase: one case and one kind of slash.
    key = key.replace('/', '\\').toLower();
#endif
    return key;
}

QColor VideoColors::for_path(const std::filesystem::path& path) {
    const QString key = path_key(path);
    if (!colors_.contains(key)) {
        const QColor color(series_palette[colors_.size() % series_palette.size()]);
        colors_[key] = color;
        defaults_[key] = color;
    }
    return colors_.at(key);
}

void VideoColors::set_color(const std::filesystem::path& path, const QColor& color) {
    // Both tabs are told only of a valid colour that is a new one.
    if (!color.isValid()) {
        return;
    }
    if (for_path(path).name() != color.name()) {
        const QString key = path_key(path);
        colors_[key] = QColor(color.name());  // opaque, as the name has it
        emit changed(key, colors_[key]);
    }
}

void VideoColors::choose(const std::filesystem::path& path, QWidget* parent) {
    QColorDialog dialog(for_path(path), parent);
    dialog.setWindowTitle(app::tr("Choose video color"));
    dialog.setOption(QColorDialog::DontUseNativeDialog, true);
    if (auto* buttons = dialog.findChild<QDialogButtonBox*>()) {
        QAbstractButton* reset = buttons->addButton(app::tr("Reset color"), QDialogButtonBox::ResetRole);
        const QColor first = defaults_.at(path_key(path));
        connect(reset, &QAbstractButton::clicked, &dialog, [&dialog, first] { dialog.setCurrentColor(first); });
    }
    if (dialog.exec() == QDialog::Accepted) {
        set_color(path, dialog.currentColor());
    }
}

bool ColorSwatchDelegate::editorEvent(QEvent* event, QAbstractItemModel* model, const QStyleOptionViewItem& option,
                                      const QModelIndex& index) {
    const QEvent::Type type = event->type();
    if (type == QEvent::MouseButtonPress || type == QEvent::MouseButtonRelease ||
        type == QEvent::MouseButtonDblClick) {
        const auto* mouse = static_cast<QMouseEvent*>(event);
        QStyleOptionViewItem styled(option);
        initStyleOption(&styled, index);
        const QStyle* style = styled.widget != nullptr ? styled.widget->style() : nullptr;
        if (mouse->button() == Qt::LeftButton && style != nullptr &&
            style->subElementRect(QStyle::SE_ItemViewItemDecoration, &styled, styled.widget)
                .contains(mouse->position().toPoint())) {
            if (type == QEvent::MouseButtonRelease) {
                emit clicked(index.row());
            }
            return true;
        }
    }
    return QStyledItemDelegate::editorEvent(event, model, option, index);
}

}  // namespace vml::app
