#include "export_panel.h"

#include <QButtonGroup>
#include <QCheckBox>
#include <QComboBox>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QRadioButton>
#include <QScreen>
#include <QVBoxLayout>

#include <algorithm>

#include "i18n.h"
#include "results_export.h"
#include "theme.h"
#include "vml/settings.h"

namespace vml::app {

namespace {

// A section's heading, as the mockup's: muted, above its controls.
QLabel* heading(const QString& text) {
    auto* label = new QLabel(text);
    theme::style(label, "color: {muted};");
    return label;
}

// The most the On this tab link takes, in pixels.

bool has(const std::vector<std::string>& keys, const std::string& key) {
    return std::find(keys.begin(), keys.end(), key) != keys.end();
}

}  // namespace

ExportPanel::ExportPanel(Settings& settings, QWidget* parent) : PopupPanel(parent), settings_(settings) {
    setObjectName("exportPanel");
    theme::set_role(this, "card");
    setFrameShape(QFrame::StyledPanel);
    setAutoFillBackground(true);
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(14, 12, 14, 12);
    layout->setSpacing(6);
    auto* title = new QLabel(app::tr("Export"));
    theme::set_role(title, "title");
    QFont bold = title->font();
    bold.setBold(true);
    title->setFont(bold);
    layout->addWidget(title);

    // Which videos.
    layout->addWidget(heading(app::tr("Videos")));
    selection_ = new QRadioButton;
    all_ = new QRadioButton;
    auto* scope = new QButtonGroup(this);
    scope->addButton(selection_);
    scope->addButton(all_);
    connect(selection_, &QRadioButton::toggled, this, [this] { refresh_export_button(); });
    layout->addWidget(selection_);
    layout->addWidget(all_);

    // What: Settings' Exports include, the same boxes' meaning, as the
    // Settings tab names them.
    layout->addSpacing(4);
    layout->addWidget(heading(app::tr("Include")));
    auto* items = new QGridLayout;
    items->setContentsMargins(0, 0, 0, 0);
    int place = 0;
    for (const std::string& key : export_item_keys()) {
        auto* box = new QCheckBox(export_item_title(key));
        connect(box, &QCheckBox::toggled, this, [this] { store_choices(); });
        items_[key] = box;
        items->addWidget(box, place / 3, place % 3);
        ++place;
    }
    layout->addLayout(items);
    auto* formats = new QHBoxLayout;
    formats->addWidget(new QLabel(app::tr("In:")));
    for (const std::string& key : export_format_keys()) {
        auto* box = new QCheckBox(QString::fromStdString(key).toUpper());
        connect(box, &QCheckBox::toggled, this, [this] { store_choices(); });
        formats_[key] = box;
        formats->addWidget(box);
    }
    formats->addSpacing(12);
    formats->addWidget(new QLabel(app::tr("As:")));
    form_ = new QComboBox;
    form_->addItem(app::tr("A .zip file"), true);
    form_->addItem(app::tr("Loose files"), false);
    connect(form_, &QComboBox::currentIndexChanged, this, [this] { store_choices(); });
    formats->addWidget(form_);
    formats->addStretch(1);
    layout->addLayout(formats);
    auto* same = new QLabel(app::tr("The same choices as Settings > Storage > Exports include: a change here is "
                                    "saved there. Graphs: always PNG."));
    same->setWordWrap(true);
    theme::style(same, "color: {muted};");
    layout->addWidget(same);

    // The weak segment rules, the picker Saved Results' Export menu had.
    auto* rules_row = new QHBoxLayout;
    rules_row->addWidget(new QLabel(app::tr("Weak segments by")));
    rules_ = new QComboBox;
    rules_->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    rules_->setToolTip(app::tr("The rules the weak segments are found by: they are not saved with the scores, but "
                               "found from them now."));
    connect(rules_, &QComboBox::currentIndexChanged, this, [this](int index) {
        if (!filling_ && index >= 0 && pick_rules) {
            pick_rules(rules_->itemData(index).toString());
        }
    });
    rules_row->addWidget(rules_, 1);
    layout->addLayout(rules_row);

    // Where.
    layout->addSpacing(4);
    layout->addWidget(heading(app::tr("To")));
    auto* folder_row = new QHBoxLayout;
    folder_label_ = new QLabel;
    folder_label_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    folder_row->addWidget(folder_label_, 1);
    auto* change = new QPushButton(app::tr("Change..."));
    connect(change, &QPushButton::clicked, this, [this] {
        if (change_folder) {
            change_folder();
        }
    });
    folder_row->addWidget(change);
    layout->addLayout(folder_row);

    auto* buttons = new QHBoxLayout;
    buttons->addStretch(1);
    auto* cancel = new QPushButton(app::tr("Cancel"));
    connect(cancel, &QPushButton::clicked, this, [this] { hide(); });
    buttons->addWidget(cancel);
    export_ = new QPushButton(app::tr("Export"));
    theme::set_role(export_, "primary");
    export_->setDefault(true);
    connect(export_, &QPushButton::clicked, this, [this] {
        const bool selection = selection_->isChecked();
        hide();
        if (export_videos) {
            export_videos(selection);
        }
    });
    buttons->addWidget(export_);
    layout->addLayout(buttons);
    setMinimumWidth(420);
}

void ExportPanel::refresh() {
    filling_ = true;
    const Videos shown = videos ? videos() : Videos{};
    selection_->setVisible(!shown.selection.isEmpty());
    selection_->setText(shown.selection);
    selection_->setEnabled(shown.selection_refused.isEmpty());
    selection_->setToolTip(shown.selection_refused);
    all_->setText(shown.all);
    all_->setEnabled(shown.all_refused.isEmpty());
    all_->setToolTip(shown.all_refused);
    // The selection where there is one to export, as the tab is looked at
    // (asked before the panel is shown: not by isVisible).
    if (!shown.selection.isEmpty() && selection_->isEnabled()) {
        selection_->setChecked(true);
    } else {
        all_->setChecked(true);
    }
    for (const auto& [key, box] : items_) {
        box->setChecked(has(settings_.export_items, key));
    }
    for (const auto& [key, box] : formats_) {
        box->setChecked(has(settings_.export_formats, key));
    }
    form_->setCurrentIndex(settings_.export_zip ? 0 : 1);
    rules_->clear();
    const QString picked = rule_choice ? rule_choice() : QString();
    for (const auto& [choice, text] : rule_choices ? rule_choices() : std::vector<std::pair<QString, QString>>{}) {
        rules_->addItem(text, choice);
        if (choice == picked) {
            rules_->setCurrentIndex(rules_->count() - 1);
        }
    }
    folder_label_->setText(folder ? folder() : QString());
    filling_ = false;
    refresh_export_button();
}

void ExportPanel::open_under(QWidget* anchor) {
    set_anchor(anchor);
    refresh();
    adjustSize();
    QPoint at = anchor->mapToGlobal(QPoint(anchor->width() - width(), anchor->height() + 2));
    if (const QScreen* screen = anchor->screen()) {
        const QRect room = screen->availableGeometry();
        at.setX(std::clamp(at.x(), room.left(), std::max(room.left(), room.right() - width())));
    }
    move(at);
    show();
}

void ExportPanel::store_choices() {
    if (filling_) {
        return;
    }
    settings_.export_items.clear();
    for (const std::string& key : export_item_keys()) {
        if (items_.at(key)->isChecked()) {
            settings_.export_items.push_back(key);
        }
    }
    settings_.export_formats.clear();
    for (const std::string& key : export_format_keys()) {
        if (formats_.at(key)->isChecked()) {
            settings_.export_formats.push_back(key);
        }
    }
    settings_.export_zip = form_->currentData().toBool();
    if (choices_changed) {
        choices_changed();
    }
    refresh_export_button();
}

void ExportPanel::refresh_export_button() {
    // Greyed, saying why, where nothing ticked can be exported or the
    // videos chosen cannot be.
    const QRadioButton* chosen = selection_->isChecked() ? selection_ : all_;
    const bool nothing = export_contents(exported_items(settings_)).isEmpty();
    export_->setEnabled(!nothing && chosen->isEnabled());
    export_->setToolTip(nothing ? export_action_text(ExportScope::all_videos, {}) : chosen->toolTip());
}

}  // namespace vml::app
