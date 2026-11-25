// The tab bar's Export button's panel (issue #224, docs/ARCHITECTURE.md,
// "Exports"): the archive, "everything for these encodes", in one place.
// Which videos (the open tab's selection, or every scored video), what it
// includes (Settings' Exports include itself, not a copy: a change here is
// saved there), the weak segment rules, and the folder it is written to;
// and a link to the open tab's own one-click save, for the one-off.
#pragma once

#include <QString>

#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "popup_panel.h"

class QCheckBox;
class QComboBox;
class QLabel;
class QPushButton;
class QRadioButton;

namespace vml {
struct Settings;
}

namespace vml::app {

class SaveButton;

class ExportPanel : public PopupPanel {
    Q_OBJECT

public:
    // `settings` are the window's: the panel's ticks are its export_items,
    // export_formats and export_zip.
    explicit ExportPanel(Settings& settings, QWidget* parent = nullptr);

    // Which videos the archive can be of, asked as the panel opens: the open
    // tab's selection where that tab lists videos ("The selected video on
    // Videos: <name>"; empty: none), and every scored video ("All scored
    // videos (3)", with how many). Why the selection or all cannot be
    // exported, where they cannot (their radio greyed, the reason its
    // tooltip).
    struct Videos {
        QString selection;
        QString selection_refused;
        QString all;
        QString all_refused;
    };
    std::function<Videos()> videos;
    // The weak segment rules the archive finds weak segments by: the choices
    // (choice, text), the one picked, and picking one.
    std::function<std::vector<std::pair<QString, QString>>()> rule_choices;
    std::function<QString()> rule_choice;
    std::function<void(const QString& choice)> pick_rules;
    // The folder it is written to, and Change... (which may change it).
    std::function<QString()> folder;
    std::function<void()> change_folder;
    // The ticks changed (settings' export choices): saved, and shown on
    // the Settings tab.
    std::function<void()> choices_changed;
    // The open tab's own save icon, for the link; none on a tab without one.
    std::function<SaveButton*()> tab_save;
    // Export: of the selection (`selection`) or of every scored video.
    std::function<void(bool selection)> export_videos;

    // Shows the panel under `anchor`, its right edge at the anchor's, with
    // what it offers made again from the settings and the window as they are.
    // A second click on `anchor` closes it (PopupPanel::set_anchor).
    void open_under(QWidget* anchor);
    // The same without showing it (the tests).
    void refresh();

    // For the tests.
    QRadioButton* selection_radio() const { return selection_; }
    QRadioButton* all_radio() const { return all_; }
    QCheckBox* item_box(const std::string& key) const { return items_.at(key); }
    QCheckBox* format_box(const std::string& key) const { return formats_.at(key); }
    QComboBox* form_combo() const { return form_; }
    QComboBox* rules_combo() const { return rules_; }
    QPushButton* export_button() const { return export_; }
    QPushButton* tab_link() const { return tab_link_; }
    QLabel* folder_label() const { return folder_label_; }

private:
    // The ticks written to the settings (choices_changed), and what that
    // changes in the panel.
    void store_choices();
    void refresh_export_button();

    Settings& settings_;
    QRadioButton* selection_ = nullptr;
    QRadioButton* all_ = nullptr;
    std::map<std::string, QCheckBox*> items_;    // by export_item_keys' key
    std::map<std::string, QCheckBox*> formats_;  // by export_format_keys' key
    QComboBox* form_ = nullptr;                  // a .zip file (true) or loose files (false)
    QComboBox* rules_ = nullptr;
    QLabel* folder_label_ = nullptr;
    QPushButton* tab_link_ = nullptr;
    QLabel* tab_label_ = nullptr;
    QPushButton* export_ = nullptr;
    bool filling_ = false;  // refresh is setting the controls: not the user's changes
};

}  // namespace vml::app
