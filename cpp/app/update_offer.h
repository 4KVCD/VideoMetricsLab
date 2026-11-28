// The update check: asks GitHub for the latest release once, when the app
// starts, and offers a newer one, unless the user chose to skip it: as a
// badge in VML Modern's title buttons, whose press opens a panel under it
// (Download, Skip this version, Later and what's new), else as a dialog at
// once.
#pragma once

#include <QObject>
#include <QPointer>
#include <QString>

#include <filesystem>
#include <functional>
#include <optional>
#include <string>

#include "app_dialog.h"
#include "vml/settings.h"
#include "vml/update_check.h"

namespace vml::app {

class PopupPanel;
class TitleButtons;

class UpdateOffer : public QObject {
public:
    // What the offer needs of its window.
    struct Window {
        QWidget* widget;  // the dialog's parent, and this object's
        Settings& settings;  // check_for_updates, skipped_update_version
        std::function<void()> save_settings;
        // The title buttons, or nullptr (VML Classic): asked at each use, as
        // they can be made after the window.
        std::function<TitleButtons*()> badge;
        std::function<std::filesystem::path()> cache_folder;  // named in the dialog
    };

    explicit UpdateOffer(Window window);

    // Asks GitHub for the latest release, without holding the window. Called
    // once, when the app starts: building a window never uses the network.
    void check();
    // The release the badge offers, shown (the badge's press).
    void show_pending();
    // The badge's panel while it is open, for the tests.
    PopupPanel* panel() const { return panel_; }

private:
    // GitHub's answer (`body`), or why it could not be asked (`error`).
    void on_answer(const std::string& body, const QString& error);
    // Offers a newer release, unless the user chose to skip it.
    void offer(const Release& release);
    // The update offered: the panel under the badge, or without one (VML
    // Classic) the dialog at once.
    void show(const Release& release);
    void show_panel(const Release& release, QWidget* badge);
    // What the offer's buttons do: Download opens the package, Skip
    // forgets this release until a newer one, Later only closes.
    void download(const Release& release) const;
    void skip(const Release& release);
    // That updating keeps the saved results and settings, and where they are.
    QString keeps_text() const;

    Window window_;
    QPointer<AppDialog> box_;         // the dialog, while it is open
    QPointer<PopupPanel> panel_;      // the badge's panel, while it is open
    QPointer<QWidget> hooked_badge_;  // the badge whose press opens it (connected once)
    std::optional<Release> pending_;  // offered on the badge, shown when it is clicked
};

}  // namespace vml::app
