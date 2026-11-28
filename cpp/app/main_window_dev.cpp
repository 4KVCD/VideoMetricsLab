// The window's side of the developers' tools (dev_tools.h): what each of
// their keys does here. Nothing in this file runs unless VML_DEV=1.
#include <QApplication>
#include <QFile>
#include <QPushButton>
#include <QTabWidget>

#include <algorithm>

#include "dev_tools.h"
#include "i18n.h"
#include "main_window.h"
#include "run_worker.h"
#include "settings_panel.h"
#include "theme.h"
#include "vml/app_log.h"
#include "vml/platform/system.h"

namespace vml::app {

namespace {

const app_log::Logger dev_logger("vml.dev");

// The steps the faked states key goes through, in order.
enum class FakeStep { running, paused, cancelled, failed, cannot_start, count };

}  // namespace

void MainWindow::add_dev_tools() {
    if (!dev::enabled()) {
        return;
    }
    dev_logger.info("Developer tools on (VML_DEV=1)");
    dev_overlay_ = new dev::Overlay(this);
    dev::install(this, [this](dev::Command command) { run_dev_command(command); });
}

void MainWindow::run_dev_command(dev::Command command) {
    switch (command) {
    case dev::Command::dump_state: {
        const std::filesystem::path file = dev::output_folder() / "ui-state.json";
        QFile out(QString::fromStdString(platform::path_text(file)));
        if (out.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
            out.write(QByteArray::fromStdString(dev::describe(this).dump(1)));
        }
        dev_logger.info("Window state written: " + platform::path_text(file));
        break;
    }
    case dev::Command::save_picture: {
        // The window in front: a dialog or popup over the main window is
        // the one pictured.
        QWidget* shown = QApplication::activePopupWidget();
        if (shown == nullptr) {
            shown = QApplication::activeModalWidget();
        }
        if (shown == nullptr) {
            shown = QApplication::activeWindow() != nullptr ? QApplication::activeWindow() : this;
        }
        const std::filesystem::path file = dev::output_folder() / "window.png";
        shown->grab().save(QString::fromStdString(platform::path_text(file)), "PNG");
        dev_logger.info("Window saved: " + platform::path_text(file));
        break;
    }
    case dev::Command::next_look: {
        // The looks in Settings > Appearance's order, light then dark.
        std::vector<std::pair<bool, QString>> looks;
        for (const bool dark : {false, true}) {
            for (const auto& [name, value] : theme::look_choices(dark)) {
                looks.emplace_back(dark, value);
            }
        }
        const auto now = std::find(looks.begin(), looks.end(), std::pair{theme::is_dark(), theme::look()});
        const auto& [dark, value] = now == looks.end() || now + 1 == looks.end() ? looks.front() : *(now + 1);
        settings_panel().choose_theme(dark ? QStringLiteral("dark") : QStringLiteral("light"));
        settings_panel().choose_look(dark, value);
        show_theme_in_title();
        dev_logger.info("Look: " + value.toStdString() + (dark ? " (dark)" : " (light)"));
        break;
    }
    case dev::Command::tab_order:
    case dev::Command::names: {
        const dev::Overlay::Mode wanted =
            command == dev::Command::tab_order ? dev::Overlay::Mode::tab_order : dev::Overlay::Mode::names;
        dev_overlay_->set_mode(dev_overlay_->mode() == wanted ? dev::Overlay::Mode::none : wanted);
        break;
    }
    case dev::Command::next_language: {
        // English, each translation in the Settings list's order, then the
        // long-text language, then English again. The long-text one is
        // asked for by VML_DEV_LANGUAGE (main.cpp), never saved: Settings'
        // language list has no entry for it.
        if (run_active_) {
            dev_logger.info("Not restarted: a run is under way");
            break;
        }
        QStringList codes;
        for (const Language& entry : languages()) {
            codes.append(entry.code);
        }
        const qsizetype now = codes.indexOf(language());
        if (language() == long_text_language) {
            qunsetenv("VML_DEV_LANGUAGE");
            settings_.language = codes.front().toStdString();
        } else if (now + 1 == codes.size()) {
            qputenv("VML_DEV_LANGUAGE", long_text_language.toUtf8());
        } else {
            settings_.language = codes.at(now < 0 ? 0 : now + 1).toStdString();
        }
        dev_logger.info("Next language from the restart");
        restart_for_dev();
        break;
    }
    case dev::Command::forced_state:
        next_fake_state();
        break;
    }
}

void MainWindow::restart_for_dev() {
    save_settings();
    restart_on_close_ = true;
    close();
    if (!closing_) {
        restart_on_close_ = false;  // the close was refused
    }
}

void MainWindow::next_fake_state() {
    // A run in progress, paused, cancelled, failed, then FFmpeg not usable
    // (the start's "can't start" message), each faked: nothing is read or
    // scored (FakeRun). A step that cannot be taken now (a run
    // of the user's own going, nothing to run) starts the cycle over.
    const auto step = static_cast<FakeStep>(dev_fake_step_);
    dev_fake_step_ = (dev_fake_step_ + 1) % static_cast<int>(FakeStep::count);
    const bool fake_running = worker_ != nullptr && worker_->isRunning() && dev_fake_run_active_;
    switch (step) {
    case FakeStep::running:
    case FakeStep::failed: {
        if (worker_ != nullptr && worker_->isRunning()) {
            dev_logger.info("Faked run not started: a run is under way");
            dev_fake_step_ = 0;
            return;
        }
        const std::vector<int> checked = checked_rows();
        if (!source_info_ || checked.empty()) {
            show_status(QStringLiteral("VML_DEV: open a reference and tick a test video for a faked run"));
            dev_fake_step_ = 0;
            return;
        }
        std::vector<std::shared_ptr<RowData>> job_rows;
        std::optional<std::vector<VmafJob>> jobs = plan_jobs(checked, job_rows);
        if (!jobs || jobs->empty()) {
            dev_fake_step_ = 0;
            return;
        }
        fake_run_ = step == FakeStep::running ? FakeRun::endless : FakeRun::failing;
        begin_run(checked, std::move(job_rows), 0);
        start_run_worker(std::move(*jobs));
        return;
    }
    case FakeStep::paused:
    case FakeStep::cancelled:
        if (!fake_running) {
            dev_fake_step_ = 0;
            next_fake_state();
            return;
        }
        (step == FakeStep::paused ? pause_button_ : cancel_button_)->click();
        return;
    case FakeStep::cannot_start: {
        // As the start shows it when FFmpeg cannot be run, then the
        // tools as they are again.
        ToolStatus missing{"ffmpeg", "C:\\vml-dev\\missing\\ffmpeg.exe", false, std::nullopt,
                           "not found (faked by VML_DEV)"};
        ToolStatus probe_missing{"ffprobe", "C:\\vml-dev\\missing\\ffprobe.exe", false, std::nullopt,
                                 "not found (faked by VML_DEV)"};
        check_ffmpeg(true, ToolsStatus{std::move(missing), std::move(probe_missing)});
        check_ffmpeg(false);
        return;
    }
    case FakeStep::count:
        return;
    }
}

}  // namespace vml::app
