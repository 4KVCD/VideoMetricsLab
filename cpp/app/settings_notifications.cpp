// Settings > Notifications: a message when a run ends, to Pushover and
// Discord (notifications.h), when and how it is sent, the services' keys and
// how they are stored, and the message's templates with a preview.
#include <QCheckBox>
#include <QComboBox>
#include <QDateTime>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QTextCursor>
#include <QTextDocument>
#include <QTimer>
#include <QVBoxLayout>

#include <cmath>
#include <string>

#include "i18n.h"
#include "notifier.h"
#include "settings_panel.h"
#include "theme.h"
#include "vml/metrics.h"
#include "vml/notifications.h"

namespace vml::app {

namespace {

QString text_of(const std::string& text) { return QString::fromStdString(text); }

// A template's editor: `lines` lines high with no scroll bar, plain text.
// The height counts what surrounds the text too: the document's margin, the
// frame and the viewport's margins (a fixed allowance cut the last line).
QPlainTextEdit* template_editor(const std::string& text, int lines) {
    auto* editor = new QPlainTextEdit(text_of(text));
    editor->setTabChangesFocus(true);
    const QMargins margins = editor->contentsMargins();
    const int around = static_cast<int>(std::ceil(2 * editor->document()->documentMargin())) +
                       2 * editor->frameWidth() + margins.top() + margins.bottom();
    editor->setFixedHeight(editor->fontMetrics().lineSpacing() * lines + around + 2);
    return editor;
}

// What a field holds, for the Insert field menu.
QString fields_help() {
    return app::tr(
        "Fields are filled in when the message is sent. In the title and the message: {outcome} how the run "
        "ended, {duration} how long it took, {reference} the reference's file name, {finished} when it ended, "
        "{app} the app and its version, {video_count}, {scored_count} and {failed_count}, and {videos}: each test "
        "video's line, one after another. In a test video's line: {name} its file name, {result} how it ended, "
        "{frames} the frames compared, {scores} each metric's mean, {scores.min} (or another statistic) each "
        "metric's, and a metric's own: {vmaf}, {vmaf.min}, {ssimulacra2.worst_1}... A metric the video has no "
        "score for shows a dash. Leave a line empty to leave it out.");
}

}  // namespace

QWidget* SettingsPanel::build_notifications_page() {
    test_notifier_ = new Notifier(this);
    connect(test_notifier_, &Notifier::sent, this, [this](const QString& service, const QString& error) {
        status->setText(error.isEmpty()
                            ? app::tr("Notification sent to {service}.", {{"service", service}})
                            : app::tr("Couldn't send the notification to {service}: {error}",
                                      {{"service", service}, {"error", error}}));
    });
    template_typing_ = new QTimer(this);
    template_typing_->setSingleShot(true);
    template_typing_->setInterval(600);
    connect(template_typing_, &QTimer::timeout, this, [this] { on_edited(); });

    auto* page = new QWidget;
    auto* columns = new QHBoxLayout(page);
    auto* left = new QVBoxLayout;
    for (QGroupBox* box : {build_notify_when_box(), build_pushover_box(), build_discord_box(), build_keys_box()}) {
        left->addWidget(box);
    }
    left->addStretch(1);
    auto* right = new QVBoxLayout;
    right->addWidget(build_message_box());
    right->addStretch(1);
    columns->addLayout(left, 2);
    columns->addLayout(right, 3);
    show_secret_note();
    show_notification_preview();
    return page;
}

QGroupBox* SettingsPanel::build_notify_when_box() {
    auto* box = new QGroupBox(app::tr("When a run ends"));
    auto* layout = new QVBoxLayout(box);
    notify_finished = new QCheckBox(app::tr("Finished"));
    notify_with_failures = new QCheckBox(app::tr("Finished with failures"));
    notify_cancelled = new QCheckBox(app::tr("Cancelled"));
    notify_finished->setChecked(settings_.notify_finished);
    notify_with_failures->setChecked(settings_.notify_with_failures);
    notify_cancelled->setChecked(settings_.notify_cancelled);
    auto* outcomes = new QHBoxLayout;
    for (QCheckBox* check : {notify_finished, notify_with_failures, notify_cancelled}) {
        connect(check, &QCheckBox::toggled, this, [this] { on_edited(); });
        outcomes->addWidget(check);
    }
    outcomes->addStretch(1);
    layout->addLayout(outcomes);
    notify_min_minutes = new QSpinBox;
    notify_min_minutes->setRange(0, 24 * 60);
    notify_min_minutes->setSuffix(app::tr(" min"));
    notify_min_minutes->setSpecialValueText(app::tr("Every run"));
    notify_min_minutes->setValue(settings_.notify_min_minutes);
    connect(notify_min_minutes, &QSpinBox::valueChanged, this, [this] { on_edited(); });
    auto* length = new QHBoxLayout;
    length->addWidget(new QLabel(app::tr("Only for runs of at least:")));
    length->addWidget(notify_min_minutes);
    length->addStretch(1);
    layout->addLayout(length);
    return box;
}

QGroupBox* SettingsPanel::build_pushover_box() {
    auto* box = new QGroupBox("Pushover");
    auto* form = new QFormLayout(box);
    pushover_enabled = new QCheckBox(app::tr("Send to {service}", {{"service", "Pushover"}}));
    pushover_enabled->setChecked(settings_.pushover_enabled);
    connect(pushover_enabled, &QCheckBox::toggled, this, [this] { on_edited(); });
    form->addRow(pushover_enabled);
    pushover_user = new QLineEdit;
    pushover_token = new QLineEdit;
    for (QLineEdit* edit : {pushover_user, pushover_token}) {
        edit->setEchoMode(QLineEdit::Password);
        connect(edit, &QLineEdit::editingFinished, this, [this] { on_edited(); });
    }
    show_secret(pushover_user, settings_.pushover_user);
    show_secret(pushover_token, settings_.pushover_token);
    form->addRow(app::tr("User key:"), pushover_user);
    form->addRow(app::tr("Application token:"), pushover_token);
    pushover_priority = new QComboBox;
    pushover_priority->addItem(app::tr("Lowest"), -2);
    pushover_priority->addItem(app::tr("Low"), -1);
    pushover_priority->addItem(app::tr("Normal"), 0);
    pushover_priority->addItem(app::tr("High"), 1);
    pushover_priority->setCurrentIndex(std::max(0, pushover_priority->findData(settings_.pushover_priority)));
    connect(pushover_priority, &QComboBox::currentIndexChanged, this, [this] { on_edited(); });
    pushover_test = new QPushButton(app::tr("Send test"));
    connect(pushover_test, &QPushButton::clicked, this, [this] { send_notification_test("Pushover"); });
    auto* row = new QHBoxLayout;
    row->addWidget(pushover_priority);
    row->addStretch(1);
    row->addWidget(pushover_test);
    form->addRow(app::tr("Priority:"), row);
    return box;
}

QGroupBox* SettingsPanel::build_discord_box() {
    auto* box = new QGroupBox("Discord");
    auto* form = new QFormLayout(box);
    discord_enabled = new QCheckBox(app::tr("Send to {service}", {{"service", "Discord"}}));
    discord_enabled->setChecked(settings_.discord_enabled);
    connect(discord_enabled, &QCheckBox::toggled, this, [this] { on_edited(); });
    form->addRow(discord_enabled);
    discord_webhook = new QLineEdit;
    discord_webhook->setEchoMode(QLineEdit::Password);
    connect(discord_webhook, &QLineEdit::editingFinished, this, [this] { on_edited(); });
    show_secret(discord_webhook, settings_.discord_webhook);
    discord_test = new QPushButton(app::tr("Send test"));
    connect(discord_test, &QPushButton::clicked, this, [this] { send_notification_test("Discord"); });
    auto* row = new QHBoxLayout;
    row->addWidget(discord_webhook, 1);
    row->addWidget(discord_test);
    form->addRow(app::tr("Webhook URL:"), row);
    return box;
}

QGroupBox* SettingsPanel::build_keys_box() {
    auto* box = new QGroupBox(app::tr("Keys"));
    auto* layout = new QVBoxLayout(box);
    secret_storage = new QComboBox;
    secret_storage->addItem(app::tr("Encrypted for this Windows user"), true);
    secret_storage->addItem(app::tr("As plain text in settings.json"), false);
    const bool can_encrypt = secrets_can_be_encrypted();
    if (!can_encrypt) {
        // Only Windows can encrypt them for its user, for now.
        secret_storage->setItemData(0, 0, Qt::UserRole - 1);  // disabled
    }
    secret_storage->setCurrentIndex(settings_.notify_secrets_encrypted && can_encrypt ? 0 : 1);
    connect(secret_storage, &QComboBox::currentIndexChanged, this, [this] {
        on_edited();
        show_secret_note();
    });
    show_keys = new QCheckBox(app::tr("Show keys"));
    connect(show_keys, &QCheckBox::toggled, this, [this](bool shown) {
        for (QLineEdit* edit : {pushover_user, pushover_token, discord_webhook}) {
            edit->setEchoMode(shown ? QLineEdit::Normal : QLineEdit::Password);
        }
    });
    auto* row = new QHBoxLayout;
    row->addWidget(new QLabel(app::tr("Stored:")));
    row->addWidget(secret_storage);
    row->addStretch(1);
    row->addWidget(show_keys);
    layout->addLayout(row);
    secret_note = new QLabel;
    secret_note->setWordWrap(true);
    layout->addWidget(secret_note);
    return box;
}

QGroupBox* SettingsPanel::build_message_box() {
    auto* box = new QGroupBox(app::tr("Message"));
    auto* layout = new QVBoxLayout(box);
    notify_presets = new QPushButton(app::tr("Start from"));
    auto* presets = new QMenu(notify_presets);
    for (const NotificationPreset& preset : notification_presets()) {
        presets->addAction(preset.name, this, [this, preset] {
            QWidget* const editing = last_template_;  // not the last one set here
            notify_title->setText(text_of(preset.title));
            notify_message->setPlainText(text_of(preset.message));
            notify_video_line->setPlainText(text_of(preset.video_line));
            last_template_ = editing;
            template_typing_->stop();
            show_notification_preview();
            on_edited();
        });
    }
    notify_presets->setMenu(presets);
    insert_field = new QPushButton(app::tr("Insert field"));
    insert_field->setToolTip(fields_help());
    auto* fields = new QMenu(insert_field);
    QMenu* run_fields = fields->addMenu(app::tr("The run"));
    for (const char* field : {"{outcome}", "{duration}", "{reference}", "{finished}", "{app}", "{video_count}",
                              "{scored_count}", "{failed_count}", "{videos}"}) {
        run_fields->addAction(field, this, [this, field] { insert_notification_field(field); });
    }
    QMenu* video_fields = fields->addMenu(app::tr("Each test video"));
    for (const char* field : {"{name}", "{result}", "{frames}"}) {
        video_fields->addAction(field, this, [this, field] { insert_notification_field(field); });
    }
    video_fields->addSeparator();
    for (const std::string_view statistic : notification_statistics()) {
        const QString field = statistic == "mean" ? QString("{scores}") : QString("{scores.%1}").arg(statistic);
        video_fields->addAction(field, this, [this, field] { insert_notification_field(field); });
    }
    for (const MetricDefinition& metric : metrics()) {
        QMenu* metric_fields = video_fields->addMenu(text_of(std::string(metric.label)));
        for (const std::string_view statistic : notification_statistics()) {
            const QString key = text_of(std::string(metric.key));
            const QString field =
                statistic == "mean" ? QString("{%1}").arg(key) : QString("{%1.%2}").arg(key).arg(statistic);
            metric_fields->addAction(field, this, [this, field] { insert_notification_field(field); });
        }
    }
    insert_field->setMenu(fields);
    auto* buttons = new QHBoxLayout;
    buttons->addWidget(notify_presets);
    buttons->addWidget(insert_field);
    buttons->addStretch(1);
    layout->addLayout(buttons);

    auto* form = new QFormLayout;
    notify_title = new QLineEdit(text_of(settings_.notify_title));
    notify_message = template_editor(settings_.notify_message, 5);
    notify_video_line = template_editor(settings_.notify_video_line, 4);
    form->addRow(app::tr("Title:"), notify_title);
    form->addRow(app::tr("Message:"), notify_message);
    form->addRow(app::tr("Each test video:"), notify_video_line);
    layout->addLayout(form);
    connect(notify_title, &QLineEdit::textEdited, this, [this] {
        last_template_ = notify_title;
        show_notification_preview();
        template_typing_->start();
    });
    for (QPlainTextEdit* editor : {notify_message, notify_video_line}) {
        connect(editor, &QPlainTextEdit::textChanged, this, [this, editor] {
            last_template_ = editor;
            show_notification_preview();
            template_typing_->start();
        });
    }
    last_template_ = notify_message;

    auto* preview_label = new QLabel(app::tr("Preview"));
    theme::style(preview_label, "color: {muted};");
    layout->addWidget(preview_label);
    notify_preview = new QLabel;
    notify_preview->setObjectName("notificationPreview");
    notify_preview->setWordWrap(true);
    notify_preview->setTextFormat(Qt::RichText);
    notify_preview->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    notify_preview->setTextInteractionFlags(Qt::TextSelectableByMouse);
    notify_preview->setFrameShape(QFrame::StyledPanel);
    notify_preview->setMargin(8);
    layout->addWidget(notify_preview);
    return box;
}

void SettingsPanel::show_secret(QLineEdit* edit, const std::string& stored) {
    const std::optional<std::string> secret = secret_of(stored);
    edit->setText(secret ? text_of(*secret) : QString());
    edit->setPlaceholderText(secret ? QString()
                                    : app::tr("Saved by another Windows user or on another PC: enter it again"));
}

void SettingsPanel::read_notification_controls() {
    settings_.notify_finished = notify_finished->isChecked();
    settings_.notify_with_failures = notify_with_failures->isChecked();
    settings_.notify_cancelled = notify_cancelled->isChecked();
    settings_.notify_min_minutes = notify_min_minutes->value();
    settings_.notify_title = notify_title->text().toStdString();
    settings_.notify_message = notify_message->toPlainText().toStdString();
    settings_.notify_video_line = notify_video_line->toPlainText().toStdString();
    settings_.pushover_enabled = pushover_enabled->isChecked();
    settings_.pushover_priority = pushover_priority->currentData().toInt();
    settings_.discord_enabled = discord_enabled->isChecked();
    const bool encrypted = secret_storage->currentData().toBool();
    settings_.notify_secrets_encrypted = encrypted;
    // Stored again only where it changed, or how it is to be stored did; one
    // that cannot be read back here is kept until something is typed.
    const auto store = [encrypted](std::string& stored, const QLineEdit* edit) {
        const std::string typed = edit->text().trimmed().toStdString();
        const std::optional<std::string> now = secret_of(stored);
        if (!now && typed.empty()) {
            return;
        }
        if (!now || *now != typed || (encrypted && !typed.empty()) != secret_is_encrypted(stored)) {
            stored = stored_secret(typed, encrypted);
        }
    };
    store(settings_.pushover_user, pushover_user);
    store(settings_.pushover_token, pushover_token);
    store(settings_.discord_webhook, discord_webhook);
}

void SettingsPanel::show_notification_preview() {
    const NotifiedRun run = sample_notified_run(app::tr("Finished"));
    const std::string line = notify_video_line->toPlainText().toStdString();
    const QString title = text_of(render_notification(notify_title->text().toStdString(), run, line));
    const QString message = text_of(render_notification(notify_message->toPlainText().toStdString(), run, line));
    QString html;
    if (!title.trimmed().isEmpty()) {
        html += "<b>" + title.toHtmlEscaped() + "</b>";
    }
    if (!message.trimmed().isEmpty()) {
        html += (html.isEmpty() ? "" : "<br>") + message.toHtmlEscaped().replace('\n', "<br>");
    }
    notify_preview->setText(html.isEmpty() ? app::tr("Finished").toHtmlEscaped() : html);
}

void SettingsPanel::show_secret_note() {
    const bool encrypted = secret_storage->currentData().toBool();
    secret_note->setText(
        encrypted ? app::tr("Encrypted: only this Windows user, on this PC, can read them back. On another PC, or "
                            "after Windows is reinstalled, enter them again.")
                  : QStringLiteral("⚠ ") +
                        app::tr("Plain text: anyone who can read settings.json (a backup, a synced or shared "
                                "folder, a copy of your profile, another program) can read these keys and send "
                                "messages as you. Choose Encrypted unless you need to copy them between PCs."));
    theme::style(secret_note, encrypted ? "color: {muted};" : "");
}

void SettingsPanel::insert_notification_field(const QString& field) {
    if (auto* editor = qobject_cast<QPlainTextEdit*>(last_template_)) {
        editor->insertPlainText(field);
        editor->setFocus();
    } else if (auto* edit = qobject_cast<QLineEdit*>(last_template_)) {
        edit->insert(field);
        edit->setFocus();
        show_notification_preview();
        template_typing_->start();
    }
}

void SettingsPanel::send_notification_test(const QString& service) {
    template_typing_->stop();
    on_edited();  // what is typed now, saved
    const NotificationPlan plan = plan_notification(
        settings_, sample_notified_run(app::tr("Test notification")),
        QDateTime::currentDateTimeUtc().toString(Qt::ISODate).toStdString(), service.toStdString());
    if (plan.requests.empty()) {
        status->setText(notification_problems(plan).join(' '));
        return;
    }
    status->setText(app::tr("Sending a test to {service}...", {{"service", service}}));
    test_notifier_->send(plan.requests);
}

}  // namespace vml::app
