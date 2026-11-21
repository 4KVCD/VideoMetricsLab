// Settings > Notifications: a message when a run ends, to the services
// (notifications.h: Pushover, Discord, Gotify, ntfy, Pushbullet, Slack,
// Telegram, a webhook), when and how it is sent, each service's keys and how
// they are stored, and the message's templates with a preview.
#include <QCheckBox>
#include <QComboBox>
#include <QDateTime>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMenu>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QStackedWidget>
#include <QStyle>
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
    // Keys first: Show keys is read as the services' key fields are made.
    QGroupBox* keys = build_keys_box();
    for (QGroupBox* box : {build_notify_when_box(), build_services_box(), keys}) {
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

QLineEdit* SettingsPanel::secret_edit(const std::string& stored) {
    auto* edit = new QLineEdit;
    edit->setEchoMode(show_keys != nullptr && show_keys->isChecked() ? QLineEdit::Normal : QLineEdit::Password);
    connect(edit, &QLineEdit::editingFinished, this, [this] { on_edited(); });
    show_secret(edit, stored);
    secret_edits_.push_back(edit);
    return edit;
}

QWidget* SettingsPanel::service_page(const QString& service, QCheckBox*& enabled, bool on,
                                     const std::vector<std::pair<QString, QWidget*>>& rows, const QString& hint) {
    auto* page = new QWidget;
    auto* form = new QFormLayout(page);
    form->setContentsMargins(0, 0, 0, 0);
    enabled = new QCheckBox(app::tr("Send to {service}", {{"service", service}}));
    enabled->setChecked(on);
    form->addRow(enabled);
    for (const auto& [label, widget] : rows) {
        if (label.isEmpty()) {
            form->addRow(widget);
        } else {
            form->addRow(label, widget);
        }
    }
    auto* note = new QLabel(hint);
    note->setWordWrap(true);
    note->setOpenExternalLinks(true);
    note->setTextFormat(Qt::RichText);
    theme::style(note, "color: {muted};");
    auto* test = new QPushButton(app::tr("Send test"));
    connect(test, &QPushButton::clicked, this, [this, service] { send_notification_test(service); });
    auto* row = new QHBoxLayout;
    row->addWidget(note, 1);
    row->addWidget(test, 0, Qt::AlignTop);
    form->addRow(row);
    // The list's tick and the page's box are one setting.
    QCheckBox* const box = enabled;
    connect(box, &QCheckBox::toggled, this, [this, box, service](bool checked) {
        for (int at = 0; at < notify_services->count(); ++at) {
            if (QListWidgetItem* item = notify_services->item(at); item->data(Qt::UserRole).toString() == service) {
                const QSignalBlocker blocker(notify_services);
                item->setCheckState(checked ? Qt::Checked : Qt::Unchecked);
            }
        }
        on_edited();
    });
    return page;
}

QGroupBox* SettingsPanel::build_services_box() {
    auto* box = new QGroupBox(app::tr("Services"));
    auto* layout = new QHBoxLayout(box);
    notify_services = new QListWidget;
    notify_services->setObjectName("notificationServices");
    service_pages = new QStackedWidget;

    // A page's note: its words as they are (escaped), {link} a link.
    const auto hint = [](const QString& words, const char* url = nullptr, const QString& shown = QString()) {
        QString html = words.toHtmlEscaped();
        if (url != nullptr) {
            html.replace("{link}", QStringLiteral("<a href=\"%1\">%2</a>")
                                       .arg(QString::fromLatin1(url), shown.toHtmlEscaped()));
        }
        return html;
    };

    // Pushover
    pushover_user = secret_edit(settings_.pushover_user);
    pushover_token = secret_edit(settings_.pushover_token);
    pushover_priority = new QComboBox;
    pushover_priority->addItem(app::tr("Lowest"), -2);
    pushover_priority->addItem(app::tr("Low"), -1);
    pushover_priority->addItem(app::tr("Normal"), 0);
    pushover_priority->addItem(app::tr("High"), 1);
    pushover_priority->setCurrentIndex(std::max(0, pushover_priority->findData(settings_.pushover_priority)));
    connect(pushover_priority, &QComboBox::currentIndexChanged, this, [this] { on_edited(); });
    QWidget* pushover = service_page(
        "Pushover", pushover_enabled, settings_.pushover_enabled,
        {{app::tr("User key:"), pushover_user},
         {app::tr("Application token:"), pushover_token},
         {app::tr("Priority:"), pushover_priority}},
        hint(app::tr("Your user key is on {link}'s front page once signed in; make an application there for its token."), "https://pushover.net", "pushover.net"));

    // Discord
    discord_webhook = secret_edit(settings_.discord_webhook);
    discord_mention = new QLineEdit(text_of(settings_.discord_mention));
    discord_mention->setPlaceholderText(app::tr("No one"));
    discord_mention->setToolTip(app::tr("Whom the message mentions, so that they get a push: <@&role ID> for a "
                                        "role, <@user ID> for a person (Discord's Developer Mode shows IDs: "
                                        "right-click, Copy ID), @here or @everyone. Leave it empty for no one."));
    connect(discord_mention, &QLineEdit::editingFinished, this, [this] { on_edited(); });
    discord_mention_when = new QComboBox;
    discord_mention_when->addItem(app::tr("When a run has failures or is cancelled"), "problems");
    discord_mention_when->addItem(app::tr("Every run"), "always");
    discord_mention_when->setCurrentIndex(
        std::max(0, discord_mention_when->findData(text_of(settings_.discord_mention_when))));
    connect(discord_mention_when, &QComboBox::currentIndexChanged, this, [this] { on_edited(); });
    discord_silent = new QCheckBox(app::tr("Send without a push or a sound when no one is mentioned"));
    discord_silent->setChecked(settings_.discord_silent);
    connect(discord_silent, &QCheckBox::toggled, this, [this] { on_edited(); });
    QWidget* discord = service_page(
        "Discord", discord_enabled, settings_.discord_enabled,
        {{app::tr("Webhook URL:"), discord_webhook},
         {app::tr("Mention:"), discord_mention},
         {app::tr("Mention them:"), discord_mention_when},
         {QString(), discord_silent}},
        hint(app::tr("Make a webhook in the channel's settings, Integrations > Webhooks, and copy its URL. Each test "
                "video is a field of its own when the message lists them.")));

    // Gotify
    gotify_url = new QLineEdit(text_of(settings_.gotify_url));
    gotify_url->setPlaceholderText("https://gotify.example.org");
    connect(gotify_url, &QLineEdit::editingFinished, this, [this] { on_edited(); });
    gotify_token = secret_edit(settings_.gotify_token);
    gotify_priority = new QSpinBox;
    gotify_priority->setRange(0, 10);
    gotify_priority->setValue(settings_.gotify_priority);
    connect(gotify_priority, &QSpinBox::valueChanged, this, [this] { on_edited(); });
    QWidget* gotify = service_page(
        "Gotify", gotify_enabled, settings_.gotify_enabled,
        {{app::tr("Server:"), gotify_url},
         {app::tr("Application token:"), gotify_token},
         {app::tr("Priority:"), gotify_priority}},
        hint(app::tr("Your Gotify server's address, and the token of an application made in its Apps tab. An http:// "
                "address sends the token unencrypted: use it only on your own network.")));

    // ntfy
    ntfy_url = new QLineEdit(text_of(settings_.ntfy_url));
    ntfy_url->setPlaceholderText("https://ntfy.sh");
    connect(ntfy_url, &QLineEdit::editingFinished, this, [this] { on_edited(); });
    ntfy_topic = secret_edit(settings_.ntfy_topic);
    ntfy_topic->setToolTip(app::tr("Anyone who knows a topic's name on a public server can read its messages: "
                                   "choose one that can't be guessed. Letters, digits, - and _, up to 64."));
    ntfy_token = secret_edit(settings_.ntfy_token);
    if (secret_of(settings_.ntfy_token)) {  // else it says to enter it again
        ntfy_token->setPlaceholderText(app::tr("Only for a topic that needs one"));
    }
    ntfy_priority = new QComboBox;
    ntfy_priority->addItem(app::tr("Lowest"), 1);
    ntfy_priority->addItem(app::tr("Low"), 2);
    ntfy_priority->addItem(app::tr("Normal"), 3);
    ntfy_priority->addItem(app::tr("High"), 4);
    ntfy_priority->addItem(app::tr("Highest"), 5);
    ntfy_priority->setCurrentIndex(std::max(0, ntfy_priority->findData(settings_.ntfy_priority)));
    connect(ntfy_priority, &QComboBox::currentIndexChanged, this, [this] { on_edited(); });
    QWidget* ntfy = service_page(
        "ntfy", ntfy_enabled, settings_.ntfy_enabled,
        {{app::tr("Server:"), ntfy_url},
         {app::tr("Topic:"), ntfy_topic},
         {app::tr("Access token:"), ntfy_token},
         {app::tr("Priority:"), ntfy_priority}},
        hint(app::tr("Subscribe to the topic in the ntfy app ({link}). A topic on your own server can need an access "
                "token."), "https://ntfy.sh", "ntfy.sh"));

    // Pushbullet
    pushbullet_token = secret_edit(settings_.pushbullet_token);
    pushbullet_device = new QLineEdit(text_of(settings_.pushbullet_device));
    pushbullet_device->setPlaceholderText(app::tr("Every device"));
    connect(pushbullet_device, &QLineEdit::editingFinished, this, [this] { on_edited(); });
    QWidget* pushbullet = service_page(
        "Pushbullet", pushbullet_enabled, settings_.pushbullet_enabled,
        {{app::tr("Access token:"), pushbullet_token}, {app::tr("Device iden:"), pushbullet_device}},
        hint(app::tr("Make an access token in {link}, Settings > Account."), "https://www.pushbullet.com/#settings/account", "pushbullet.com"));

    // Slack
    slack_webhook = secret_edit(settings_.slack_webhook);
    QWidget* slack = service_page(
        "Slack", slack_enabled, settings_.slack_enabled, {{app::tr("Webhook URL:"), slack_webhook}},
        hint(app::tr("Add Incoming Webhooks to a Slack app, pick the channel, and copy the webhook's URL "
                "(https://hooks.slack.com/...).")));

    // Telegram
    telegram_token = secret_edit(settings_.telegram_token);
    telegram_chat = new QLineEdit(text_of(settings_.telegram_chat));
    telegram_chat->setPlaceholderText(app::tr("A chat's ID, or @channelname"));
    connect(telegram_chat, &QLineEdit::editingFinished, this, [this] { on_edited(); });
    telegram_silent = new QCheckBox(app::tr("Send without a sound"));
    telegram_silent->setChecked(settings_.telegram_silent);
    connect(telegram_silent, &QCheckBox::toggled, this, [this] { on_edited(); });
    QWidget* telegram = service_page(
        "Telegram", telegram_enabled, settings_.telegram_enabled,
        {{app::tr("Bot token:"), telegram_token}, {app::tr("Chat:"), telegram_chat}, {QString(), telegram_silent}},
        hint(app::tr("Make a bot with {link} for its token, and send it a message: the chat's ID is in "
                "api.telegram.org/bot<token>/getUpdates."), "https://t.me/BotFather", "@BotFather"));

    // Webhook
    webhook_url = secret_edit(settings_.webhook_url);
    webhook_header = secret_edit(settings_.webhook_header);
    if (secret_of(settings_.webhook_header)) {  // else it says to enter it again
        webhook_header->setPlaceholderText(app::tr("Optional: Authorization: Bearer ..."));
    }
    QWidget* webhook = service_page(
        "Webhook", webhook_enabled, settings_.webhook_enabled,
        {{app::tr("URL:"), webhook_url}, {app::tr("Header:"), webhook_header}},
        hint(app::tr("Any address that takes a POST (Home Assistant, n8n, a script): it gets the run as JSON, with the "
                "message and each test video's statistics. An http:// address sends it unencrypted: use it only on "
                "your own network.")));

    for (QWidget* page : {pushover, discord, gotify, ntfy, pushbullet, slack, telegram, webhook}) {
        service_pages->addWidget(page);
    }
    for (const std::string_view name : notification_services()) {
        auto* item = new QListWidgetItem(text_of(std::string(name)), notify_services);
        item->setData(Qt::UserRole, text_of(std::string(name)));
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(service_enabled(settings_, name) ? Qt::Checked : Qt::Unchecked);
    }
    const std::vector<QCheckBox*> boxes{pushover_enabled,   discord_enabled, gotify_enabled,   ntfy_enabled,
                                        pushbullet_enabled, slack_enabled,   telegram_enabled, webhook_enabled};
    connect(notify_services, &QListWidget::itemChanged, this, [this, boxes](QListWidgetItem* item) {
        boxes[static_cast<std::size_t>(notify_services->row(item))]->setChecked(item->checkState() == Qt::Checked);
    });
    connect(notify_services, &QListWidget::currentRowChanged, service_pages, &QStackedWidget::setCurrentIndex);
    notify_services->setCurrentRow(0);
    // Every service in sight, with no scroll bar: as wide as the longest
    // name with its box, as tall as all the rows.
    notify_services->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    notify_services->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    notify_services->setFixedWidth(notify_services->sizeHintForColumn(0) + 2 * notify_services->frameWidth() +
                                   notify_services->style()->pixelMetric(QStyle::PM_IndicatorWidth) + 24);
    notify_services->setMinimumHeight(notify_services->sizeHintForRow(0) * notify_services->count() +
                                      2 * notify_services->frameWidth() + 4);
    layout->addWidget(notify_services);
    layout->addWidget(service_pages, 1);
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
        for (QLineEdit* edit : secret_edits_) {
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
    settings_.discord_mention = discord_mention->text().trimmed().toStdString();
    settings_.discord_mention_when = discord_mention_when->currentData().toString().toStdString();
    settings_.discord_silent = discord_silent->isChecked();
    settings_.gotify_enabled = gotify_enabled->isChecked();
    settings_.gotify_url = gotify_url->text().trimmed().toStdString();
    settings_.gotify_priority = gotify_priority->value();
    settings_.ntfy_enabled = ntfy_enabled->isChecked();
    settings_.ntfy_url = ntfy_url->text().trimmed().toStdString();
    settings_.ntfy_priority = ntfy_priority->currentData().toInt();
    settings_.pushbullet_enabled = pushbullet_enabled->isChecked();
    settings_.pushbullet_device = pushbullet_device->text().trimmed().toStdString();
    settings_.slack_enabled = slack_enabled->isChecked();
    settings_.telegram_enabled = telegram_enabled->isChecked();
    settings_.telegram_chat = telegram_chat->text().trimmed().toStdString();
    settings_.telegram_silent = telegram_silent->isChecked();
    settings_.webhook_enabled = webhook_enabled->isChecked();
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
    store(settings_.gotify_token, gotify_token);
    store(settings_.ntfy_topic, ntfy_topic);
    store(settings_.ntfy_token, ntfy_token);
    store(settings_.pushbullet_token, pushbullet_token);
    store(settings_.slack_webhook, slack_webhook);
    store(settings_.telegram_token, telegram_token);
    store(settings_.webhook_url, webhook_url);
    store(settings_.webhook_header, webhook_header);
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
