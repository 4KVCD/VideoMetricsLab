#include "vml/saved_index.h"

#include <sqlite3.h>

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <system_error>
#include <unordered_map>
#include <utility>

#include "metric_cache_parts.h"
#include "vml/app_log.h"
#include "vml/metric_cache.h"
#include "vml/platform/system.h"

namespace vml {

namespace {

// The form of the index's tables (PRAGMA user_version): raised with every
// change to them. An index of an older form is made again from the folders:
// it holds nothing they do not.
constexpr int index_form = 1;

constexpr const char* schema = R"(
CREATE TABLE comparison(
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    readable INTEGER NOT NULL,
    label TEXT NOT NULL,
    source TEXT NOT NULL,
    distorted TEXT NOT NULL,
    source_identity TEXT NOT NULL,
    source_sampled TEXT NOT NULL,
    saved INTEGER NOT NULL,
    bytes INTEGER NOT NULL
);
CREATE TABLE record(
    comparison INTEGER PRIMARY KEY REFERENCES comparison(id) ON DELETE CASCADE,
    recipe TEXT NOT NULL,
    identity TEXT NOT NULL
);
CREATE TABLE score(
    comparison INTEGER NOT NULL REFERENCES comparison(id) ON DELETE CASCADE,
    file TEXT NOT NULL,
    key TEXT NOT NULL,
    saved INTEGER NOT NULL,
    bytes INTEGER NOT NULL,
    PRIMARY KEY(comparison, file)
) WITHOUT ROWID;
CREATE TABLE pending(name TEXT PRIMARY KEY) WITHOUT ROWID;
)";

// How long a process waits for another's write before giving up: a first
// reading of a large cache writes for a few seconds.
constexpr int busy_ms = 30000;

// How many comparisons a first reading writes in one transaction: what was
// read is kept should it be stopped, and other processes get their turn.
constexpr std::size_t batch = 500;

struct SqliteError : std::runtime_error {
    int code;
    SqliteError(sqlite3* db, int result)
        : std::runtime_error(db ? sqlite3_errmsg(db) : sqlite3_errstr(result)), code(result) {}
};

void check(sqlite3* db, int result) {
    if (result != SQLITE_OK && result != SQLITE_ROW && result != SQLITE_DONE) {
        throw SqliteError(db, result);
    }
}

void exec(sqlite3* db, const char* sql) { check(db, sqlite3_exec(db, sql, nullptr, nullptr, nullptr)); }

// A prepared statement, finalized when it goes.
class Statement {
public:
    Statement(sqlite3* db, const char* sql) : db_(db) { check(db, sqlite3_prepare_v2(db, sql, -1, &statement_, nullptr)); }
    ~Statement() { sqlite3_finalize(statement_); }
    Statement(const Statement&) = delete;
    Statement& operator=(const Statement&) = delete;

    Statement& bind(int index, const std::string& text) {
        check(db_, sqlite3_bind_text(statement_, index, text.data(), static_cast<int>(text.size()), SQLITE_TRANSIENT));
        return *this;
    }
    Statement& bind(int index, std::int64_t number) {
        check(db_, sqlite3_bind_int64(statement_, index, number));
        return *this;
    }
    // Whether a row came.
    bool step() {
        const int result = sqlite3_step(statement_);
        check(db_, result);
        return result == SQLITE_ROW;
    }
    void run() {
        step();
        sqlite3_reset(statement_);
        sqlite3_clear_bindings(statement_);
    }
    std::string text(int column) const {
        const auto* bytes = reinterpret_cast<const char*>(sqlite3_column_text(statement_, column));
        return bytes ? std::string(bytes, static_cast<std::size_t>(sqlite3_column_bytes(statement_, column)))
                     : std::string();
    }
    std::int64_t number(int column) const { return sqlite3_column_int64(statement_, column); }

private:
    sqlite3* db_;
    sqlite3_stmt* statement_ = nullptr;
};

// A transaction that writes (taking the write lock at once, so that two
// processes never both read and then both wait to write), rolled back
// unless committed.
class Transaction {
public:
    explicit Transaction(sqlite3* db) : db_(db) { exec(db, "BEGIN IMMEDIATE"); }
    ~Transaction() {
        if (!done_) {
            sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
        }
    }
    Transaction(const Transaction&) = delete;
    Transaction& operator=(const Transaction&) = delete;
    void commit() {
        exec(db_, "COMMIT");
        done_ = true;
    }

private:
    sqlite3* db_;
    bool done_ = false;
};

std::int64_t us_of(std::chrono::system_clock::time_point time) {
    return std::chrono::duration_cast<std::chrono::microseconds>(time.time_since_epoch()).count();
}

std::chrono::system_clock::time_point time_of(std::int64_t us) {
    return std::chrono::system_clock::time_point(
        std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::microseconds(us)));
}

// A file's time on the system clock: file_time_type's clock is not the
// system's everywhere, so its age is carried over (as saved_results.cpp).
std::chrono::system_clock::time_point system_time(std::filesystem::file_time_type time) {
    const auto age = std::filesystem::file_time_type::clock::now() - time;
    return std::chrono::system_clock::now() - std::chrono::duration_cast<std::chrono::system_clock::duration>(age);
}

std::string text_at(const nlohmann::json& data, const char* side, const char* field) {
    const nlohmann::json value = data.value(side, nlohmann::json::object()).value(field, nlohmann::json());
    return value.is_string() ? value.get<std::string>() : std::string();
}

// A comparison's folder as read for the index.
struct ReadFolder {
    IndexedComparison comparison;
    std::string recipe;
    std::string identity;
};

// The comparison in `folder`; nothing where it has no context.json (not a
// comparison, or gone).
std::optional<ReadFolder> read_folder(const std::filesystem::path& folder) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(folder / "context.json", error)) {
        return std::nullopt;
    }
    ReadFolder read;
    IndexedComparison& comparison = read.comparison;
    comparison.name = platform::path_text(folder.filename());
    // As list_saved_comparisons and cache_summary read a folder: its files'
    // sizes and times from the directory's entries, its scores by name.
    std::filesystem::directory_iterator file(folder, error);
    for (; !error && file != std::filesystem::directory_iterator(); file.increment(error)) {
        std::error_code unread;
        if (!file->is_regular_file(unread)) {
            continue;
        }
        const std::uintmax_t bytes = file->file_size(unread);
        comparison.bytes += unread ? 0 : bytes;
        unread.clear();
        const std::filesystem::file_time_type written = file->last_write_time(unread);
        const auto when = unread ? std::chrono::system_clock::time_point() : system_time(written);
        comparison.saved = std::max(comparison.saved, when);
        const std::string name = platform::path_text(file->path().filename());
        const std::size_t hash = name.rfind('_');
        if (name.ends_with(".npz") && hash != std::string::npos && hash > 0) {
            comparison.scores.push_back({name, name.substr(0, hash), when, bytes});
        }
    }
    const std::optional<std::pair<ComparisonResult, std::string>> context = saved_comparison(folder);
    comparison.readable = context.has_value();
    if (context) {
        comparison.label = context->second;
        comparison.source = platform::path_text(context->first.source);
        comparison.distorted = platform::path_text(context->first.distorted);
        if (const std::optional<nlohmann::json> json = cache_parts::read_json(folder / "context.json");
            json && json->is_object()) {
            read.recipe = json->value("recipe", nlohmann::json()).dump();
        }
    }
    if (const std::optional<nlohmann::json> identity = cache_parts::read_json(folder / "identity.json");
        identity && identity->is_object()) {
        comparison.source_identity = text_at(*identity, "source", "path");
        comparison.source_sampled = text_at(*identity, "source", "sampled");
        read.identity = identity->dump();
    }
    return read;
}

// An index's database, open for one use: each listing and each change
// opens it and closes it after (0.2 ms on a RAM drive; 4 ms with a write on
// an SSD, the log written back on closing), so that no file of it stays
// open while nothing uses it (a cache folder deleted or moved, a test's
// folder).
struct Index {
    sqlite3* db = nullptr;
    Index() = default;
    Index(const Index&) = delete;
    Index& operator=(const Index&) = delete;
    ~Index() { sqlite3_close_v2(db); }
};

void remove_files(const std::filesystem::path& file) {
    for (const char* end : {"", "-wal", "-shm", "-journal"}) {
        std::error_code ignored;
        std::filesystem::remove(platform::path_from_text(platform::path_text(file) + end), ignored);
    }
}

// Opens the index's database and makes its tables where it has none (or of
// another form). Throws SqliteError.
sqlite3* open_database(const std::filesystem::path& base) {
    const std::filesystem::path path = saved_index::file(base);
    sqlite3* db = nullptr;
    const int opened = sqlite3_open_v2(platform::path_text(path).c_str(), &db,
                                       SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX, nullptr);
    std::unique_ptr<sqlite3, int (*)(sqlite3*)> owned(db, sqlite3_close_v2);
    check(db, opened);
    sqlite3_busy_timeout(db, busy_ms);
    // Closed without a checkpoint: each use opens and closes it (an open
    // handle would keep the folder from being deleted), and a checkpoint at
    // each close waits for the disk twice. On a VM runner's disk that was
    // 30 ms a close, two per save: a save took 77 ms, not 23, and deleting
    // 100 comparisons 3.9 s, not 0.5. The log is left beside the file and
    // written back as it grows (SQLite's automatic checkpoint, every 1,000
    // pages); a commit in it waits for no disk (synchronous=NORMAL).
    sqlite3_db_config(db, SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, 1, nullptr);
    // The write-ahead log: readers never wait for a writer, and a crash
    // loses no committed write. It needs memory shared on one computer:
    // on a network drive, a rollback journal.
    exec(db, platform::on_network_drive(path) ? "PRAGMA journal_mode=DELETE" : "PRAGMA journal_mode=WAL");
    exec(db, "PRAGMA synchronous=NORMAL");
    exec(db, "PRAGMA foreign_keys=ON");
    const auto form = [db] {
        Statement version(db, "PRAGMA user_version");
        return version.step() ? version.number(0) : 0;
    };
    if (form() != index_form) {
        Transaction transaction(db);
        const std::int64_t now = form();  // another process may have made it meanwhile
        if (now != index_form) {
            if (now > index_form) {
                throw std::runtime_error("the index is of a newer version of the app");
            }
            exec(db, "DROP TABLE IF EXISTS score; DROP TABLE IF EXISTS record; DROP TABLE IF EXISTS comparison; "
                     "DROP TABLE IF EXISTS pending;");
            exec(db, schema);
            exec(db, ("PRAGMA user_version=" + std::to_string(index_form)).c_str());
        }
        transaction.commit();
    }
    return owned.release();
}

// The index of `base` opened; with `create`, made where there is none.
// Nothing where it cannot be used (said in the log).
std::unique_ptr<Index> index_of(const std::filesystem::path& base, bool create) {
    std::error_code error;
    if (!create && !std::filesystem::exists(saved_index::file(base), error)) {
        return nullptr;
    }
    auto index = std::make_unique<Index>();
    for (int attempt = 0; attempt < 2 && !index->db; ++attempt) {
        try {
            index->db = open_database(base);
        } catch (const SqliteError& failed) {
            // A damaged file: it holds nothing the folders do not, so it is
            // made again (once).
            if (attempt == 0 && (failed.code == SQLITE_CORRUPT || failed.code == SQLITE_NOTADB)) {
                app_log::write(app_log::Level::warning, "vml.saved_index",
                               std::string("The saved results' index was damaged and is made again: ") + failed.what());
                remove_files(saved_index::file(base));
                continue;
            }
            app_log::write(app_log::Level::warning, "vml.saved_index",
                           std::string("The saved results' index cannot be used; the folders are read: ") +
                               failed.what());
            return nullptr;
        } catch (const std::exception& failed) {
            app_log::write(app_log::Level::warning, "vml.saved_index",
                           std::string("The saved results' index cannot be used; the folders are read: ") +
                               failed.what());
            return nullptr;
        }
    }
    return index->db ? std::move(index) : nullptr;
}

void write_folder(sqlite3* db, const std::string& name, const std::optional<ReadFolder>& read) {
    Statement remove(db, "DELETE FROM comparison WHERE name = ?1");
    remove.bind(1, name).run();
    Statement unmark(db, "DELETE FROM pending WHERE name = ?1");
    unmark.bind(1, name).run();
    if (!read) {
        return;
    }
    const IndexedComparison& comparison = read->comparison;
    Statement add(db,
                  "INSERT INTO comparison(name, readable, label, source, distorted, source_identity, source_sampled, "
                  "saved, bytes) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)");
    add.bind(1, name)
        .bind(2, std::int64_t{comparison.readable})
        .bind(3, comparison.label)
        .bind(4, comparison.source)
        .bind(5, comparison.distorted)
        .bind(6, comparison.source_identity)
        .bind(7, comparison.source_sampled)
        .bind(8, us_of(comparison.saved))
        .bind(9, static_cast<std::int64_t>(comparison.bytes))
        .run();
    // What its context.json and identity.json record of its recipe and
    // files, kept apart: a listing reads none of it.
    const std::int64_t id = sqlite3_last_insert_rowid(db);
    Statement record(db, "INSERT INTO record(comparison, recipe, identity) VALUES (?1, ?2, ?3)");
    record.bind(1, id).bind(2, read->recipe).bind(3, read->identity).run();
    Statement score(db, "INSERT INTO score(comparison, file, key, saved, bytes) VALUES (?1, ?2, ?3, ?4, ?5)");
    for (const IndexedComparison::Score& file : comparison.scores) {
        score.bind(1, id)
            .bind(2, file.file)
            .bind(3, file.key)
            .bind(4, us_of(file.saved))
            .bind(5, static_cast<std::int64_t>(file.bytes))
            .run();
    }
}

// Brings the index in line with the folders under `base`: reads each
// comparison new or marked pending, and drops those gone. One listing of
// the folder of comparisons, of their names only: what changed inside a
// comparison's folder, this app and vml-cli write to the index as they
// change it (saved_index::changed). A folder's time cannot tell: on NTFS,
// a folder's listing gives its subfolders' times as they were when the
// subfolders were made, and asking each subfolder its own took 210 ms for
// 20,000. Whether it got to the end (not stopped).
bool bring_up_to_date(Index& index, const std::filesystem::path& base, const std::function<bool()>& stopped) {
    sqlite3* db = index.db;
    const std::filesystem::path root = base / cache_parts::format_folder;
    std::set<std::string> folders;
    std::error_code error;
    std::filesystem::directory_iterator entry(root, error);
    const bool no_folder = error.value() != 0;
    for (; !error && entry != std::filesystem::directory_iterator(); entry.increment(error)) {
        std::error_code unread;
        if (!entry->is_directory(unread)) {
            continue;
        }
        std::string name = platform::path_text(entry->path().filename());
        if (!name.starts_with('.')) {  // .deleted: what delete_saved_comparison moved aside
            folders.insert(std::move(name));
        }
    }
    if (error && !no_folder) {
        return true;  // a listing cut short (a share gone): the index as it is
    }
    std::set<std::string> indexed;
    {
        Statement rows(db, "SELECT name FROM comparison");
        while (rows.step()) {
            indexed.insert(rows.text(0));
        }
    }
    std::set<std::string> pending;
    {
        Statement rows(db, "SELECT name FROM pending");
        while (rows.step()) {
            pending.insert(rows.text(0));
        }
    }
    std::vector<std::string> gone;
    for (const std::string& name : indexed) {
        if (!folders.contains(name)) {
            gone.push_back(name);
        }
    }
    for (const std::string& name : pending) {
        if (!folders.contains(name) && !indexed.contains(name)) {
            gone.push_back(name);  // marked, never written: only the mark to drop
        }
    }
    std::vector<std::string> to_read;
    for (const std::string& name : folders) {
        if (!indexed.contains(name) || pending.contains(name)) {
            to_read.push_back(name);
        }
    }
    if (!gone.empty()) {
        Transaction transaction(db);
        for (const std::string& name : gone) {
            write_folder(db, name, std::nullopt);
        }
        transaction.commit();
    }
    std::vector<std::pair<std::string, std::optional<ReadFolder>>> read;
    const auto write_read = [db, &read] {
        if (read.empty()) {
            return;
        }
        Transaction transaction(db);
        for (const auto& [name, folder] : read) {
            write_folder(db, name, folder);
        }
        transaction.commit();
        read.clear();
    };
    for (const std::string& name : to_read) {
        if (stopped && stopped()) {
            write_read();
            return false;
        }
        read.emplace_back(name, read_folder(root / platform::path_from_text(name)));
        if (read.size() >= batch) {
            write_read();
        }
    }
    write_read();
    if (to_read.size() > batch) {
        app_log::write(app_log::Level::info, "vml.saved_index",
                       "Read " + std::to_string(to_read.size()) + " saved comparisons into the index");
    }
    return true;
}

std::vector<IndexedComparison> all_rows(sqlite3* db) {
    std::vector<IndexedComparison> list;
    std::unordered_map<std::int64_t, std::size_t> place;
    {
        Statement rows(db,
                       "SELECT name, readable, label, source, distorted, source_identity, source_sampled, saved, "
                       "bytes, id FROM comparison ORDER BY saved DESC, name");
        while (rows.step()) {
            IndexedComparison comparison;
            comparison.name = rows.text(0);
            comparison.readable = rows.number(1) != 0;
            comparison.label = rows.text(2);
            comparison.source = rows.text(3);
            comparison.distorted = rows.text(4);
            comparison.source_identity = rows.text(5);
            comparison.source_sampled = rows.text(6);
            comparison.saved = time_of(rows.number(7));
            comparison.bytes = static_cast<std::uintmax_t>(rows.number(8));
            place.emplace(rows.number(9), list.size());
            list.push_back(std::move(comparison));
        }
    }
    Statement rows(db, "SELECT comparison, file, key, saved, bytes FROM score");
    while (rows.step()) {
        if (const auto found = place.find(rows.number(0)); found != place.end()) {
            list[found->second].scores.push_back(
                {rows.text(1), rows.text(2), time_of(rows.number(3)), static_cast<std::uintmax_t>(rows.number(4))});
        }
    }
    return list;
}

void warn(const std::exception& failed) {
    app_log::write(app_log::Level::warning, "vml.saved_index",
                   std::string("The saved results' index could not be read or written: ") + failed.what());
}

}  // namespace

namespace saved_index {

std::filesystem::path file(const std::filesystem::path& base) { return base / "saved_results.sqlite"; }

std::optional<std::vector<IndexedComparison>> comparisons(const std::filesystem::path& base,
                                                          const std::function<bool()>& stopped) {
    std::error_code error;
    if (!std::filesystem::is_directory(base / cache_parts::format_folder, error)) {
        // Nothing saved (or all of it deleted): an index there is emptied,
        // none is made.
        if (const std::unique_ptr<Index> index = index_of(base, false)) {
            try {
                Transaction transaction(index->db);
                exec(index->db, "DELETE FROM comparison; DELETE FROM pending;");
                transaction.commit();
            } catch (const std::exception& failed) {
                warn(failed);
            }
        }
        return std::vector<IndexedComparison>{};
    }
    const std::unique_ptr<Index> index = index_of(base, true);
    if (!index) {
        return std::nullopt;
    }
    try {
        if (!bring_up_to_date(*index, base, stopped)) {
            return std::nullopt;
        }
        return all_rows(index->db);
    } catch (const std::exception& failed) {
        warn(failed);
        return std::nullopt;
    }
}

void changing(const std::filesystem::path& folder) {
    const std::unique_ptr<Index> index = index_of(folder.parent_path().parent_path(), false);
    if (!index) {
        return;  // none yet: the first listing reads every folder
    }
    try {
        Statement mark(index->db, "INSERT OR IGNORE INTO pending(name) VALUES (?1)");
        mark.bind(1, platform::path_text(folder.filename())).run();
    } catch (const std::exception& failed) {
        warn(failed);
    }
}

void changed(const std::filesystem::path& folder) { changed(std::vector<std::filesystem::path>{folder}); }

void changed(const std::vector<std::filesystem::path>& folders) {
    if (folders.empty()) {
        return;
    }
    const std::unique_ptr<Index> index = index_of(folders.front().parent_path().parent_path(), false);
    if (!index) {
        return;
    }
    std::vector<std::optional<ReadFolder>> read;
    read.reserve(folders.size());
    for (const std::filesystem::path& folder : folders) {
        read.push_back(read_folder(folder));
    }
    try {
        Transaction transaction(index->db);
        for (std::size_t i = 0; i < folders.size(); ++i) {
            write_folder(index->db, platform::path_text(folders[i].filename()), read[i]);
        }
        transaction.commit();
    } catch (const std::exception& failed) {
        warn(failed);
    }
}

std::optional<int> rebuild(const std::filesystem::path& base, const std::function<bool()>& stopped) {
    if (const std::unique_ptr<Index> index = index_of(base, false)) {
        try {
            Transaction transaction(index->db);
            exec(index->db, "DELETE FROM comparison; DELETE FROM pending;");
            transaction.commit();
        } catch (const std::exception& failed) {
            warn(failed);
            return std::nullopt;
        }
    }
    const std::optional<std::vector<IndexedComparison>> all = comparisons(base, stopped);
    if (!all) {
        return std::nullopt;
    }
    return static_cast<int>(all->size());
}
}  // namespace saved_index

}  // namespace vml
