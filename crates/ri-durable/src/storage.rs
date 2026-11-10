//! Where a session's events are kept. One process writes a session at a time.

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::event::Event;

const FORMAT: u32 = 1;
const LOG_FILE: &str = "log.jsonl";

/// An append-only log of commits. A commit is stored whole or not at all.
pub trait Storage: Send + 'static {
    /// Every stored commit's events, oldest first.
    fn load(&mut self) -> io::Result<Vec<Event>>;

    /// Stores `events` as one commit. They are durable once this returns.
    fn append(&mut self, events: &[Event]) -> io::Result<()>;
}

/// Keeps nothing across restarts. Clones share one log, which lets tests reopen it.
#[derive(Debug, Clone, Default)]
pub struct MemoryStorage {
    commits: Arc<Mutex<Vec<Event>>>,
}

impl MemoryStorage {
    pub fn new() -> Self {
        MemoryStorage::default()
    }
}

impl Storage for MemoryStorage {
    fn load(&mut self) -> io::Result<Vec<Event>> {
        Ok(self.commits.lock().expect("memory storage lock").clone())
    }

    fn append(&mut self, events: &[Event]) -> io::Result<()> {
        self.commits
            .lock()
            .expect("memory storage lock")
            .extend_from_slice(events);
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct Record {
    format: u32,
    seq: u64,
    events: Vec<Event>,
}

/// One JSON line per commit in `<directory>/log.jsonl`. The whole history is read into memory on open.
#[derive(Debug)]
pub struct JsonlStorage {
    file: File,
    path: PathBuf,
    seq: u64,
    fsync: bool,
}

impl JsonlStorage {
    /// Opens or creates the log in `directory` and takes its writer lock.
    pub fn open(directory: impl AsRef<Path>) -> io::Result<JsonlStorage> {
        fs::create_dir_all(&directory)?;
        let path = directory.as_ref().join(LOG_FILE);
        let file = OpenOptions::new().read(true).append(true).create(true).open(&path)?;
        file.try_lock().map_err(|_| {
            io::Error::new(
                io::ErrorKind::WouldBlock,
                format!("{} is open in another process", path.display()),
            )
        })?;
        Ok(JsonlStorage {
            file,
            path,
            seq: 0,
            fsync: false,
        })
    }

    /// Flushes every commit to the disk before it counts as stored. Without it commits survive a process crash,
    /// but the newest may be lost on power failure.
    pub fn fsync(mut self, fsync: bool) -> Self {
        self.fsync = fsync;
        self
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Storage for JsonlStorage {
    fn load(&mut self) -> io::Result<Vec<Event>> {
        self.file.seek(SeekFrom::Start(0))?;
        let mut reader = BufReader::new(&self.file);
        let mut events = Vec::new();
        let mut valid_length = 0u64;
        let mut line = String::new();
        loop {
            line.clear();
            let read = reader.read_line(&mut line)? as u64;
            if read == 0 {
                break;
            }
            // A commit counts once its line is complete; a crash mid-write leaves a torn last line.
            let record = line
                .ends_with('\n')
                .then(|| serde_json::from_str::<Record>(&line))
                .and_then(Result::ok);
            let Some(record) = record else { break };
            if record.format != FORMAT {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "{} has format {}, this version reads {FORMAT}",
                        self.path.display(),
                        record.format
                    ),
                ));
            }
            self.seq = record.seq;
            events.extend(record.events);
            valid_length += read;
        }
        if valid_length < self.file.metadata()?.len() {
            self.file.set_len(valid_length)?;
        }
        Ok(events)
    }

    fn append(&mut self, events: &[Event]) -> io::Result<()> {
        let record = Record {
            format: FORMAT,
            seq: self.seq + 1,
            events: events.to_vec(),
        };
        let mut line = serde_json::to_vec(&record).map_err(io::Error::other)?;
        line.push(b'\n');
        self.file.write_all(&line)?;
        if self.fsync {
            self.file.sync_data()?;
        }
        self.seq += 1;
        Ok(())
    }
}

/// One row per commit in a SQLite database, in WAL mode with `synchronous = NORMAL`: commits survive process
/// crashes, and other processes can read the database while this one writes.
#[cfg(feature = "sqlite")]
pub struct SqliteStorage {
    connection: rusqlite::Connection,
}

#[cfg(feature = "sqlite")]
impl SqliteStorage {
    pub fn open(path: impl AsRef<Path>) -> io::Result<SqliteStorage> {
        if let Some(parent) = path.as_ref().parent().filter(|parent| !parent.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        let setup = "PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL; PRAGMA locking_mode = EXCLUSIVE;
            CREATE TABLE IF NOT EXISTS ri_commits (seq INTEGER PRIMARY KEY AUTOINCREMENT, events TEXT NOT NULL);";
        let connection = rusqlite::Connection::open(path).map_err(io::Error::other)?;
        connection.execute_batch(setup).map_err(io::Error::other)?;
        Ok(SqliteStorage { connection })
    }
}

#[cfg(feature = "sqlite")]
impl Storage for SqliteStorage {
    fn load(&mut self) -> io::Result<Vec<Event>> {
        let mut statement = self
            .connection
            .prepare("SELECT events FROM ri_commits ORDER BY seq")
            .map_err(io::Error::other)?;
        let rows = statement
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(io::Error::other)?;
        rows.map(|row| serde_json::from_str::<Vec<Event>>(&row.map_err(io::Error::other)?).map_err(io::Error::other))
            .collect::<io::Result<Vec<Vec<Event>>>>()
            .map(|commits| commits.into_iter().flatten().collect())
    }

    fn append(&mut self, events: &[Event]) -> io::Result<()> {
        let events = serde_json::to_string(events).map_err(io::Error::other)?;
        self.connection
            .prepare_cached("INSERT INTO ri_commits (events) VALUES (?1)")
            .and_then(|mut statement| statement.execute([events]))
            .map(|_| ())
            .map_err(io::Error::other)
    }
}

/// The directory for an application's sessions: `$RI_DURABLE_DIR/<app>`, otherwise
/// `$XDG_DATA_HOME/ri-durable/<app>`, otherwise `~/.local/share/ri-durable/<app>`.
pub fn data_dir(app: &str) -> PathBuf {
    let set = |name: &str| {
        std::env::var_os(name)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    };
    let base = set("RI_DURABLE_DIR")
        .or_else(|| set("XDG_DATA_HOME").map(|data| data.join("ri-durable")))
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
                .join(".local")
                .join("share")
                .join("ri-durable")
        });
    base.join(app)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(key: &str) -> Event {
        Event::Value {
            conversation: "c".into(),
            key: key.into(),
            value: Some(serde_json::json!(1)),
        }
    }

    #[test]
    fn reopening_returns_the_commits_and_drops_a_torn_line() {
        let directory = tempfile::tempdir().unwrap();
        {
            let mut storage = JsonlStorage::open(directory.path()).unwrap();
            assert!(storage.load().unwrap().is_empty());
            storage.append(&[value("a"), value("b")]).unwrap();
            storage.append(&[value("c")]).unwrap();
        }
        let path = directory.path().join(LOG_FILE);
        let complete = fs::metadata(&path).unwrap().len();
        OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"{\"format\":1,\"seq\":3,\"eve")
            .unwrap();

        let mut storage = JsonlStorage::open(directory.path()).unwrap();
        assert_eq!(storage.load().unwrap(), vec![value("a"), value("b"), value("c")]);
        assert_eq!(fs::metadata(&path).unwrap().len(), complete);
        storage.append(&[value("d")]).unwrap();
        drop(storage);
        let mut storage = JsonlStorage::open(directory.path()).unwrap();
        assert_eq!(storage.load().unwrap().len(), 4);
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn sqlite_returns_the_commits_after_reopening() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("nested").join("session.sqlite");
        {
            let mut storage = SqliteStorage::open(&path).unwrap();
            storage.append(&[value("a"), value("b")]).unwrap();
            storage.append(&[value("c")]).unwrap();
        }
        let mut storage = SqliteStorage::open(&path).unwrap();
        assert_eq!(storage.load().unwrap(), vec![value("a"), value("b"), value("c")]);
    }

    #[test]
    fn a_second_writer_is_refused() {
        let directory = tempfile::tempdir().unwrap();
        let _first = JsonlStorage::open(directory.path()).unwrap();
        assert_eq!(
            JsonlStorage::open(directory.path()).unwrap_err().kind(),
            io::ErrorKind::WouldBlock
        );
    }
}
