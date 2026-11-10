//! Cross-process file lock compatible with the `proper-lockfile` npm package, which pi uses for its config files.
//!
//! The lock for `file` is the directory `<realpath(file)>.lock`, created atomically. The holder keeps the
//! directory's mtime fresh; a lock whose mtime is older than the stale threshold belongs to a dead process and may
//! be removed by anyone. Releasing removes the directory.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime};

const STALE: Duration = Duration::from_secs(30);
const HEARTBEAT: Duration = Duration::from_secs(5);
const MAX_RETRY_DELAY: Duration = Duration::from_secs(1);

pub struct FileLock {
    directory: PathBuf,
    released: Arc<AtomicBool>,
}

impl FileLock {
    /// Waits up to the stale threshold for the lock on `file`, which must exist.
    pub async fn acquire(file: &Path) -> io::Result<FileLock> {
        let directory = lock_directory(file)?;
        let deadline = tokio::time::Instant::now() + STALE;
        let mut delay = Duration::from_millis(10);
        loop {
            match fs::create_dir(&directory) {
                Ok(()) => break,
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                    if is_stale(&directory) {
                        // The holder died. Whoever removes the directory still has to win the next create.
                        let _ = fs::remove_dir(&directory);
                        continue;
                    }
                    if tokio::time::Instant::now() >= deadline {
                        return Err(io::Error::new(
                            io::ErrorKind::TimedOut,
                            format!("{} is locked by another process", file.display()),
                        ));
                    }
                    tokio::time::sleep(delay).await;
                    delay = (delay * 2).min(MAX_RETRY_DELAY);
                }
                Err(error) => return Err(error),
            }
        }
        let released = Arc::new(AtomicBool::new(false));
        tokio::spawn(heartbeat(directory.clone(), released.clone()));
        Ok(FileLock { directory, released })
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        self.released.store(true, Ordering::SeqCst);
        let _ = fs::remove_dir(&self.directory);
    }
}

fn lock_directory(file: &Path) -> io::Result<PathBuf> {
    let mut path = fs::canonicalize(file)?.into_os_string();
    path.push(".lock");
    Ok(PathBuf::from(path))
}

fn is_stale(directory: &Path) -> bool {
    fs::metadata(directory)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| SystemTime::now().duration_since(modified).ok())
        .is_some_and(|age| age > STALE)
}

async fn heartbeat(directory: PathBuf, released: Arc<AtomicBool>) {
    loop {
        tokio::time::sleep(HEARTBEAT).await;
        if released.load(Ordering::SeqCst) {
            return;
        }
        let touched = fs::File::open(&directory).and_then(|handle| handle.set_modified(SystemTime::now()));
        if touched.is_err() {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn holds_the_lock_directory_until_dropped() {
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("auth.json");
        fs::write(&file, "{}").unwrap();
        let lock_directory = lock_directory(&file).unwrap();

        let lock = FileLock::acquire(&file).await.unwrap();
        assert!(lock_directory.is_dir());
        drop(lock);
        assert!(!lock_directory.exists());
    }

    #[tokio::test]
    async fn takes_over_a_stale_lock() {
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("auth.json");
        fs::write(&file, "{}").unwrap();
        let lock_directory = lock_directory(&file).unwrap();
        fs::create_dir(&lock_directory).unwrap();
        let old = SystemTime::now() - STALE - Duration::from_secs(1);
        fs::File::open(&lock_directory).unwrap().set_modified(old).unwrap();

        let _lock = FileLock::acquire(&file).await.unwrap();
        assert!(!is_stale(&lock_directory));
    }
}
