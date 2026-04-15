from pathlib import Path

from app.config import INDEX_CONFIG_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR


def ensure_directories() -> None:
    for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, INDEX_CONFIG_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print("Data directories are ready.")
