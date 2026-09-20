"""Content fingerprints prevent accidentally mixing different source versions."""

import hashlib
from pathlib import Path


def find_input_csv(directory: Path) -> Path:
    """Ignore Excel owner/lock files even when they have a .csv extension."""
    candidates = sorted(path for path in Path(directory).glob("*.csv")
                        if not path.name.startswith(("~$", ".")) and path.is_file())
    if len(candidates) != 1:
        raise ValueError("원본 CSV가 정확히 하나여야 합니다(Excel 임시 파일 제외). 입력 경로를 명시하세요.")
    return candidates[0]


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"name": Path(path).name, "bytes": Path(path).stat().st_size, "sha256": digest.hexdigest()}
