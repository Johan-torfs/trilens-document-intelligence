from hashlib import sha256
from pathlib import Path


def calculate_checksum(file_path: Path) -> str:
    digest = sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()


def calculate_checksum_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()