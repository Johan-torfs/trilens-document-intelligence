from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image

from app.domain.document import (
    DocumentMetadata,
    DocumentRecord,
)


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def create_document_from_bytes(
    filename: str,
    content_type: str | None,
    file_bytes: bytes,
    document_type: str,
    upload_dir: Path,
) -> tuple[DocumentRecord, Path]:
    if not file_bytes:
        raise ValueError("Het geüploade bestand is leeg.")

    cleaned_document_type = document_type.strip()

    if not cleaned_document_type:
        raise ValueError("Het documenttype mag niet leeg zijn.")

    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Alleen PNG-, JPG- en JPEG-bestanden worden ondersteund."
        )

    try:
        with Image.open(BytesIO(file_bytes)) as image:
            image.verify()

        with Image.open(BytesIO(file_bytes)) as image:
            width, height = image.size

    except Exception as error:
        raise ValueError(
            "Het geüploade bestand is geen geldige afbeelding."
        ) from error

    checksum = sha256(file_bytes).hexdigest()

    upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stored_path = upload_dir / f"{checksum}{extension}"

    if not stored_path.exists():
        stored_path.write_bytes(file_bytes)

    document = DocumentRecord(
        id=str(uuid4()),
        original_filename=filename,
        stored_path=str(stored_path),
        checksum=checksum,
        width=width,
        height=height,
        mime_type=content_type or "application/octet-stream",
        document_type=cleaned_document_type,
        metadata=DocumentMetadata(
            document_type=cleaned_document_type,
        ),
    )

    return document, stored_path