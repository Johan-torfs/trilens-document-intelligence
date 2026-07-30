from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image
from streamlit.runtime.uploaded_file_manager import UploadedFile

from app.domain.document import (
    DocumentMetadata,
    DocumentRecord,
)


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def create_document_from_upload(
    uploaded_file: UploadedFile,
    document_type: str,
    upload_dir: Path,
) -> tuple[DocumentRecord, Path]:
    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        raise ValueError("Het geüploade bestand is leeg.")

    extension = Path(uploaded_file.name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Alleen PNG-, JPG- en JPEG-bestanden worden ondersteund."
        )

    checksum = sha256(file_bytes).hexdigest()

    try:
        with Image.open(BytesIO(file_bytes)) as image:
            width, height = image.size
    except Exception as error:
        raise ValueError(
            "Het geüploade bestand is geen geldige afbeelding."
        ) from error

    upload_dir.mkdir(parents=True, exist_ok=True)

    # De checksum als bestandsnaam voorkomt dubbele bestanden
    # wanneer exact dezelfde afbeelding opnieuw wordt geüpload.
    stored_path = upload_dir / f"{checksum}{extension}"

    if not stored_path.exists():
        stored_path.write_bytes(file_bytes)

    document = DocumentRecord(
        id=str(uuid4()),
        original_filename=uploaded_file.name,
        stored_path=str(stored_path),
        checksum=checksum,
        width=width,
        height=height,
        mime_type=uploaded_file.type or "application/octet-stream",
        document_type=document_type,
        metadata=DocumentMetadata(
            document_type=document_type,
        ),
    )

    return document, stored_path