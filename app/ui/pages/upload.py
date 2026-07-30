from pathlib import Path

import streamlit as st

from app.ui.dependencies import (
    PROJECT_ROOT,
    get_pipeline,
)
from app.ui.upload_helpers import (
    create_document_from_upload,
)


UPLOAD_DIR = PROJECT_ROOT / "data" / "runtime" / "uploads"

DOCUMENT_TYPES = {
    "Invoice": "invoice",
    "Purchase order": "purchase_order",
    "Receipt": "receipt",
    "Delivery note": "delivery_note",
    "Application form": "application_form",
    "Identity card": "identity_card",
}


st.title("Document upload")

st.write(
    "Upload een document om preprocessing, "
    "CLIP-indexering en BLIP-captioning uit te voeren."
)

with st.form("document-upload-form"):
    uploaded_file = st.file_uploader(
        "Document",
        type=["png", "jpg", "jpeg"],
    )

    selected_type_label = st.selectbox(
        "Documenttype",
        options=list(DOCUMENT_TYPES),
    )

    submitted = st.form_submit_button(
        "Document verwerken",
        type="primary",
    )


if submitted:
    if uploaded_file is None:
        st.error("Selecteer eerst een document.")
        st.stop()

    try:
        document, image_path = create_document_from_upload(
            uploaded_file=uploaded_file,
            document_type=DOCUMENT_TYPES[
                selected_type_label
            ],
            upload_dir=UPLOAD_DIR,
        )

        pipeline = get_pipeline()

        with st.spinner(
            "Document preprocessen, indexeren en captionen..."
        ):
            outcome = pipeline.index_document(
                document=document,
                image_path=Path(image_path),
            )

    except Exception as error:
        st.error(f"Documentverwerking mislukt: {error}")
        st.stop()

    st.success("Documentverwerking voltooid.")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Zoekbaar",
        "Ja" if outcome.is_searchable else "Nee",
    )

    col2.metric(
        "Caption",
        "Ja" if outcome.has_caption else "Nee",
    )

    col3.metric(
        "Runtime",
        f"{outcome.duration_ms:.0f} ms",
    )

    if outcome.reused_document:
        st.info(
            "Dit document was al bekend. Bestaande "
            "modelartifacts zijn waar mogelijk hergebruikt."
        )

    st.image(
        str(image_path),
        caption=outcome.document.original_filename,
        width=500,
    )

    st.subheader("Verwerkingsresultaat")

    st.write(
        {
            "document_id": outcome.document.id,
            "document_type": outcome.document.document_type,
            "checksum": outcome.document.checksum,
            "clip_model": (
                outcome.embedding_artifact.model_name
                if outcome.embedding_artifact
                else None
            ),
            "caption_model": (
                outcome.caption_artifact.model_name
                if outcome.caption_artifact
                else None
            ),
            "caption": (
                outcome.caption_artifact.content
                if outcome.caption_artifact
                else None
            ),
        }
    )

    if outcome.embedding_error:
        st.error(
            f"CLIP-indexering mislukte: "
            f"{outcome.embedding_error}"
        )

    if outcome.caption_error:
        st.warning(
            "Het document is geïndexeerd, maar captioning "
            f"mislukte: {outcome.caption_error}"
        )