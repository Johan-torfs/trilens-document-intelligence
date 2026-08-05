import streamlit as st

from app.domain.prepared_document import DocumentSource
from app.ui.dependencies import (
    get_pipeline,
)


DOCUMENT_TYPES: dict[str, str | None] = {
    "Auto-detect": None,
    "Invoice": "invoice",
    "Purchase order": "purchase_order",
    "Receipt": "receipt",
    "Delivery note": "delivery_note",
    "Application form": "application_form",
    "Identity card": "identity_card",
    "Contract": "contract",
    "Letter": "letter",
    "Report": "report",
    "Bank statement": "bank_statement",
    "Pay slip": "pay_slip",
    "Quotation": "quotation",
    "Certificate": "certificate",
    "Tax document": "tax_document",
}


st.title("Document upload")

st.write(
    "Upload een document om preprocessing "
    "en SigLIP-indexering uit te voeren."
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
        source = DocumentSource(
            filename=uploaded_file.name,
            mime_type=uploaded_file.type
            or "application/octet-stream",
            content=uploaded_file.getvalue(),
        )

        pipeline = get_pipeline()

        with st.spinner(
            "Document preprocessen en indexeren..."
        ):
            outcome = pipeline.index_document(
                source=source,
                document_type=DOCUMENT_TYPES[selected_type_label],
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
        "OCR",
        "Ja" if outcome.has_ocr else "Nee",
    )

    col3.metric(
        "Runtime",
        f"{outcome.duration_ms:.0f} ms",
    )

    if outcome.classification_confidence is not None:
        st.info(
            f"Automatisch gedetecteerd type: **{outcome.document.document_type}** "
            f"(zekerheid: {outcome.classification_confidence:.0%})"
        )

    if outcome.reused_document:
        st.info(
            "Dit document was al bekend. "
            "Bestaande vectorindices zijn hergebruikt."
        )

    image_path = outcome.document.stored_path

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
            "indexing_model": (
                outcome.indexing_result.model_name
                if outcome.indexing_result
                else None
            ),
            "page_count": (
                outcome.indexing_result.page_count
                if outcome.indexing_result
                else None
            ),
            "has_ocr": outcome.has_ocr,
        }
    )

    if outcome.indexing_error:
        st.error(
            f"Indexering mislukte: "
            f"{outcome.indexing_error}"
        )

    if outcome.ocr_error:
        st.warning(
            "Indexering geslaagd, maar OCR mislukte: "
            f"{outcome.ocr_error}"
        )