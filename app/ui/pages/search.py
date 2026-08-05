from pathlib import Path

import streamlit as st

from app.domain.search import SearchQuery
from app.ui.dependencies import get_pipeline


DOCUMENT_TYPES = {
    "Alle types": None,
    "Invoice": "invoice",
    "Purchase order": "purchase_order",
    "Receipt": "receipt",
    "Delivery note": "delivery_note",
    "Application form": "application_form",
    "Identity card": "identity_card",
}


st.title("Document search")

st.write(
    "Zoek documenten met hybride ranking."
)

with st.form("document-search-form"):
    query_text = st.text_input(
        "Zoekopdracht",
        placeholder="Bijvoorbeeld: invoice with several product rows",
    )

    col1, col2 = st.columns(2)

    selected_type_label = col1.selectbox(
        "Documenttype",
        options=list(DOCUMENT_TYPES),
    )

    top_k = col2.number_input(
        "Aantal resultaten",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    submitted = st.form_submit_button(
        "Zoeken",
        type="primary",
    )


if submitted:
    if not query_text.strip():
        st.error("Voer eerst een zoekopdracht in.")
        st.stop()

    query = SearchQuery(
        text=query_text.strip(),
        top_k=int(top_k),
        document_type=DOCUMENT_TYPES[
            selected_type_label
        ],
    )

    try:
        pipeline = get_pipeline()

        with st.spinner("Documenten doorzoeken..."):
            outcome = pipeline.search(
                query=query,
            )

        st.session_state["search_outcome"] = outcome

    except Exception as error:
        st.error(f"Zoeken mislukt: {error}")
        st.stop()


outcome = st.session_state.get("search_outcome")

if outcome is not None:
    st.subheader("Zoekresultaten")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Resultaten",
        len(outcome.results),
    )

    col2.metric(
        "Ranking",
        outcome.ranking_mode.upper(),
    )

    col3.metric(
        "Runtime",
        f"{outcome.duration_ms:.0f} ms",
    )

    if not outcome.results:
        st.info("Geen overeenkomende documenten gevonden.")

    for result in outcome.results:
        with st.container(border=True):
            image_column, details_column = st.columns(
                [1, 2]
            )

            image_path = Path(result.stored_path)

            if image_path.exists():
                image_column.image(
                    str(image_path),
                    caption=result.document_type,
                    width="stretch"
                )
            else:
                image_column.warning(
                    "Afbeeldingsbestand niet gevonden."
                )

            details_column.subheader(
                f"#{result.rank} — {result.document_type}"
            )

            score_col1, score_col2, score_col3 = (
                details_column.columns(3)
            )

            score_col1.metric(
                "Final score",
                f"{result.final_score:.3f}",
            )

            score_col2.metric(
                "Visual score",
                f"{result.visual_score:.3f}",
            )

            score_col3, = (details_column.columns(1),)

            score_col3.metric(
                "Text score",
                f"{result.text_score:.3f}",
            )

            details_column.caption(
                f"Document ID: {result.document_id}"
            )

            if details_column.button(
                "Selecteer voor analyse",
                key=f"analyze-{result.document_id}",
            ):
                st.session_state[
                    "selected_document_id"
                ] = result.document_id

                st.success(
                    "Document geselecteerd. "
                    "Open nu de Analysis-pagina."
                )