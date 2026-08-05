from pathlib import Path

import streamlit as st

from app.services.analysis_service import (
    AnalysisDisabledError,
)
from app.ui.dependencies import get_pipeline


EXAMPLE_QUESTIONS = [
    "Beschrijf de structuur van dit document.",
    "Is er een handtekening zichtbaar?",
    "Welke visuele elementen wijzen op een factuur?",
    "Lijkt dit document meerdere productregels te bevatten?",
]


st.title("Document analysis")

st.warning(
    "Deze analyse is modelgegenereerd en kan onjuist zijn. "
    "Gebruik het resultaat niet als gegarandeerd feit."
)

selected_document_id = st.session_state.get(
    "selected_document_id",
    "",
)

if selected_document_id:
    st.success(
        f"Geselecteerd document: {selected_document_id}"
    )
else:
    st.info(
        "Selecteer eerst een document via de Search-pagina "
        "of voer hieronder handmatig een document-ID in."
    )


with st.form("document-analysis-form"):
    document_id = st.text_input(
        "Document-ID",
        value=selected_document_id,
    )

    example_question = st.selectbox(
        "Voorbeeldvraag",
        options=EXAMPLE_QUESTIONS,
    )

    question = st.text_area(
        "Analysevraag",
        value=example_question,
        height=100,
    )

    submitted = st.form_submit_button(
        "Analyse uitvoeren",
        type="primary",
    )


if submitted:
    if not document_id.strip():
        st.error("Voer eerst een document-ID in.")
        st.stop()

    if not question.strip():
        st.error("Voer eerst een analysevraag in.")
        st.stop()

    try:
        pipeline = get_pipeline()

        with st.spinner(
            "Document analyseren..."
        ):
            outcome = pipeline.analyze_document(
                document_id=document_id,
                question=question,
            )

        st.session_state["analysis_outcome"] = outcome

    except AnalysisDisabledError:
        st.error(
            "OpenFlamingo-analyse is uitgeschakeld. "
            "Herstart Streamlit met "
            "`TRILENS_OPEN_FLAMINGO_ENABLED=true`."
        )
        st.stop()

    except Exception as error:
        st.error(f"Analyse mislukt: {error}")
        st.stop()


outcome = st.session_state.get("analysis_outcome")

if outcome is not None:
    st.subheader("Modelgegenereerde analyse")

    image_path = Path(outcome.document.stored_path)

    image_column, result_column = st.columns(
        [1, 2]
    )

    if image_path.exists():
        image_column.image(
            str(image_path),
            caption=outcome.document.original_filename,
            width="stretch"
        )
    else:
        image_column.warning(
            "Het opgeslagen afbeeldingsbestand "
            "werd niet gevonden."
        )

    result_column.markdown(
        f"**Vraag:** {outcome.question}"
    )

    result_column.write(
        outcome.analysis.text
    )

    result_column.info(
        "Experimentele OpenFlamingo-output. "
        "Verifieer het resultaat altijd."
    )

    metric_col1, metric_col2 = st.columns(2)

    metric_col1.metric(
        "Modelruntime",
        f"{outcome.analysis.duration_ms:.0f} ms",
    )

    metric_col2.metric(
        "Totale runtime",
        f"{outcome.duration_ms:.0f} ms",
    )

    st.caption(
        " | ".join(
            [
                f"Document: {outcome.document.id}",
                f"Model: {outcome.analysis.model_name}",
                (
                    "Versie: "
                    f"{outcome.analysis.model_version or 'standaard'}"
                ),
            ]
        )
    )