from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.set_page_config(
    page_title="TriLens Document Intelligence",
    page_icon="🔎",
    layout="wide",
)

pages = [
    st.Page(
        "app/ui/pages/upload.py",
        title="Upload",
    ),
    st.Page(
        "app/ui/pages/search.py",
        title="Search",
    ),
    st.Page(
        "app/ui/pages/analysis.py",
        title="Analysis",
    ),
]

navigation = st.navigation(pages)
navigation.run()