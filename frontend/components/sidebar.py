import streamlit as st
import httpx
import os

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


def render_sidebar() -> dict:
    with st.sidebar:
        st.title("Financial Copilot")
        st.caption("Query financial documents with cited answers.")
        st.divider()

        st.subheader("Upload Document")
        uploaded_file = st.file_uploader(
            "PDF, CSV, or Markdown",
            type=["pdf", "csv", "txt", "md"],
        )

        if uploaded_file and st.button("Ingest"):
            with st.spinner("Uploading..."):
                response = httpx.post(
                    f"{API_BASE}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.read(), "application/pdf")},
                    timeout=120,
                )
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Uploaded: {data['filename']}")
                    st.session_state["document_id"] = data["document_id"]
                else:
                    st.error(f"Upload failed: {response.text}")

        st.divider()

        query_mode = st.radio(
            "Query Mode",
            options=["Standard RAG", "Multi-Agent"],
            help="Multi-Agent runs a planner → parallel research → analysis → summary pipeline.",
        )

        st.divider()

        try:
            health = httpx.get(f"{API_BASE}/health", timeout=3).json()
            st.caption(f"API: {health['status']} · {health['llm_provider']}")
        except Exception:
            st.caption("API: unreachable")

    return {"query_mode": query_mode}
