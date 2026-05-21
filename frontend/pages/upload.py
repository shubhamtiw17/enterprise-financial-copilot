import streamlit as st
import httpx
import time
import os

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


def render_upload() -> None:
    st.title("Document Upload")
    st.caption("Upload financial documents to make them queryable.")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "csv", "txt", "md"],
    )

    if uploaded_file and st.button("Upload and Ingest"):
        with st.spinner("Uploading..."):
            response = httpx.post(
                f"{API_BASE}/upload",
                files={"file": (uploaded_file.name, uploaded_file.read(), "application/pdf")},
                timeout=120,
            )

        if response.status_code != 200:
            st.error(f"Upload failed: {response.text}")
            return

        data = response.json()
        document_id = data["document_id"]
        st.success(f"Uploaded: {data['filename']} (ID: {document_id})")

        progress = st.progress(0, text="Ingesting document...")
        for i in range(60):
            time.sleep(2)
            status_resp = httpx.get(f"{API_BASE}/upload/status/{document_id}", timeout=5)
            status = status_resp.json().get("status", "processing")
            progress.progress((i + 1) / 60, text=f"Status: {status}")
            if status == "ready":
                progress.progress(1.0, text="Ingestion complete.")
                st.balloons()
                break
            if status.startswith("error"):
                st.error(f"Ingestion error: {status}")
                break
