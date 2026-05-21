import streamlit as st
import httpx
import os

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Financial Copilot",
    page_icon="📊",
    layout="wide"
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Financial Copilot")
    st.caption("Upload financial documents and ask questions.")
    st.divider()

    st.subheader("Upload Document")
    uploaded_file = st.file_uploader(
        "SEC filings, annual reports, PDFs",
        type=["pdf", "csv", "txt"]
    )

    if uploaded_file and st.button("Ingest Document"):
        with st.spinner("Uploading..."):
            response = httpx.post(
                f"{API_BASE}/upload",
                files={"file": (uploaded_file.name, uploaded_file.read(), "application/pdf")},
                timeout=120
            )
            if response.status_code == 200:
                data = response.json()
                st.success(f"Uploaded: {data['filename']}")
                st.caption(f"Document ID: {data['document_id']}")
                st.session_state["document_id"] = data["document_id"]
            else:
                st.error(f"Upload failed: {response.text}")

    st.divider()

    # Health check
    try:
        health = httpx.get(f"{API_BASE}/health", timeout=3).json()
        st.caption(f"API: {health['status']} · {health['llm_provider']}")
    except Exception:
        st.caption("API: unreachable")

# ── Main Chat ──────────────────────────────────────────────────────────────────
st.title("Ask your financial documents")
st.caption("Upload a PDF in the sidebar, then ask questions below.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander(f"Sources ({len(msg['citations'])} citations)"):
                for c in msg["citations"]:
                    st.markdown(
                        f"**{c['filename']}** · Chunk {c['chunk_index']} · Score: {c['score']}"
                    )
                    st.caption(c["excerpt"])
                    st.divider()

# Chat input
if prompt := st.chat_input("Write a message.."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer from API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = httpx.post(
                    f"{API_BASE}/query",
                    json={"question": prompt, "top_k": 6},
                    timeout=60
                ).json()

                answer = response.get("answer", "No answer returned.")
                citations = response.get("citations", [])

            except Exception as e:
                answer = f"Error: {e}"
                citations = []

        st.markdown(answer)

        if citations:
            with st.expander(f"Sources ({len(citations)} citations)"):
                for c in citations:
                    st.markdown(
                        f"**{c['filename']}** · Chunk {c['chunk_index']} · Score: {c['score']}"
                    )
                    st.caption(c["excerpt"])
                    st.divider()

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations
    })