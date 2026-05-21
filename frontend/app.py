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

    # Query mode toggle
    st.subheader("Query Mode")
    query_mode = st.radio(
        "Select mode",
        options=["Standard RAG", "Multi-Agent (Deep Analysis)"],
        help=(
            "Standard RAG: single retrieval pass, fast.\n"
            "Multi-Agent: planner → parallel research → analysis → summary. "
            "Best for complex or multi-company questions."
        ),
    )
    st.session_state["query_mode"] = query_mode

    st.divider()

    # Health check
    try:
        health = httpx.get(f"{API_BASE}/health", timeout=3).json()
        st.caption(f"API: {health['status']} · {health['llm_provider']}")
    except Exception:
        st.caption("API: unreachable")

# ── Main Chat ──────────────────────────────────────────────────────────────────
st.title("Ask your financial documents")

mode_label = st.session_state.get("query_mode", "Standard RAG")
if mode_label == "Multi-Agent (Deep Analysis)":
    st.info(
        "Multi-Agent mode active. The planner will break your question into "
        "sub-questions, research each one in parallel, then synthesize a final answer.",
        icon="🤖",
    )
else:
    st.caption("Upload a PDF in the sidebar, then ask questions below.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("sub_questions"):
            with st.expander("Planner sub-questions"):
                for i, q in enumerate(msg["sub_questions"], 1):
                    st.markdown(f"**{i}.** {q}")

        if msg.get("analysis"):
            with st.expander("Financial analysis (intermediate step)"):
                st.markdown(msg["analysis"])

        if msg.get("citations"):
            with st.expander(f"Sources ({len(msg['citations'])} citations)"):
                for c in msg["citations"]:
                    st.markdown(
                        f"**{c['filename']}** · Chunk {c['chunk_index']} · Score: {c['score']}"
                    )
                    st.caption(c["excerpt"])
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a question about your financial documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        use_agents = st.session_state.get("query_mode") == "Multi-Agent (Deep Analysis)"

        spinner_msg = (
            "Running multi-agent pipeline (planner → research → analysis → summary)..."
            if use_agents
            else "Thinking..."
        )

        with st.spinner(spinner_msg):
            try:
                if use_agents:
                    response = httpx.post(
                        f"{API_BASE}/agent-query",
                        json={"question": prompt},
                        timeout=120,
                    ).json()
                    answer = response.get("answer", "No answer returned.")
                    citations = response.get("citations", [])
                    sub_questions = response.get("sub_questions", [])
                    analysis = response.get("analysis", "")
                else:
                    response = httpx.post(
                        f"{API_BASE}/query",
                        json={"question": prompt, "top_k": 6},
                        timeout=60,
                    ).json()
                    answer = response.get("answer", "No answer returned.")
                    citations = response.get("citations", [])
                    sub_questions = []
                    analysis = ""

            except Exception as e:
                answer = f"Error contacting API: {e}"
                citations = []
                sub_questions = []
                analysis = ""

        st.markdown(answer)

        if sub_questions:
            with st.expander("Planner sub-questions"):
                for i, q in enumerate(sub_questions, 1):
                    st.markdown(f"**{i}.** {q}")

        if analysis:
            with st.expander("Financial analysis (intermediate step)"):
                st.markdown(analysis)

        if citations:
            with st.expander(f"Sources ({len(citations)} citations)"):
                for c in citations:
                    st.markdown(
                        f"**{c['filename']}** · Chunk {c['chunk_index']} · Score: {c['score']}"
                    )
                    st.caption(c["excerpt"])
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
        "sub_questions": sub_questions,
        "analysis": analysis,
    })
