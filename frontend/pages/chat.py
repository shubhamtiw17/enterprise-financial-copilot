import streamlit as st
import httpx
import os
from frontend.components.citation_card import render_citations

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


def render_chat(query_mode: str) -> None:
    st.title("Ask your financial documents")

    if query_mode == "Multi-Agent":
        st.info("Multi-Agent mode: planner → parallel research → analysis → summary.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sub_questions"):
                with st.expander("Planner sub-questions"):
                    for i, q in enumerate(msg["sub_questions"], 1):
                        st.markdown(f"**{i}.** {q}")
            if msg.get("analysis"):
                with st.expander("Financial analysis"):
                    st.markdown(msg["analysis"])
            render_citations(msg.get("citations", []))

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            use_agents = query_mode == "Multi-Agent"
            spinner = "Running multi-agent pipeline..." if use_agents else "Thinking..."

            with st.spinner(spinner):
                try:
                    if use_agents:
                        resp = httpx.post(
                            f"{API_BASE}/agent-query",
                            json={"question": prompt},
                            timeout=120,
                        ).json()
                        answer = resp.get("answer", "No answer returned.")
                        citations = resp.get("citations", [])
                        sub_questions = resp.get("sub_questions", [])
                        analysis = resp.get("analysis", "")
                    else:
                        resp = httpx.post(
                            f"{API_BASE}/query",
                            json={"question": prompt, "top_k": 6},
                            timeout=60,
                        ).json()
                        answer = resp.get("answer", "No answer returned.")
                        citations = resp.get("citations", [])
                        sub_questions = []
                        analysis = ""
                except Exception as e:
                    answer = f"Error: {e}"
                    citations, sub_questions, analysis = [], [], ""

            st.markdown(answer)
            if sub_questions:
                with st.expander("Planner sub-questions"):
                    for i, q in enumerate(sub_questions, 1):
                        st.markdown(f"**{i}.** {q}")
            if analysis:
                with st.expander("Financial analysis"):
                    st.markdown(analysis)
            render_citations(citations)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "sub_questions": sub_questions,
            "analysis": analysis,
        })
