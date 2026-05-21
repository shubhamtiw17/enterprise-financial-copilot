import streamlit as st


def render_citation_card(citation: dict, index: int) -> None:
    st.markdown(f"**[{index}] {citation['filename']}** · Page chunk {citation['chunk_index']} · Score: `{citation['score']}`")
    st.caption(citation["excerpt"])
    st.divider()


def render_citations(citations: list) -> None:
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)} citations)"):
        for i, c in enumerate(citations, 1):
            render_citation_card(c, i)
