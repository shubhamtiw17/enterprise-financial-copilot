import streamlit as st
from frontend.components.sidebar import render_sidebar
from frontend.pages.chat import render_chat
from frontend.pages.upload import render_upload
from frontend.pages.analytics import render_analytics

st.set_page_config(
    page_title="Financial Copilot",
    page_icon="📊",
    layout="wide",
)

config = render_sidebar()

page = st.sidebar.selectbox(
    "Page",
    ["Chat", "Upload", "Analytics"],
    label_visibility="collapsed",
)

if page == "Chat":
    render_chat(config["query_mode"])
elif page == "Upload":
    render_upload()
elif page == "Analytics":
    render_analytics()
