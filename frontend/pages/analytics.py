import streamlit as st
import httpx
import os

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


def render_analytics() -> None:
    st.title("System Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("API Health")
        try:
            health = httpx.get(f"{API_BASE}/health", timeout=3).json()
            st.metric("Status", health.get("status", "unknown"))
            st.metric("LLM Provider", health.get("llm_provider", "unknown"))
            st.metric("Environment", health.get("environment", "unknown"))
        except Exception:
            st.error("API unreachable")

    with col2:
        st.subheader("Uptime")
        try:
            metrics = httpx.get(f"{API_BASE}/metrics", timeout=3).json()
            uptime = metrics.get("uptime_seconds", 0)
            hours = int(uptime // 3600)
            minutes = int((uptime % 3600) // 60)
            st.metric("Uptime", f"{hours}h {minutes}m")
        except Exception:
            st.error("Metrics unavailable")
