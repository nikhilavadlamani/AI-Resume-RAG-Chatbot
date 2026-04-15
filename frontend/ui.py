from __future__ import annotations

import os
import uuid

import requests
import streamlit as st

from app.config import get_settings
from app.services.documents import load_documents


settings = get_settings()
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


st.set_page_config(page_title=settings.app_name, page_icon="💼", layout="wide")


def format_source_label(source: dict) -> str:
    score = source.get("score")
    score_suffix = f" (score: {score})" if score is not None else ""
    return f"{source.get('section', 'Document')} - {source.get('source', 'Unknown source')}{score_suffix}"


def call_api(question: str, conversation_id: str) -> dict:
    response = requests.post(
        f"{API_BASE_URL}{settings.api_prefix}/chat",
        json={"question": question, "conversation_id": conversation_id},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def rebuild_index() -> None:
    response = requests.post(f"{API_BASE_URL}{settings.api_prefix}/index/rebuild", timeout=120)
    response.raise_for_status()


def main():
    st.title(settings.app_name)
    st.caption("Ask grounded questions about the candidate's resume and GitHub projects through the FastAPI backend.")

    if "conversation_id" not in st.session_state:
        st.session_state["conversation_id"] = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": (
                    "I can help with resume summaries, skills, work experience, project explanations, "
                    "AI job fit, recruiter replies, and interview-style questions."
                ),
            }
        ]

    with st.sidebar:
        st.subheader("Project Setup")
        data_files = load_documents()
        st.write(f"Documents found: {len(data_files)}")
        for document in data_files:
            st.write(f"- {document['section']}: `{document['source']}`")

        if st.button("Build / Refresh Index", use_container_width=True):
            with st.spinner("Rebuilding retrieval index through the API..."):
                rebuild_index()
            st.success("Index is ready.")

        st.markdown("Sample questions")
        for question in [
            "Can you summarize my profile?",
            "What are my top skills?",
            "How many years of experience do I have?",
            "Am I a good fit for an AI Engineer role?",
            "Write a response to a recruiter reaching out for an AI role.",
            "Give me projects to improve my resume.",
        ]:
            if st.button(question, use_container_width=True):
                st.session_state["pending_question"] = question

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("confidence"):
                st.caption(f"Answer confidence: {message['confidence'].title()}")
            if message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.markdown(f"**{format_source_label(source)}**")
                        content = source.get("content", "")
                        st.write(content[:500] + ("..." if len(content) > 500 else ""))

    user_question = st.chat_input("Ask a question about the candidate")
    if not user_question and st.session_state.get("pending_question"):
        user_question = st.session_state.pop("pending_question")

    if user_question:
        st.session_state["messages"].append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the resume knowledge base..."):
                result = call_api(user_question, st.session_state["conversation_id"])
            st.markdown(result["answer"])
            st.caption(f"Answer confidence: {result.get('confidence', 'unknown').title()}")
            with st.expander("Sources"):
                for source in result["sources"]:
                    st.markdown(f"**{format_source_label(source)}**")
                    content = source.get("content", "")
                    st.write(content[:500] + ("..." if len(content) > 500 else ""))

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "confidence": result.get("confidence"),
            }
        )


if __name__ == "__main__":
    main()
