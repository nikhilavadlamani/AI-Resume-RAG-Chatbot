from pathlib import Path

import streamlit as st

from extract_text import load_documents
from rag_pipeline import ask_question, create_rag_chain
from vector_store import VECTOR_STORE_FILE, build_vector_store


st.set_page_config(
    page_title="AI Resume LLM RAG Chatbot",
    page_icon="💼",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_rag_chain():
    return create_rag_chain()


def refresh_chain():
    get_rag_chain.clear()
    return get_rag_chain()


def ensure_index_exists() -> None:
    if not VECTOR_STORE_FILE.exists():
        build_vector_store()


def format_source_label(source) -> str:
    score = source.metadata.get("score")
    score_suffix = f" (score: {score})" if score is not None else ""
    return f"{source.metadata.get('section', 'Document')} - {source.metadata.get('source', 'Unknown source')}{score_suffix}"


def main():
    st.title("AI Resume LLM RAG Chatbot")
    st.caption("Ask exact questions about the candidate's resume and projects. Answers stay grounded in the indexed documents.")

    with st.sidebar:
        st.subheader("Project Setup")
        data_files = load_documents()
        st.write(f"Documents found: {len(data_files)}")
        for document in data_files:
            st.write(f"- {document['section']}: `{document['source']}`")

        index_ready = VECTOR_STORE_FILE.exists()
        st.write(f"Local index ready: {'Yes' if index_ready else 'No'}")

        if st.button("Build / Refresh Index", use_container_width=True):
            with st.spinner("Building local retrieval index..."):
                build_vector_store()
                refresh_chain()
            st.success("Index is ready.")

        st.markdown("Sample questions")
        sample_questions = [
            "Can you summarize my profile?",
            "What are my top skills?",
            "How many years of experience do I have?",
            "Am I a good fit for an AI Engineer role?",
            "Write a response to a recruiter reaching out for an AI role.",
            "Give me projects to improve my resume.",
        ]
        for question in sample_questions:
            if st.button(question, use_container_width=True):
                st.session_state["pending_question"] = question

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

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("confidence"):
                st.caption(f"Answer confidence: {message['confidence'].title()}")
            if message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.markdown(f"**{format_source_label(source)}**")
                        st.write(source.page_content[:500] + ("..." if len(source.page_content) > 500 else ""))

    user_question = st.chat_input("Ask a question about the candidate")
    if not user_question and st.session_state.get("pending_question"):
        user_question = st.session_state.pop("pending_question")

    if user_question:
        st.session_state["messages"].append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the resume knowledge base..."):
                ensure_index_exists()
                chain = get_rag_chain()
                result = ask_question(chain, user_question)

            st.markdown(result["answer"])
            st.caption(f"Answer confidence: {result.get('confidence', 'unknown').title()}")
            with st.expander("Sources"):
                for source in result["sources"]:
                    st.markdown(f"**{format_source_label(source)}**")
                    st.write(source.page_content[:500] + ("..." if len(source.page_content) > 500 else ""))

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
