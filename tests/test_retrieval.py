from app.components.hybrid_retriever import LocalVectorStore
from app.components.reranker import SimpleReranker


def test_hybrid_retriever_returns_project_match():
    store = LocalVectorStore(
        [
            {
                "content": "Built a multimodal RAG chatbot project with FastAPI and Streamlit.",
                "source": "github_projects.txt",
                "section": "GitHub Projects",
                "chunk_id": 1,
                "term_frequencies": {"built": 1, "multimodal": 1, "rag": 1, "chatbot": 1, "project": 1},
            },
            {
                "content": "Worked on data analysis and reporting.",
                "source": "resume.pdf",
                "section": "Resume",
                "chunk_id": 2,
                "term_frequencies": {"worked": 1, "data": 1, "analysis": 1},
            },
        ]
    )

    results = store.search("rag chatbot project", k=2)

    assert results
    assert results[0].metadata["section"] == "GitHub Projects"


def test_reranker_prefers_keyword_overlap():
    reranker = SimpleReranker()
    candidates = [
        type("Doc", (), {"page_content": "Python machine learning projects", "metadata": {"score": 0.4}})(),
        type("Doc", (), {"page_content": "Java backend services", "metadata": {"score": 0.4}})(),
    ]

    ranked = reranker.rerank("python machine learning", candidates)

    assert ranked[0].page_content == "Python machine learning projects"
