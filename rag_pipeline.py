from langchain_huggingface import HuggingFaceEndpoint
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from vector_store import load_vector_store
from dotenv import load_dotenv
import os

load_dotenv()

SYSTEM_PROMPT = """You are a professional AI assistant representing a job candidate.
Answer questions based ONLY on the context below from the resume and GitHub projects.
Be professional, concise, and specific.
If the information is not in the context, say "I don't have that information in the provided documents."
Do not make up or assume any information.

Context:
{context}

Question: {question}

Answer:"""

def create_rag_chain():
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Free HuggingFace Inference API — Mistral 7B is strong and free
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.3,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    prompt = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return chain

def ask_question(chain, question):
    result = chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": result["source_documents"]
    }

if __name__ == "__main__":
    print("Building RAG chain...")
    chain = create_rag_chain()

    questions = [
        "What projects has this candidate built?",
        "What programming languages do they know?",
        "What is their work experience?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        result = ask_question(chain, q)
        print(f"A: {result['answer']}")
        print("-" * 60)