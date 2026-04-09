import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint

from embeddings import extract_keywords, keyword_overlap_score, sentence_windows
from extract_text import BASE_DIR, extract_pdf_text
from vector_store import Document, load_vector_store


load_dotenv()


SYSTEM_PROMPT = """You are a polished AI resume assistant.
Answer like a strong professional assistant helping the candidate with resumes, interviews, recruiter communication, and project explanations.

Rules:
- Use ONLY the provided evidence.
- Be concise, specific, and natural.
- If the user asks for career advice or fit analysis, make grounded suggestions from the resume and clearly separate strengths from possible gaps.
- If information is missing, say so plainly instead of guessing.
- Prefer bullets when the user asks for a list, skills, strengths, weaknesses, or comparisons.

User question:
{question}

Evidence:
{context}

Answer:
"""


@dataclass
class ResumeAgentResult:
    answer: str
    sources: list[Document]
    confidence: str


class ResumeRAGAgent:
    def __init__(self):
        self.vector_store = load_vector_store()
        self.llm = self._build_llm()
        self.resume_text = self._load_resume_text()

    def _build_llm(self):
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            return None

        return HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            task="text-generation",
            max_new_tokens=450,
            temperature=0.15,
            huggingfacehub_api_token=token,
        )

    def _load_resume_text(self) -> str:
        resume_path = BASE_DIR / "data" / "resume.pdf"
        if resume_path.exists():
            return extract_pdf_text(resume_path)

        resume_records = [
            record for record in self.vector_store.records if str(record.get("section", "")).lower() == "resume"
        ]
        ordered = sorted(resume_records, key=lambda record: int(record.get("chunk_id", 0)))
        return "\n".join(record["content"] for record in ordered)

    def _spaced_heading_pattern(self, heading: str) -> str:
        compact = heading.replace(" ", "")
        return r"\s*".join(re.escape(char) for char in compact)

    def _extract_resume_section(self, heading: str, next_headings: list[str]) -> str:
        if not self.resume_text.strip():
            return ""

        heading_pattern = self._spaced_heading_pattern(heading)
        next_pattern = "|".join(self._spaced_heading_pattern(item) for item in next_headings)
        pattern = rf"{heading_pattern}\s*(.*?)(?={next_pattern}|$)"
        match = re.search(pattern, self.resume_text, flags=re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""

    def _summary_text(self) -> str:
        return self._extract_resume_section(
            "SUMMARY",
            ["TECHNICAL SKILLS", "PROFESSIONAL EXPERIENCE", "KEY PROJECTS", "EDUCATION", "CERTIFICATIONS"],
        )

    def _classify_question(self, question: str) -> str:
        lowered = question.lower()
        if any(phrase in lowered for phrase in ["tell me about yourself", "summarize my profile", "profile summary"]):
            return "profile_summary"
        if any(phrase in lowered for phrase in ["strength", "weakness"]):
            return "strengths_weaknesses"
        if any(phrase in lowered for phrase in ["recruiter", "write a response", "reply to recruiter"]):
            return "recruiter_reply"
        if any(phrase in lowered for phrase in ["good fit", "fit for", "match my profile", "match score"]):
            return "job_fit"
        if any(phrase in lowered for phrase in ["improve my resume", "project suggestions", "projects to improve"]):
            return "resume_improvement"
        if any(phrase in lowered for phrase in ["simpler terms", "simple terms", "explain that project"]):
            return "project_simplify"
        if any(word in lowered for word in ["skill", "technology", "tool", "language", "stack"]):
            return "skills"
        if any(word in lowered for word in ["project", "build", "portfolio", "github"]):
            return "projects"
        if any(word in lowered for word in ["experience", "work", "role", "company", "job", "years"]):
            return "experience"
        if any(word in lowered for word in ["education", "degree", "university", "college"]):
            return "education"
        if any(word in lowered for word in ["email", "phone", "contact", "linkedin"]):
            return "contact"
        return "general"

    def _query_variants(self, question: str) -> list[str]:
        category = self._classify_question(question)
        keywords = extract_keywords(question)
        keyword_query = " ".join(keywords[:8]).strip()

        variants = [question]
        if keyword_query and keyword_query.lower() != question.lower():
            variants.append(keyword_query)
        if category != "general":
            variants.append(f"{category} {question}")

        deduped: list[str] = []
        seen = set()
        for variant in variants:
            normalized = variant.lower().strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped.append(variant)
        return deduped

    def _retrieve(self, question: str) -> list[Document]:
        candidates: list[Document] = []
        seen_chunks = set()

        for variant in self._query_variants(question):
            results = self.vector_store.search(variant, k=5)
            for result in results:
                key = (
                    result.metadata.get("source"),
                    result.metadata.get("section"),
                    result.metadata.get("chunk_id"),
                )
                if key not in seen_chunks:
                    seen_chunks.add(key)
                    candidates.append(result)

        ranked = sorted(
            candidates,
            key=lambda doc: (
                float(doc.metadata.get("score") or 0.0),
                keyword_overlap_score(question, doc.page_content),
            ),
            reverse=True,
        )
        return ranked[:6]

    def _extract_evidence_sentences(self, question: str, sources: list[Document]) -> list[str]:
        question_keywords = set(extract_keywords(question))
        evidence: list[tuple[float, str]] = []

        for source in sources:
            for window in sentence_windows(source.page_content):
                score = keyword_overlap_score(question, window)
                if question_keywords and score == 0:
                    continue
                evidence.append((score, window.strip()))

        ranked = sorted(evidence, key=lambda item: (item[0], len(item[1])), reverse=True)
        unique_sentences: list[str] = []
        seen = set()
        for _, sentence in ranked:
            normalized = sentence.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_sentences.append(sentence)
            if len(unique_sentences) == 5:
                break
        return unique_sentences

    def _build_context(self, sources: list[Document]) -> str:
        chunks = []
        for index, source in enumerate(sources, start=1):
            label = f"[Source {index} | {source.metadata.get('section')} | {source.metadata.get('source')}]"
            chunks.append(f"{label}\n{source.page_content}")
        return "\n\n".join(chunks)

    def _confidence_label(self, sources: list[Document], evidence_sentences: list[str]) -> str:
        if not sources:
            return "low"
        best_score = max(float(source.metadata.get("score") or 0.0) for source in sources)
        if best_score >= 0.6 and len(evidence_sentences) >= 2:
            return "high"
        if best_score >= 0.3:
            return "medium"
        return "low"

    def _extract_resume_skills(self) -> list[str]:
        section = self._extract_resume_section(
            "TECHNICAL SKILLS",
            ["PROFESSIONAL EXPERIENCE", "KEY PROJECTS", "EDUCATION", "CERTIFICATIONS"],
        )
        if not section:
            return []

        skills: list[str] = []
        seen = set()
        for line in section.splitlines():
            cleaned = line.strip(" •\t")
            if ":" in cleaned:
                _, values = cleaned.split(":", 1)
                for item in [part.strip() for part in values.split(",")]:
                    normalized = item.lower()
                    if item and normalized not in seen:
                        seen.add(normalized)
                        skills.append(item)
        return skills[:20]

    def _extract_resume_education(self) -> list[str]:
        section = self._extract_resume_section("EDUCATION", ["CERTIFICATIONS"])
        if not section:
            return []

        lines = [line.strip() for line in section.splitlines() if line.strip()]
        education: list[str] = []
        buffer = ""
        for line in lines:
            if any(token in line for token in ["M.S.", "B.Tech", "B.S.", "Bachelor", "Master"]):
                if buffer:
                    education.append(buffer.strip())
                buffer = line
            else:
                buffer = f"{buffer} {line}".strip()
        if buffer:
            education.append(buffer.strip())
        return education[:5]

    def _extract_resume_experience(self) -> list[dict[str, str]]:
        section = self._extract_resume_section(
            "PROFESSIONAL EXPERIENCE",
            ["KEY PROJECTS", "EDUCATION", "CERTIFICATIONS"],
        )
        if not section:
            return []

        text = section.replace("•", "\n•").replace("·", " · ")
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        roles: list[dict[str, str]] = []
        current: dict[str, str | list[str]] | None = None
        role_pattern = re.compile(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{4})", re.IGNORECASE)

        for line in lines:
            if line.startswith("•"):
                if current is not None:
                    current["bullets"].append(line.lstrip("• ").strip())
                continue

            if "·" in line and current is not None and not current.get("company"):
                current["company"] = line
                continue

            if role_pattern.search(line):
                if current is not None:
                    roles.append(
                        {
                            "role": str(current.get("role", "")),
                            "company": str(current.get("company", "")),
                            "bullets": " ".join(current.get("bullets", [])[:3]),
                        }
                    )
                current = {"role": line, "company": "", "bullets": []}

        if current is not None:
            roles.append(
                {
                    "role": str(current.get("role", "")),
                    "company": str(current.get("company", "")),
                    "bullets": " ".join(current.get("bullets", [])[:3]),
                }
            )

        return roles[:5]

    def _extract_project_list(self, sources: list[Document]) -> list[str]:
        projects: list[str] = []
        seen = set()

        for source in sources:
            for line in source.page_content.splitlines():
                cleaned = line.strip().lstrip("#").strip()
                if not cleaned or cleaned.lower() == "github projects":
                    continue
                if cleaned.lower().startswith("project ") or line.strip().startswith("## "):
                    normalized = cleaned.lower()
                    if normalized not in seen:
                        seen.add(normalized)
                        projects.append(cleaned)
        return projects[:8]

    def _extract_technology_list(self, sources: list[Document]) -> list[str]:
        technologies: list[str] = []
        seen = set()

        for source in sources:
            matches = re.findall(r"\*\*Technologies:\*\*\s*(.+)", source.page_content, flags=re.IGNORECASE)
            for match in matches:
                for item in [part.strip() for part in match.split(",")]:
                    normalized = item.lower()
                    if item and normalized not in seen:
                        seen.add(normalized)
                        technologies.append(item)

        return technologies[:15]

    def _extract_total_years(self) -> str | None:
        summary = self._summary_text()
        match = re.search(r"(\d+\+?)\s+years of experience", summary, flags=re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _project_blurb(self, project_name_hint: str | None = None) -> str | None:
        projects_section = self._extract_resume_section("KEY PROJECTS", ["EDUCATION", "CERTIFICATIONS"])
        if not projects_section:
            return None

        chunks = [chunk.strip() for chunk in re.split(r"\n(?=[A-Z0-9].+—)", projects_section) if chunk.strip()]
        if project_name_hint:
            hint = project_name_hint.lower()
            for chunk in chunks:
                if hint in chunk.lower():
                    return chunk
        return chunks[0] if chunks else None

    def _format_profile_summary(self) -> str | None:
        summary = self._summary_text()
        if not summary:
            return None
        cleaned = " ".join(summary.split())
        return (
            f"{cleaned} This background positions the candidate well for AI, data science, "
            "and full stack data application roles."
        )

    def _format_skills_answer(self) -> str | None:
        skills = self._extract_resume_skills()
        if not skills:
            return None

        categories = {
            "Machine Learning": [],
            "Data Science": [],
            "Languages": [],
            "Full Stack": [],
            "DevOps": [],
        }
        for skill in skills:
            lowered = skill.lower()
            if lowered in {"pytorch", "tensorflow", "scikit-learn", "nlp", "deep learning", "cnns"}:
                categories["Machine Learning"].append(skill)
            elif lowered in {"numpy", "pandas", "statistical modeling", "data manipulation", "large-scale dataset processing", "power bi", "d3.js"}:
                categories["Data Science"].append(skill)
            elif lowered in {"python", "sql", "c++", "javascript", "r", "matlab - scripting"}:
                categories["Languages"].append(skill)
            elif lowered in {"flask", "node.js", "react.js", "html/css", "mongodb", "rest apis", "microservices"}:
                categories["Full Stack"].append(skill)
            else:
                categories["DevOps"].append(skill)

        lines = ["Your key strengths include:"]
        for label, values in categories.items():
            if values:
                lines.append(f"- {label}: {', '.join(values[:6])}")
        return "\n".join(lines)

    def _format_experience_summary(self) -> str | None:
        roles = self._extract_resume_experience()
        if not roles:
            return None

        lines = []
        for role in roles[:4]:
            headline = role["role"]
            if role["company"]:
                headline = f"{headline} at {role['company']}"
            details = role["bullets"] or "Relevant experience is listed in the resume."
            lines.append(f"- {headline}: {details}")
        return "\n".join(lines)

    def _format_education_answer(self) -> str | None:
        education = self._extract_resume_education()
        if not education:
            return None
        return "\n".join(f"- {item}" for item in education)

    def _format_years_experience_answer(self) -> str | None:
        years = self._extract_total_years()
        if not years:
            return None
        return f"The resume states {years} years of experience, primarily in Python-based algorithm development, machine learning, and large-scale data processing."

    def _format_projects_answer(self, question: str) -> str | None:
        if any(phrase in question.lower() for phrase in ["explain", "impact", "simpler"]):
            hint = None
            if "image detection" in question.lower():
                hint = "image detection"
            elif "flowdigm" in question.lower():
                hint = "flowdigm"
            elif "conversational ai" in question.lower():
                hint = "conversational ai"
            blurb = self._project_blurb(hint)
            if blurb:
                cleaned = " ".join(blurb.split())
                if "simpler" in question.lower():
                    return f"In simple terms, {cleaned}"
                return cleaned

        projects = self._extract_project_list(self._retrieve("projects github portfolio"))
        if projects:
            return "\n".join(f"- {project}" for project in projects)
        return None

    def _format_job_fit_answer(self) -> str | None:
        summary = self._summary_text()
        skills = self._extract_resume_skills()
        if not summary and not skills:
            return None

        strengths = [
            "Python-based machine learning and algorithm development",
            "Experience with retrieval-augmented generation and NLP-style applications",
            "Full stack data application exposure with Flask, Node.js, and React.js",
        ]
        gaps = [
            "Deeper production-scale deployment experience could strengthen the profile further",
            "More explicit cloud-native or large-scale distributed system work would help for senior AI roles",
        ]
        lines = [
            "Yes, the profile is a strong fit for entry-level to early-career AI Engineer and Data Scientist roles.",
            "",
            "Strengths:",
            *(f"- {item}" for item in strengths),
            "",
            "Possible gaps:",
            *(f"- {item}" for item in gaps),
        ]
        return "\n".join(lines)

    def _format_recruiter_reply(self) -> str | None:
        return (
            "Hi [Recruiter Name],\n\n"
            "Thank you for reaching out. I’m interested in opportunities related to AI, machine learning, and data-driven application development. "
            "My background includes Python-based machine learning pipelines, retrieval-augmented generation work, and full stack data applications.\n\n"
            "I’d be glad to learn more about the role and discuss how my experience aligns with your team’s needs.\n\n"
            "Best regards,\n"
            "Nikhila Vadlamani"
        )

    def _format_strengths_weaknesses(self) -> str | None:
        return (
            "**Strengths**\n"
            "- Strong foundation in Python, machine learning, and data processing\n"
            "- Hands-on work across AI, data visualization, and full stack application development\n"
            "- Ability to work across research-style and applied engineering projects\n\n"
            "**Areas to improve**\n"
            "- Production-scale deployment depth can be strengthened further\n"
            "- Additional advanced deep learning and cloud system experience would broaden the profile even more"
        )

    def _format_resume_improvement(self) -> str | None:
        return (
            "Based on the current profile, these projects would strengthen the resume further:\n"
            "- A multimodal RAG assistant that works with text, PDFs, and images\n"
            "- A production-ready ML API with FastAPI, Docker, and monitoring\n"
            "- An edge AI or hardware-aware ML project connecting the AI background with embedded systems"
        )

    def _deterministic_answer(self, question: str) -> str | None:
        category = self._classify_question(question)
        lowered = question.lower()

        if "how many years" in lowered or "years of experience" in lowered:
            return self._format_years_experience_answer()
        if category == "profile_summary":
            return self._format_profile_summary()
        if category == "skills":
            return self._format_skills_answer()
        if category == "experience":
            return self._format_experience_summary()
        if category == "education":
            return self._format_education_answer()
        if category == "projects" or category == "project_simplify":
            return self._format_projects_answer(question)
        if category == "job_fit":
            return self._format_job_fit_answer()
        if category == "recruiter_reply":
            return self._format_recruiter_reply()
        if category == "strengths_weaknesses":
            return self._format_strengths_weaknesses()
        if category == "resume_improvement":
            return self._format_resume_improvement()
        return None

    def _answer_without_llm(self, question: str, sources: list[Document], evidence_sentences: list[str]) -> str:
        deterministic = self._deterministic_answer(question)
        if deterministic:
            return deterministic

        if not sources or not evidence_sentences:
            return "I don't have that information in the provided documents."

        lowered = question.lower()
        if "list" in lowered:
            return "\n".join(f"- {sentence}" for sentence in evidence_sentences[:4])

        if any(phrase in lowered for phrase in ["tell me about yourself", "summarize"]):
            return " ".join(evidence_sentences[:3])

        return " ".join(evidence_sentences[:2])

    def _answer_with_llm(self, question: str, sources: list[Document], evidence_sentences: list[str]) -> str:
        deterministic = self._deterministic_answer(question)
        if deterministic:
            return deterministic

        if not self.llm:
            return self._answer_without_llm(question, sources, evidence_sentences)

        context = self._build_context(sources)
        prompt = SYSTEM_PROMPT.format(question=question, context=context)
        response = self.llm.invoke(prompt)
        return response.strip()

    def ask(self, question: str) -> ResumeAgentResult:
        sources = self._retrieve(question)
        evidence_sentences = self._extract_evidence_sentences(question, sources)
        deterministic = self._deterministic_answer(question)
        answer = deterministic or self._answer_with_llm(question, sources, evidence_sentences)
        confidence = "high" if deterministic else self._confidence_label(sources, evidence_sentences)
        return ResumeAgentResult(answer=answer, sources=sources, confidence=confidence)


def create_rag_chain():
    return ResumeRAGAgent()


def ask_question(chain: ResumeRAGAgent, question: str):
    result = chain.ask(question)
    return {
        "answer": result.answer,
        "sources": result.sources,
        "confidence": result.confidence,
    }


if __name__ == "__main__":
    agent = create_rag_chain()
    questions = [
        "Can you summarize my profile?",
        "What are my top skills?",
        "How many years of experience do I have?",
        "Am I a good fit for an AI Engineer role?",
    ]

    for question in questions:
        result = ask_question(agent, question)
        print(f"\nQ: {question}")
        print(f"A: {result['answer']}")
        print(f"Confidence: {result['confidence']}")
        print("-" * 60)
