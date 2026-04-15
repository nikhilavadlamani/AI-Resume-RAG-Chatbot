from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_huggingface import HuggingFaceEndpoint

from app.agents.adaptive_router import AdaptiveRouter
from app.agents.document_grader import DocumentGrader
from app.agents.tools.vector_search import search_vector_store
from app.components.hybrid_retriever import Document, load_vector_store
from app.components.reranker import SimpleReranker
from app.config import DATA_DIR, get_settings
from app.models import ChatResponse, SourceDocument
from app.prompts.registry import get_prompt
from app.security.content_filter import ContentFilter
from app.security.input_guard import InputGuard
from app.security.output_filter import OutputFilter
from app.services.documents import extract_pdf_text
from app.services.query_rewriter import QueryRewriter
from app.services.text_utils import extract_keywords, keyword_overlap_score, sentence_windows


@dataclass
class ResumeAgentResult:
    answer: str
    sources: list[Document]
    confidence: str
    route: str


class ResumeRAGPipeline:
    def __init__(self):
        self.settings = get_settings()
        self.vector_store = load_vector_store()
        self.input_guard = InputGuard()
        self.content_filter = ContentFilter()
        self.output_filter = OutputFilter()
        self.router = AdaptiveRouter()
        self.query_rewriter = QueryRewriter()
        self.reranker = SimpleReranker()
        self.document_grader = DocumentGrader()
        self.llm = self._build_llm()
        self.resume_text = self._load_resume_text()

    def _build_llm(self):
        token = self.settings.huggingfacehub_api_token
        if not token:
            return None
        return HuggingFaceEndpoint(
            repo_id=self.settings.huggingface_repo_id,
            task="text-generation",
            max_new_tokens=self.settings.max_new_tokens,
            temperature=self.settings.temperature,
            huggingfacehub_api_token=token,
        )

    def _load_resume_text(self) -> str:
        for candidate in [DATA_DIR / "resume.pdf", DATA_DIR / "raw" / "resume.pdf"]:
            if candidate.exists():
                return extract_pdf_text(candidate)
        resume_records = [record for record in self.vector_store.records if str(record.get("section", "")).lower() == "resume"]
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

    def _build_context(self, sources: list[Document]) -> str:
        chunks = []
        for index, source in enumerate(sources, start=1):
            label = f"[Source {index} | {source.metadata.get('section')} | {source.metadata.get('source')}]"
            chunks.append(f"{label}\n{source.page_content}")
        return self.content_filter.filter_context("\n\n".join(chunks))

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
            cleaned = line.strip(" \t-•")
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
        section = self._extract_resume_section("PROFESSIONAL EXPERIENCE", ["KEY PROJECTS", "EDUCATION", "CERTIFICATIONS"])
        if not section:
            return []
        normalized_section = section.replace("â€¢", "\nâ€¢").replace("Â·", " Â· ").replace("•", "\n•")
        lines = [line.strip() for line in normalized_section.splitlines() if line.strip()]
        roles: list[dict[str, str]] = []
        current: dict[str, str | list[str]] | None = None
        role_pattern = re.compile(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{4})", re.IGNORECASE)

        for line in lines:
            if line.startswith(("â€¢", "•")):
                if current is not None:
                    current["bullets"].append(line.lstrip("â€¢• ").strip())
                continue
            if "Â·" in line and current is not None and not current.get("company"):
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

    def _extract_total_years(self) -> str | None:
        summary = self._summary_text()
        match = re.search(r"(\d+\+?)\s+years of experience", summary, flags=re.IGNORECASE)
        return match.group(1) if match else None

    def _project_blurb(self, project_name_hint: str | None = None) -> str | None:
        projects_section = self._extract_resume_section("KEY PROJECTS", ["EDUCATION", "CERTIFICATIONS"])
        if not projects_section:
            return None
        chunks = [chunk.strip() for chunk in re.split(r"\n(?=[A-Z0-9].+â€”)", projects_section) if chunk.strip()]
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
        return f"{cleaned} This background positions the candidate well for AI, data science, and full stack data application roles."

    def _format_skills_answer(self) -> str | None:
        skills = self._extract_resume_skills()
        if not skills:
            return None
        categories = {"Machine Learning": [], "Data Science": [], "Languages": [], "Full Stack": [], "DevOps": []}
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
        lowered = question.lower()
        if any(phrase in lowered for phrase in ["explain", "impact", "simpler"]):
            hint = None
            if "image detection" in lowered:
                hint = "image detection"
            elif "flowdigm" in lowered:
                hint = "flowdigm"
            elif "conversational ai" in lowered:
                hint = "conversational ai"
            blurb = self._project_blurb(hint)
            if blurb:
                cleaned = " ".join(blurb.split())
                return f"In simple terms, {cleaned}" if "simpler" in lowered else cleaned
        projects = self._extract_project_list(self._retrieve("projects github portfolio", "projects"))
        return "\n".join(f"- {project}" for project in projects) if projects else None

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

    def _deterministic_answer(self, question: str, route: str) -> str | None:
        lowered = question.lower()
        if "how many years" in lowered or "years of experience" in lowered:
            return self._format_years_experience_answer()
        if route == "profile_summary":
            return self._format_profile_summary()
        if route == "skills":
            return self._format_skills_answer()
        if route == "experience":
            return self._format_experience_summary()
        if route == "education":
            return self._format_education_answer()
        if route in {"projects", "project_simplify"}:
            return self._format_projects_answer(question)
        if route == "job_fit":
            return self._format_job_fit_answer()
        if route == "recruiter_reply":
            return self._format_recruiter_reply()
        if route == "strengths_weaknesses":
            return self._format_strengths_weaknesses()
        if route == "resume_improvement":
            return self._format_resume_improvement()
        return None

    def _answer_without_llm(self, question: str, sources: list[Document], evidence_sentences: list[str], route: str) -> str:
        deterministic = self._deterministic_answer(question, route)
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

    def _answer_with_llm(self, question: str, sources: list[Document], evidence_sentences: list[str], route: str) -> str:
        deterministic = self._deterministic_answer(question, route)
        if deterministic:
            return deterministic
        if not self.llm:
            return self._answer_without_llm(question, sources, evidence_sentences, route)
        prompt = get_prompt().format(question=question, context=self._build_context(sources))
        response = self.llm.invoke(prompt)
        return str(response).strip()

    def _retrieve(self, question: str, route: str) -> list[Document]:
        queries = self.query_rewriter.rewrite(question, route)
        candidates = search_vector_store(self.vector_store, queries, k=5)
        return self.reranker.rerank(question, self.document_grader.keep(candidates))

    def ask(self, question: str) -> ResumeAgentResult:
        question = self.input_guard.validate(question)
        route = self.router.select_route(question)
        sources = self._retrieve(question, route)
        evidence_sentences = self._extract_evidence_sentences(question, sources)
        deterministic = self._deterministic_answer(question, route)
        answer = deterministic or self._answer_with_llm(question, sources, evidence_sentences, route)
        confidence = "high" if deterministic else self._confidence_label(sources, evidence_sentences)
        return ResumeAgentResult(
            answer=self.output_filter.filter_answer(answer),
            sources=sources,
            confidence=confidence,
            route=route,
        )

    def ask_as_response(self, question: str, conversation_id: str | None = None, cached: bool = False) -> ChatResponse:
        result = self.ask(question)
        return ChatResponse(
            answer=result.answer,
            confidence=result.confidence,
            route=result.route,
            conversation_id=conversation_id,
            cached=cached,
            sources=[
                SourceDocument(
                    source=source.metadata.get("source", "unknown"),
                    section=source.metadata.get("section", "unknown"),
                    chunk_id=source.metadata.get("chunk_id"),
                    score=source.metadata.get("score"),
                    content=source.page_content,
                )
                for source in result.sources
            ],
        )


def create_rag_chain() -> ResumeRAGPipeline:
    return ResumeRAGPipeline()


def ask_question(chain: ResumeRAGPipeline, question: str):
    result = chain.ask(question)
    return {"answer": result.answer, "sources": result.sources, "confidence": result.confidence, "route": result.route}
