from __future__ import annotations

import math
import re
from collections import Counter


TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9+#./-]+\b")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "of", "on", "or", "that", "the", "their", "they", "this",
    "to", "was", "what", "which", "who", "with",
}


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def split_into_chunks(text: str, chunk_size: int = 900, chunk_overlap: int = 180) -> list[str]:
    if not text.strip():
        return []
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()] or [text.strip()]
    chunks: list[str] = []
    current_chunk = ""

    for paragraph in paragraphs:
        candidate = f"{current_chunk}\n\n{paragraph}".strip() if current_chunk else paragraph
        if len(candidate) <= chunk_size:
            current_chunk = candidate
            continue
        if current_chunk:
            chunks.append(current_chunk)
        if len(paragraph) <= chunk_size:
            current_chunk = paragraph
            continue
        start = 0
        while start < len(paragraph):
            end = min(start + chunk_size, len(paragraph))
            piece = paragraph[start:end].strip()
            if piece:
                chunks.append(piece)
            if end == len(paragraph):
                break
            start = max(end - chunk_overlap, start + 1)
        current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def extract_keywords(text: str) -> list[str]:
    return [token for token in tokenize(text) if len(token) >= 2 and token not in STOPWORDS]


def build_term_frequencies(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot_product = sum(left[token] * right.get(token, 0) for token in left)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def keyword_overlap_score(query: str, text: str) -> float:
    query_keywords = set(extract_keywords(query))
    if not query_keywords:
        return 0.0
    text_tokens = set(tokenize(text))
    return len(query_keywords & text_tokens) / len(query_keywords)


def exact_phrase_score(query: str, text: str) -> float:
    normalized_query = normalize_text(query)
    normalized_text = normalize_text(text)
    if not normalized_query or len(normalized_query) < 5:
        return 0.0
    if normalized_query in normalized_text:
        return 1.0
    for phrase in [part.strip() for part in re.split(r"[?,]", normalized_query) if len(part.strip()) >= 5]:
        if phrase in normalized_text:
            return 0.6
    return 0.0


def sentence_windows(text: str) -> list[str]:
    sentences = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(text) if part.strip()]
    windows: list[str] = []
    for index, sentence in enumerate(sentences):
        windows.append(sentence)
        if index + 1 < len(sentences):
            windows.append(f"{sentence} {sentences[index + 1]}")
    return windows or [text.strip()]
