"""A small, typed RAG system for the financial-report corpus."""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np
import requests
from dotenv import load_dotenv
from langsmith import traceable as _traceable
from pypdf import PdfReader
from requests.models import CONTENT_CHUNK_SIZE

try:
    import fitz
except ImportError:  # pragma: no cover - pypdf remains a supported fallback
    fitz = None

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = DATA_DIR / "reports"
DOCUMENTS_FILE = DATA_DIR / "documents.jsonl"
INDEX_DIR = ROOT / "index"
INDEX_CONFIG = INDEX_DIR / "config.json"
INDEX_VERSION = "page-layout-v2"
_LLM_LIMITER: RateLimiter | None = None


def _status(message: str) -> None:
    print(f"[rag] {message}", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class PageChunk:
    """A searchable piece of one PDF page."""

    doc_id: str
    page: int
    organization: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    chunk: PageChunk
    score: float


class CourseEmbeddings:
    """The course's deepvk/USER-bge-m3 embedding service from lecture.md."""

    def __init__(
        self, api_key: str, url: str, workers: int = 4, requests_per_minute: int = 60
    ) -> None:
        if not api_key:
            raise RuntimeError(
                "EMB_KEY is not set. Add the course embedding key to .env."
            )
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.url = url.rstrip("/")
        self.workers = min(workers, 4)
        self.limiter = RateLimiter(requests_per_minute, burst=20)

    def embed_query(self, text: str) -> list[float]:
        retries = 0
        while True:
            self.limiter.wait()
            response = requests.post(
                f"{self.url}/embed",
                headers=self.headers,
                json={"text": text[:20000]},
                timeout=60,
            )
            if response.status_code == 429:
                retries += 1
                _status(f"Embedding rate limit (429); waiting before retry #{retries}.")
                time.sleep(_retry_delay(response, self.limiter.interval))
                continue
            if response.status_code in (500, 502, 503, 504):
                retries += 1
                _status(
                    f"Embedding service returned {response.status_code}; retry #{retries}."
                )
                time.sleep(_retry_delay(response, self.limiter.interval))
                continue
            response.raise_for_status()
            break
        embedding = response.json().get("embedding")
        if not isinstance(embedding, list) or not all(
            isinstance(value, (int, float)) for value in embedding
        ):
            raise ValueError("The embedding service returned an invalid embedding.")
        return [float(value) for value in embedding]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        total = len(texts)
        _status(
            f"Embedding {total} chunks with {self.workers} workers "
            "(limit: 60 requests/minute, burst 20)."
        )
        started = time.monotonic()
        embeddings: list[list[float] | None] = [None] * total
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures: dict[Future[list[float]], int] = {
                pool.submit(self.embed_query, text): index
                for index, text in enumerate(texts)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                embeddings[futures[future]] = future.result()
                if completed == total or completed % 10 == 0:
                    elapsed = time.monotonic() - started
                    rate = completed / elapsed if elapsed else 0.0
                    remaining = (total - completed) / rate if rate else 0.0
                    _status(
                        f"Embedded {completed}/{total} chunks "
                        f"({rate:.2f}/s, ETA {remaining / 60:.1f} min)."
                    )
        return [embedding for embedding in embeddings if embedding is not None]


class EmbeddingIndex:
    """A cosine-similarity index backed by the course embeddings."""

    def __init__(self, chunks: Sequence[PageChunk], vectors: np.ndarray) -> None:
        self.chunks = list(chunks)
        self.vectors = vectors

    @classmethod
    def build(
        cls, chunks: Sequence[PageChunk], embedder: CourseEmbeddings
    ) -> EmbeddingIndex:
        vectors = np.asarray(embedder.embed_documents([chunk.text for chunk in chunks]))
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.where(norms == 0, 1, norms)
        return cls(chunks, vectors)

    def search(
        self, query: str, embedder: CourseEmbeddings, top_k: int = 8
    ) -> list[SearchResult]:
        if not self.chunks:
            return []
        query_vector = np.asarray(embedder.embed_query(query), dtype=np.float32)
        norm = float(np.linalg.norm(query_vector))
        if norm == 0:
            return []
        query_vector /= norm
        scores = self.vectors @ query_vector
        candidate_count = min(len(self.chunks), max(top_k * 8, 32))
        indices = np.argsort(scores)[::-1][:candidate_count]
        query_terms = _terms(query)
        query_years = set(re.findall(r"20\d{2}", query))
        ranked: list[SearchResult] = []
        for index in indices:
            chunk = self.chunks[int(index)]
            score = float(scores[int(index)])
            chunk_terms = _terms(f"{chunk.doc_id} {chunk.organization} {chunk.text}")
            lexical_overlap = len(query_terms & chunk_terms)
            score += min(0.10, lexical_overlap * 0.01)
            if query_years:
                matching_year = any(year in chunk.doc_id for year in query_years)
                score += 0.08 if matching_year else -0.015
            ranked.append(SearchResult(chunk, score))
        ranked.sort(key=lambda result: result.score, reverse=True)

        # A page may contain several chunks. Returning only its best chunk
        # prevents one long page from consuming the whole context budget.
        selected: list[SearchResult] = []
        seen_pages: set[tuple[str, int]] = set()
        for result in ranked:
            key = (result.chunk.doc_id, result.chunk.page)
            if key in seen_pages:
                continue
            seen_pages.add(key)
            selected.append(result)
            if len(selected) == top_k:
                break
        return selected

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "chunks.json").write_text(
            json.dumps([asdict(chunk) for chunk in self.chunks], ensure_ascii=False),
            encoding="utf-8",
        )
        np.save(directory / "vectors.npy", self.vectors)
        INDEX_CONFIG.write_text(
            json.dumps(
                {"version": INDEX_VERSION, "embedding_model": "deepvk/USER-bge-m3"},
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "EmbeddingIndex":
        chunks_data = json.loads(
            (directory / "chunks.json").read_text(encoding="utf-8")
        )
        chunks = [PageChunk(**item) for item in chunks_data]
        return cls(chunks, np.load(directory / "vectors.npy"))


def _read_document_metadata() -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    with DOCUMENTS_FILE.open(encoding="utf-8") as source:
        for line in source:
            item = json.loads(line)
            metadata[str(item["doc_id"])] = {
                "path": str(item["path"]),
                "organization": str(item["organization"]),
            }
    return metadata


def _terms(text: str) -> set[str]:
    return {
        term.lower()
        for term in re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", text)
        if len(term) > 2
    }


def _chunks_from_pdf(doc_id: str, metadata: Mapping[str, str]) -> list[PageChunk]:
    path = REPORTS_DIR / metadata["path"]
    if not path.exists():
        raise FileNotFoundError(
            f"Report {doc_id} is missing: {path}. Extract data/reports.zip into data/reports."
        )
    chunks: list[PageChunk] = []
    if fitz is not None:
        pdf = fitz.open(str(path))
        pages: Iterable[tuple[int, str | None]] = (
            (page_number, pdf[page_number - 1].get_text("text", sort=True))
            for page_number in range(1, len(pdf) + 1)
        )
    else:
        reader = PdfReader(str(path))
        extracted_pages: list[tuple[int, str | None]] = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text(extraction_mode="layout")
            except TypeError:
                text = page.extract_text()
            extracted_pages.append((page_number, text))
        pages = extracted_pages
    for page_number, raw_text in pages:
        text = (raw_text or "").strip()
        if not text:
            continue
        # A page is a useful unit for financial tables because it preserves
        # the page metadata required by the evaluator.

        CHUNK_SIZE = 5000
        OVERLAP = 200
        for start in range(0, len(text), CHUNK_SIZE):
            piece = text[start : start + (CHUNK_SIZE + OVERLAP)].strip()
            if piece:
                chunks.append(
                    PageChunk(doc_id, page_number, metadata["organization"], piece)
                )
    if fitz is not None:
        pdf.close()
    return chunks


@_traceable(name="build_report_index")
def build_index(force: bool = False) -> EmbeddingIndex:
    """Parse all reports and create or load the local vector index."""
    if (
        not force
        and (INDEX_DIR / "chunks.json").exists()
        and (INDEX_DIR / "vectors.npy").exists()
        and INDEX_CONFIG.exists()
        and json.loads(INDEX_CONFIG.read_text(encoding="utf-8")).get("version")
        == INDEX_VERSION
    ):
        _status("Loading existing index from index/.")
        return EmbeddingIndex.load(INDEX_DIR)
    started = time.monotonic()
    _status("Starting PDF parsing and index build.")
    embedder = CourseEmbeddings(
        api_key=os.getenv("EMB_KEY", ""),
        url=os.getenv("EMB_URL", "https://embeddings.parsers360.ru:10443"),
    )
    metadata = _read_document_metadata()
    chunks: list[PageChunk] = []
    for document_number, (doc_id, item) in enumerate(metadata.items(), start=1):
        _status(
            f"Parsing report {document_number}/{len(metadata)}: {doc_id} "
            f"({item['path']})."
        )
        chunks.extend(_chunks_from_pdf(doc_id, item))
        _status(f"Collected {len(chunks)} chunks so far.")
    if not chunks:
        raise ValueError("No text was extracted from the reports.")
    index = EmbeddingIndex.build(chunks, embedder)
    index.save(INDEX_DIR)
    _status(
        f"Index saved: {len(index.chunks)} chunks in "
        f"{(time.monotonic() - started) / 60:.1f} min."
    )
    return index


class TextModel(Protocol):
    def generate(self, question: str, context: Sequence[SearchResult]) -> str: ...


class RateLimiter:
    def __init__(self, requests_per_minute: int, burst: int = 1) -> None:
        self.rate = max(1, requests_per_minute) / 60.0
        self.interval = 1.0 / self.rate
        self.capacity = float(max(1, burst))
        self._tokens = self.capacity
        self._updated_at = time.monotonic()
        self._lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self.capacity,
                    self._tokens + (now - self._updated_at) * self.rate,
                )
                self._updated_at = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                delay = (1 - self._tokens) / self.rate
            time.sleep(delay)


def get_llm_limiter() -> RateLimiter:
    """Return the process-wide limiter shared by answers and evaluation."""
    global _LLM_LIMITER
    if _LLM_LIMITER is None:
        _LLM_LIMITER = RateLimiter(int(os.getenv("LLM_RPM", "10")))
    return _LLM_LIMITER


def _retry_delay(response: requests.Response, fallback: float) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return max(float(retry_after), 1.0)
        except ValueError:
            pass
    return max(fallback, 1.0)


class GatewayModel:
    """Call either the course Anthropic gateway or an OpenAI-compatible gateway."""

    def __init__(self) -> None:
        self.provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.base_url = os.getenv("LLM_BASE_URL", "").rstrip("/")
        self.model = os.getenv("LLM_MODEL", "deepseek-flash")
        self.auth_scheme = os.getenv("LLM_AUTH_SCHEME", "x-api-key").lower()
        self.limiter = get_llm_limiter()

    def generate(self, question: str, context: Sequence[SearchResult]) -> str:
        if not self.api_key:
            raise RuntimeError(
                "LLM_API_KEY is not set. Copy .env.example to .env and add the key."
            )
        return self.complete(_answer_prompt(question, context))

    def complete(self, prompt: str, system: str | None = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        if system is not None:
            messages.insert(0, {"role": "system", "content": system})
        if self.provider == "anthropic":
            return self._anthropic(messages)
        return self._openai(messages)

    def _anthropic(self, messages: list[dict[str, str]]) -> str:
        headers = {"anthropic-version": "2023-06-01"}
        if self.auth_scheme == "authorization":
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            headers["x-api-key"] = self.api_key
        while True:
            self.limiter.wait()
            response = requests.post(
                f"{self.base_url}/v1/messages",
                headers=headers,
                json={
                    "model": self.model,
                    "max_tokens": 1200,
                    "temperature": 0,
                    "messages": messages,
                },
                timeout=120,
            )
            if response.status_code == 429:
                _status("LLM rate limit (429); waiting before retry.")
                time.sleep(_retry_delay(response, self.limiter.interval))
                continue
            response.raise_for_status()
            break
        content = response.json()["content"]
        return "".join(
            part["text"] for part in content if part.get("type") == "text"
        ).strip()

    def _openai(self, messages: list[dict[str, str]]) -> str:
        while True:
            self.limiter.wait()
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "temperature": 0,
                    "messages": messages,
                },
                timeout=120,
            )
            if response.status_code == 429:
                _status("LLM rate limit (429); waiting before retry.")
                time.sleep(_retry_delay(response, self.limiter.interval))
                continue
            response.raise_for_status()
            break
        return str(response.json()["choices"][0]["message"]["content"]).strip()


def _answer_prompt(question: str, results: Sequence[SearchResult]) -> str:
    excerpts = "\n\n".join(
        f"[{result.chunk.doc_id}, стр. {result.chunk.page}]\n{result.chunk.text}"
        for result in results
    )
    return (
        "Ты отвечаешь на вопрос по финансовой отчетности. Используй только приведенные "
        "фрагменты. Если данных недостаточно, честно скажи об этом. Сохраняй числа, "
        "знаки, периоды и единицы измерения. Ответь кратко на русском языке.\n\n"
        f"Вопрос: {question}\n\nФрагменты:\n{excerpts}"
    )


class RAG:
    """Retrieve report pages and generate an answer grounded in those pages."""

    def __init__(self, force_rebuild: bool = False) -> None:
        load_dotenv(ROOT / ".env")
        self.embedder = CourseEmbeddings(
            api_key=os.getenv("EMB_KEY", ""),
            url=os.getenv("EMB_URL", "https://embeddings.parsers360.ru:10443"),
        )
        self.index = build_index(force_rebuild)
        self.model: TextModel = GatewayModel()

    @_traceable(name="financial_report_rag")
    def rag(self, question: str) -> dict[str, Any]:
        _status("Retrieving relevant report chunks.")
        results = self.index.search(question, self.embedder, top_k=8)
        if not results:
            raise ValueError("The report index contains no searchable text.")
        _status("Generating an answer from the retrieved context.")
        answer = self.model.generate(question, results)
        _status("Answer generated.")
        pages: dict[str, set[int]] = {}
        for result in results:
            pages.setdefault(result.chunk.doc_id, set()).add(result.chunk.page)
        context = [
            {"doc_id": doc_id, "pages": sorted(document_pages)}
            for doc_id, document_pages in pages.items()
        ]
        return {"answer": answer, "context": context}


def main() -> None:
    """Build the index or answer one question from the command line."""
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="parse PDFs and build the index")
    build_parser.add_argument("--force", action="store_true")
    ask_parser = subparsers.add_parser("ask", help="answer one question")
    ask_parser.add_argument("question")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    if args.command == "build":
        index = build_index(force=args.force)
        print(f"Indexed {len(index.chunks)} chunks with CourseEmbeddings.")
        return
    result = RAG().rag(args.question)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
