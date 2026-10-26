"""
Ingestion: files → chunks with metadata (document, section, page).

- Markdown is split on headings first (MarkdownHeaderTextSplitter), so every chunk knows its
  section; long sections are split further with overlap (RecursiveCharacterTextSplitter).
- PDFs are read page by page (pypdf), so every chunk knows its page number.
- Each chunk's searchable text starts with a short header ("Doc > Section") — contextual chunk
  headers help both keyword and vector search find the right section.
- Documents are untrusted: chunks containing prompt-injection patterns are quarantined (kept
  out of the index) and reported, instead of reaching the model.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from guardrails import detect_injection_heuristic

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
SUPPORTED = {".md", ".markdown", ".txt", ".pdf"}

_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
_md_headers = MarkdownHeaderTextSplitter([("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=True)


@dataclass
class Chunk:
    id: int
    doc: str                    # file name
    title: str                  # document title (first heading, or file name)
    section: str                # heading path, e.g. "Expenses > Receipts and deadlines"
    page: Optional[int]         # PDF page number (1-based), None for text formats
    text: str                   # the passage as written

    @property
    def label(self) -> str:
        where = f"p. {self.page}" if self.page else self.section
        return f"{self.title} — {where}" if where else self.title

    @property
    def search_text(self) -> str:
        """What gets indexed: a contextual header plus the passage."""
        return f"{self.title} > {self.section}\n{self.text}" if self.section else f"{self.title}\n{self.text}"


@dataclass
class Corpus:
    chunks: List[Chunk] = field(default_factory=list)
    documents: List[str] = field(default_factory=list)
    quarantined: List[dict] = field(default_factory=list)   # chunks withheld as possible injection
    skipped: List[str] = field(default_factory=list)        # unsupported or unreadable files


def _markdown_sections(text: str):
    for doc in _md_headers.split_text(text):
        meta = doc.metadata
        yield meta.get("h1", ""), " > ".join(meta[h] for h in ("h2", "h3") if meta.get(h)), doc.page_content, None


def _pdf_pages(path: Path):
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    first = (reader.pages[0].extract_text() or "").strip().splitlines() if reader.pages else []
    title = first[0].strip() if first else path.stem
    for number, page in enumerate(reader.pages, 1):
        lines = [ln.strip() for ln in (page.extract_text() or "").splitlines() if ln.strip()]
        if number == 1 and lines and lines[0] == title:
            lines = lines[1:]
        if lines:  # first line of each page is treated as its heading
            yield title, lines[0], "\n".join(lines[1:]) or lines[0], number


def _sections(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _pdf_pages(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix in (".md", ".markdown"):
        return _markdown_sections(text)
    return iter([(path.stem, "", text, None)])


def ingest(paths) -> Corpus:
    corpus = Corpus()
    for path in sorted(Path(p) for p in paths):
        if path.suffix.lower() not in SUPPORTED:
            corpus.skipped.append(f"{path.name} (unsupported type)")
            continue
        try:
            sections = list(_sections(path))
        except Exception as exc:  # corrupt PDF, bad encoding, ...
            corpus.skipped.append(f"{path.name} ({type(exc).__name__}: {exc})")
            continue
        title = next((t for t, *_ in sections if t), path.stem.replace("_", " ").title())
        corpus.documents.append(path.name)
        for _, section, body, page in sections:
            for piece in _splitter.split_text(body):
                reason = detect_injection_heuristic(piece)
                if reason:
                    corpus.quarantined.append({"doc": path.name, "section": section, "page": page, "reason": reason})
                    continue
                corpus.chunks.append(Chunk(len(corpus.chunks), path.name, title, section, page, piece.strip()))
    return corpus
