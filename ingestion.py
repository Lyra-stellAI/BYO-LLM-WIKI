import re
from io import BytesIO
from pathlib import Path


PARAGRAPH_BREAK = re.compile(r"\n{2,}")
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _overlap_tail(text: str, overlap: int) -> str:
    if overlap <= 0 or not text:
        return ""
    if len(text) <= overlap:
        return text
    tail = text[-overlap:]
    sp = tail.find(" ")
    if 0 < sp < overlap // 2:
        tail = tail[sp + 1:]
    return tail.strip()


def chunk_text(text: str, *, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Structured overlapping chunker.

    1. Splits on paragraph boundaries (double newlines).
    2. Long paragraphs are split on sentence boundaries.
    3. Pathologically long sentences are hard-split on characters.
    4. Groups pieces up to `chunk_size`, carrying `overlap` chars
       (snapped to a word boundary) from the previous chunk so context
       doesn't get cut mid-thought.
    """
    text = normalize(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    paragraphs = [p.strip() for p in PARAGRAPH_BREAK.split(text) if p.strip()]
    pieces: list[str] = []
    for p in paragraphs:
        if len(p) <= chunk_size:
            pieces.append(p)
            continue
        sentences = [s.strip() for s in SENTENCE_BREAK.split(p) if s.strip()]
        buf = ""
        for s in sentences:
            cand = (buf + " " + s).strip() if buf else s
            if len(cand) <= chunk_size:
                buf = cand
            else:
                if buf:
                    pieces.append(buf)
                if len(s) > chunk_size:
                    for i in range(0, len(s), chunk_size):
                        pieces.append(s[i:i + chunk_size])
                    buf = ""
                else:
                    buf = s
        if buf:
            pieces.append(buf)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        cand = (current + "\n\n" + piece).strip() if current else piece
        if len(cand) <= chunk_size:
            current = cand
        else:
            if current:
                chunks.append(current)
                tail = _overlap_tail(current, overlap)
                current = (tail + "\n\n" + piece).strip() if tail else piece
            else:
                current = piece
    if current:
        chunks.append(current)
    return chunks


def parse_file(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(content)
    if suffix in (".html", ".htm"):
        return _parse_html(content)
    return content.decode("utf-8", errors="replace")


def _parse_pdf(content: bytes) -> str:
    try:
        import pypdf
    except ImportError as e:
        raise RuntimeError("pypdf not installed; cannot parse PDF.") from e
    reader = pypdf.PdfReader(BytesIO(content))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n\n".join(parts)


def _parse_html(content: bytes) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(content, "lxml")
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.body or soup
    return main.get_text(separator="\n", strip=True)
