"""
Extracts page-level text from a PDF. Every downstream fact is grounded to a
(document, page) pair so we can always point back to evidence.
"""
import fitz  # PyMuPDF


def extract_pages(pdf_path):
    """Returns a list of dicts: [{"page": 1, "text": "..."}, ...]

    Page numbers are 1-indexed and refer to the PDF's physical page order
    (not any printed page number inside the document, which can differ).
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def guess_title(pdf_path, pages):
    """Cheap heuristic title: first non-trivial line of page 1, else filename."""
    if pages:
        for line in pages[0]["text"].splitlines():
            line = line.strip()
            if len(line) > 8:
                return line[:120]
    return pdf_path.split("/")[-1]


def chunk_pages(pages, pages_per_chunk=3, overlap=0):
    """
    Groups consecutive pages into chunks so we make fewer, larger LLM calls
    (cheaper, faster, and gives the model more context per fact) while still
    tagging each chunk with the exact page range it covers, so extracted
    facts can be localized back to individual pages via the quote text.
    """
    chunks = []
    i = 0
    n = len(pages)
    while i < n:
        group = pages[i : i + pages_per_chunk]
        chunks.append({
            "start_page": group[0]["page"],
            "end_page": group[-1]["page"],
            "text": "\n\n".join(
                f"[PAGE {p['page']}]\n{p['text']}" for p in group
            ),
        })
        i += pages_per_chunk - overlap
    return chunks
