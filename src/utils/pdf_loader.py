"""
PDF Text Extraction Utility using PyMuPDF (fitz) or pypdf fallback.
Extracts clean, normalized text from academic research papers and documents.
"""
import io
import re
from typing import Dict, Any, List
from src.preprocessing.cleaner import clean_text
from src.preprocessing.academic_cleaner import clean_academic_paper


def extract_text_from_pdf(pdf_source, remove_citations_flag: bool = True) -> Dict[str, Any]:
    """
    Extracts text from a PDF file path or file-like buffer (e.g. Streamlit UploadedFile),
    prunes academic noise (running headers, footers, figure captions, references),
    and isolates the author Abstract as a gold-standard reference.
    
    Args:
        pdf_source: File path (str) or bytes buffer (BytesIO).
        remove_citations_flag: If True, strips academic bracket/author-year citations.
        
    Returns:
        Dict containing:
            - full_text: Cleaned narrative text (Abstract + Body, without references)
            - abstract: Extracted author-written Abstract (or None)
            - body_text: Body text without abstract or references
            - has_references_removed: Boolean
            - pages: List of text per page
            - page_count: Total pages
            - original_word_count: Pre-cleaning word count
            - cleaned_word_count: Post-cleaning word count
            - status: 'success' or 'error'
    """
    pages_text: List[str] = []
    
    # Try PyMuPDF (fitz) first (recommended in project PDF)
    try:
        import fitz  # PyMuPDF
        
        if isinstance(pdf_source, (str, bytes)):
            doc = fitz.open(pdf_source) if isinstance(pdf_source, str) else fitz.open(stream=pdf_source, filetype="pdf")
        else:
            # BytesIO or file-like stream
            if hasattr(pdf_source, "seek"):
                pdf_source.seek(0)
            data = pdf_source.read() if hasattr(pdf_source, "read") else pdf_source
            doc = fitz.open(stream=data, filetype="pdf")

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            # Filter common standalone page numbers
            cleaned_page = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
            pages_text.append(cleaned_page.strip())

        doc.close()

    except ImportError:
        # Fallback to pypdf if fitz is not installed
        try:
            import pypdf
            reader = pypdf.PdfReader(pdf_source)
            for page in reader.pages:
                pages_text.append(page.extract_text() or "")
        except Exception as e:
            return {
                "full_text": "",
                "abstract": None,
                "body_text": "",
                "pages": [],
                "page_count": 0,
                "status": f"Error: PDF libraries (pymupdf or pypdf) not available ({e})"
            }
    except Exception as e:
        return {
            "full_text": "",
            "abstract": None,
            "body_text": "",
            "pages": [],
            "page_count": 0,
            "status": f"Error reading PDF: {e}"
        }

    raw_combined = "\n\n".join(pages_text)
    
    # Apply specialized academic paper cleaning
    paper_data = clean_academic_paper(raw_combined)
    cleaned_full = paper_data["clean_full_text"]
    if not cleaned_full:
        cleaned_full = clean_text(raw_combined, remove_academic_citations=remove_citations_flag)

    return {
        "full_text": cleaned_full,
        "abstract": paper_data.get("abstract"),
        "body_text": paper_data.get("body_text", ""),
        "has_references_removed": paper_data.get("has_references_removed", False),
        "original_word_count": paper_data.get("original_word_count", len(raw_combined.split())),
        "cleaned_word_count": paper_data.get("cleaned_word_count", len(cleaned_full.split())),
        "pages": pages_text,
        "page_count": len(pages_text),
        "status": "success"
    }
