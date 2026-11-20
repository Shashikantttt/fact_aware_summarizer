"""
Academic Research Paper Cleaner.
Specialized text extractor and cleaner for scientific PDFs, preprints, and journals:
1. Strips journal running headers, footers, page numbers, and copyright/license lines.
2. Strips author blocks, affiliations, addresses, emails, and ORCID identifiers.
3. Automatically extracts and isolates the Abstract (for use as a gold-standard reference summary).
4. Drops everything after 'REFERENCES' / 'BIBLIOGRAPHY' / 'Works Cited'.
5. Removes figure/table captions, DOIs, URLs, and citation fragments.
6. Intelligently unwraps PDF line breaks without corrupting sentence boundaries.
"""
import re
from typing import Dict, Any, Optional, Tuple, List


# Patterns for academic headers, footers, and copyright
RUNNING_HEADER_PATTERNS = [
    r'(?i)\d{4}\s+Journal of\s+.*?(?:\n|$)',
    r'(?i)Volume\s+\d+,\s+No\.\s+\d+.*?(?:\n|$)',
    r'(?i)Proc\.\s+\d+.*?(?:\n|$)',
    r'(?i)IEEE\s+Transactions.*?(?:\n|$)',
    r'(?i)ACM\s+Conference.*?(?:\n|$)',
    r'(?i)arXiv:\d+\.\d+.*?(?:\n|$)',
    r'(?i)©\s*\d{4}\s+.*?(?:Creative Commons|All rights reserved|www\.\S+).*?(?:\n|$)',
    r'(?i)www\.[a-z0-9\-]+(?:\.[a-z0-9\-]+)+',
    r'^\s*\d+\s*$'  # standalone page numbers
]

# Affiliation and author block indicators
AFFILIATION_PATTERNS = [
    r'\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b',  # ORCID iD
    r'(?i)\b(?:Department|Dept\.|Program|Institute|University|School|College|Laboratory|Collaboratory)\b.*?(?:\n|$)',
    r'(?i)\b[A-Za-z\s]+,\s*(?:[A-Z]{2}|USA|Canada|UK|United States|Germany|France|India|China)\b',
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # emails
]

# Section headers indicating the start of bibliography/references
REFERENCE_HEADINGS = [
    r'\n\s*(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography|Works Cited)\s*\n',
    r'\n\s*\[1\]\s+[A-Z].*',
]


def unwrap_pdf_lines(text: str) -> str:
    """
    Unwraps PDF lines where text was split across column margins:
    - Joins hyphenated words across line breaks (e.g., 'trans-\\nformer' -> 'transformer')
    - Joins wrapped lines into flowing paragraphs without inserting spurious periods.
    """
    if not text:
        return ""

    # Rejoin words broken by a hyphen at a line break (hyphen followed by newline and a lowercase letter)
    # (e.g., "sta-\nkeholder" -> "stakeholder", "pro-\ncess" -> "process")
    text = re.sub(r'(\b[A-Za-z]+)-\s*[\r\n]+\s*([a-z]+)', r'\1\2', text)
    
    # If a line ends with a letter/comma and the next line begins with a lowercase letter,
    # or capital letter mid-sentence, join with a space.
    lines = text.split('\n')
    unwrapped_lines = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            unwrapped_lines.append("")
            continue
            
        if unwrapped_lines and unwrapped_lines[-1] != "":
            prev = unwrapped_lines[-1]
            # If previous line did not end with sentence-closing punctuation (. ! ? :)
            if not re.search(r'[.!?:]\s*$', prev):
                # Join with space
                unwrapped_lines[-1] = prev + " " + stripped
                continue
                
        unwrapped_lines.append(stripped)

    return "\n".join(unwrapped_lines)


def strip_headers_and_footers(text: str) -> str:
    """Removes journal banners, copyright notices, and page numbers."""
    cleaned = text
    for pat in RUNNING_HEADER_PATTERNS:
        cleaned = re.sub(pat, ' ', cleaned)
    return cleaned


def strip_captions_and_artifacts(text: str) -> str:
    """
    Removes figure captions (e.g. 'Fig. 1. RC3TF game map...'),
    table captions, and raw URLs/DOIs.
    """
    # Remove Figure and Table captions
    text = re.sub(r'(?m)^\s*(?:Fig(?:ure)?\.?|Table)\s*\d+[\.:].*?(?:\n|$)', ' ', text)
    text = re.sub(r'(?i)\bFig(?:ure)?\s*\d+\b\s+shows\s+image.*?(?:\n|$)', ' ', text)

    # Remove DOIs and URLs
    text = re.sub(r'(?i)\b(?:doi:\s*|https?://(?:dx\.)?doi\.org/)\S+', ' ', text)
    text = re.sub(r'https?://\S+', ' ', text)

    # Remove Roman numeral section titles standing on their own line (e.g. 'I. INTRODUCTION')
    text = re.sub(r'(?m)^\s*[IVXLCDM]+\.\s+[A-Z\s]{3,}.*?(?:\n|$)', ' ', text)
    text = re.sub(r'(?m)^\s*[A-Z]\.\s+["\'“‘]?[A-Z][a-z].*?(?:\n|$)', ' ', text)

    # Remove standalone keywords line
    text = re.sub(r'(?i)Keywords[—\-:\s].*?(?:\n|$)', ' ', text)

    return text


def strip_references(text: str) -> Tuple[str, bool]:
    """
    Truncates the paper text at the beginning of the 'REFERENCES' or 'BIBLIOGRAPHY' section.
    Returns (cleaned_text, found_references_flag).
    """
    # Search for reference section start
    match = re.search(r'(?m)^\s*(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography|Works Cited)\s*$', text)
    if match:
        cut_pos = match.start()
        return text[:cut_pos].strip(), True

    # Fallback: check for [1] Author pattern near the end of the text
    bracket_ref = re.search(r'(?m)^\s*\[1\]\s+[A-Z][a-z]+', text)
    if bracket_ref and bracket_ref.start() > len(text) * 0.4:
        return text[:bracket_ref.start()].strip(), True

    return text, False


def extract_abstract(text: str) -> Tuple[Optional[str], str]:
    """
    Extracts the paper Abstract cleanly and separates it from the remaining body.
    Returns (abstract_text, remaining_text).
    """
    # Search for Abstract heading
    abs_match = re.search(r'(?i)\bAbstract\s*[—\-:\.]\s*', text)
    if not abs_match:
        # Try standalone Abstract line
        abs_match = re.search(r'(?m)^\s*ABSTRACT\s*$', text)

    if not abs_match:
        return None, text

    start_idx = abs_match.end()
    rest = text[start_idx:]

    # The abstract typically ends at 'Keywords', 'Index Terms', or section 'I. INTRODUCTION'
    end_match = re.search(r'(?i)(?:\bKeywords\s*[—\-:\.]|\bIndex Terms\s*[—\-:\.]|^\s*I\.\s+INTRODUCTION|\n\s*1\.\s+Introduction)', rest)
    if end_match:
        abstract = rest[:end_match.start()].strip()
        body = rest[end_match.start():].strip()
    else:
        # Take first 3-5 sentences or up to 250 words as abstract
        lines = rest.split('\n\n')
        abstract = lines[0].strip() if lines else rest[:1200].strip()
        body = rest[len(abstract):].strip()

    # Clean the abstract
    abstract = re.sub(r'\s+', ' ', abstract).strip()
    return abstract, body


def strip_author_block(pre_abstract_text: str) -> str:
    """
    Strips author names, ORCIDs, and university affiliations that appear before the Abstract.
    """
    cleaned = pre_abstract_text
    for pat in AFFILIATION_PATTERNS:
        cleaned = re.sub(pat, ' ', cleaned)
    
    # Remove lines with high density of names/affiliations
    lines = cleaned.split('\n')
    filtered = []
    for line in lines:
        words = line.strip().split()
        # Skip very short lines or affiliation-like chunks
        if len(words) < 2:
            continue
        filtered.append(line.strip())

    return " ".join(filtered)



def extract_sections(body_text: str) -> List[Dict[str, Any]]:
    """
    Splits paper body into structured sections based on Roman numeral or numeric headings
    (e.g., 'I. INTRODUCTION', 'II. BACKGROUND', 'III. INTRODUCING THE RC3TF', 'IV. DISCUSSION').
    If fewer than 2 headings are found, partitions the text into 4 chronological narrative phases.
    """
    heading_pattern = r'(?m)^\s*([IVXLCDM]+\.|\d+\.)\s+([A-Z\s\(\)“”—\-]{3,}.*?)(?:\n|$)'
    matches = list(re.finditer(heading_pattern, body_text))

    sections = []
    if len(matches) >= 2:
        for idx, match in enumerate(matches):
            h_num = match.group(1).strip()
            h_title = match.group(2).strip()
            full_heading = f"{h_num} {h_title}"
            
            start_pos = match.end()
            end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(body_text)
            sec_content = body_text[start_pos:end_pos].strip()
            
            # Clean section artifacts
            sec_clean = strip_captions_and_artifacts(sec_content)
            sec_clean = re.sub(r'\[\s*\d+(?:\s*[-–,]\s*\d+)*\s*\]', '', sec_clean)
            sec_clean = re.sub(r'\s{2,}', ' ', sec_clean).strip()
            
            if len(sec_clean.split()) >= 15:
                sections.append({
                    "title": full_heading,
                    "text": sec_clean
                })

    # If no explicit headings, partition into 4 narrative phases
    if len(sections) < 2:
        words = body_text.split()
        q_len = max(1, len(words) // 4)
        phase_names = [
            "Introduction & Motivation",
            "Background & Context",
            "System Architecture & Implementation",
            "Discussion & Conclusions"
        ]
        sections = []
        for i in range(4):
            part_words = words[i * q_len : (i + 1) * q_len if i < 3 else len(words)]
            part_text = " ".join(part_words).strip()
            if part_text:
                sections.append({
                    "title": phase_names[i],
                    "text": part_text
                })

    return sections


def clean_academic_paper(raw_text: str) -> Dict[str, Any]:
    """
    Full pipeline to clean an academic research paper:
    1. Truncates bibliography & references.
    2. Strips headers, footers, copyright lines, and page numbers.
    3. Isolates Abstract (gold-standard reference summary).
    4. Extracts structured narrative sections (Intro, System, Discussion, Conclusion).
    5. Strips author affiliation block.
    6. Strips figure captions, table labels, and Roman numeral fragments.
    7. Normalizes whitespace and unhyphenates line splits.
    
    Returns:
        Dict with:
            - 'abstract': Extracted abstract text (useful for ROUGE benchmark)
            - 'body_text': Cleaned narrative text of the paper (EXCLUDING Abstract)
            - 'sections': List of structured section dicts (title, text)
            - 'clean_full_text': Combined abstract + body text
            - 'has_references_removed': True if references section was pruned
            - 'original_word_count': Words before cleaning
            - 'cleaned_word_count': Words after removing references and noise
    """
    if not raw_text or not isinstance(raw_text, str):
        return {
            "abstract": None,
            "body_text": "",
            "sections": [],
            "clean_full_text": "",
            "has_references_removed": False,
            "original_word_count": 0,
            "cleaned_word_count": 0
        }

    orig_words = len(raw_text.split())

    # Step 1: Strip everything from References onwards
    text_no_refs, refs_removed = strip_references(raw_text)

    # Step 2: Unwrap PDF line wraps and hyphenation
    unwrapped = unwrap_pdf_lines(text_no_refs)

    # Step 3: Strip journal banners, footers, and page numbers
    no_headers = strip_headers_and_footers(unwrapped)

    # Step 4: Extract the Abstract and body
    abstract, body = extract_abstract(no_headers)

    raw_body = body if body else no_headers

    # Step 5: Extract structured sections before stripping heading tags
    sections = extract_sections(raw_body)

    # Step 6: Clean the body text (remove captions, DOIs, URLs, section titles)
    cleaned_body = strip_captions_and_artifacts(raw_body)

    # Step 7: Remove bracket citations [1], [2, 3] from body
    cleaned_body = re.sub(r'\[\s*\d+(?:\s*[-–,]\s*\d+)*\s*\]', '', cleaned_body)
    
    # Normalize extra whitespace
    cleaned_body = re.sub(r'\s{2,}', ' ', cleaned_body).strip()
    if abstract:
        abstract = re.sub(r'\[\s*\d+(?:\s*[-–,]\s*\d+)*\s*\]', '', abstract)
        abstract = re.sub(r'\s{2,}', ' ', abstract).strip()

    full_clean = f"{abstract} {cleaned_body}".strip() if abstract else cleaned_body

    return {
        "abstract": abstract,
        "body_text": cleaned_body,
        "sections": sections,
        "clean_full_text": full_clean,
        "has_references_removed": refs_removed,
        "original_word_count": orig_words,
        "cleaned_word_count": len(cleaned_body.split())
    }
