"""
Text cleaning and normalization module.
Handles unicode characters, whitespace, hyphenation across lines, and academic citations.
"""
import re
import unicodedata


def clean_text(text: str, remove_academic_citations: bool = False) -> str:
    """
    Cleans and normalizes raw text for both news and research papers.
    
    Args:
        text: Raw input string.
        remove_academic_citations: If True, strips bracketed ([1], [1, 2]) and parenthetical 
                                   author-year citations (Smith et al., 2020).
    Returns:
        Cleaned and normalized text string.
    """
    if not text or not isinstance(text, str):
        return ""

    # Normalize unicode characters (e.g., NFKC normalization)
    text = unicodedata.normalize("NFKC", text)

    # Standardize quotation marks and dashes
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("—", " - ").replace("–", " - ")

    # Remove academic citations if requested (helpful when processing research papers)
    if remove_academic_citations:
        text = remove_citations(text)

    # Rejoin words broken by a hyphen at a line break (hyphen followed by newline and a lowercase letter)
    # (e.g., "sta-\nkeholder" -> "stakeholder", "pro-\ncess" -> "process")
    # while leaving real hyphenated terms like "state-of-the-art" intact
    text = re.sub(r'(\b[A-Za-z]+)-\s*[\r\n]+\s*([a-z]+)', r'\1\2', text)

    # Replace remaining newlines and tabs with spaces (do NOT inject fake periods)
    text = re.sub(r'[\r\n\t]+', ' ', text)

    # Strip repeated whitespace
    text = re.sub(r'\s{2,}', ' ', text)

    return text.strip()


def remove_citations(text: str) -> str:
    """
    Removes bracketed numeric references [1], [2, 3], [4-7] and
    parenthetical author-year citations like (Author et al., 2021).
    """
    # Bracketed numbers: [1], [12, 15], [3-5]
    text = re.sub(r'\[\s*\d+(?:\s*[-–,]\s*\d+)*\s*\]', '', text)
    
    # Author-year citations: (Smith, 2020), (Smith et al., 2020), (Smith & Jones, 2019)
    author_year_pattern = r'\([A-Z][a-zA-Z\s]+(?:et al\.?|& [A-Z][a-zA-Z]+)?,\s*\d{4}[a-z]?\)'
    text = re.sub(author_year_pattern, '', text)
    
    return text
