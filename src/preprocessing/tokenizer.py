"""
Tokenization, sentence segmentation, stopword filtering, and document chunking.
Designed to run locally with zero external downloads required.
"""
import re
from typing import List, Dict, Any

# Built-in English stopwords (self-contained, no external download required)
ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than",
    "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
    "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}

# Comprehensive abbreviations for academic papers, legal, news, and conversational text
ACADEMIC_ABBREVIATIONS = (
    r'\b(?:Mr|Mrs|Ms|Dr|Prof|Gen|Gov|Sen|Rep|vs|v|e\.g|i\.e|etc|'
    r'Fig|Figs|figure|figures|Table|Tables|Vol|vols|Volume|Volumes|'
    r'No|nos|pp|p|ed|eds|ref|refs|al|sec|ch|dept|univ|approx|cf|eq|eqs|art|col|suppl)\.'
)
MONTHS_PATTERN = r'\b(?:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.'
ACRONYMS_PATTERN = r'\b(?:[A-Za-z]\.){2,}'
INITIALS_PATTERN = r'\b[A-Za-z]\.'
ROMAN_NUMERALS_PATTERN = r'\b(?:[ivxlcdm]+|[IVXLCDM]+)\.'
ET_AL_PATTERN = r'\bet\s+al\.'


def is_valid_sentence(sent: str) -> bool:
    """
    Filters out noise fragments, isolated citation strings, figure tags,
    and truncated tokens from entering summarization.
    """
    s = sent.strip()
    if not s:
        return False
        
    words = s.split()
    # Require at least 2 words and 5 characters
    if len(words) < 2 or len(s) < 5:
        return False

    # Check alphanumeric density (filter out strings of pure punctuation, DOIs, page numbers)
    alpha_chars = sum(c.isalpha() for c in s)
    if alpha_chars / len(s) < 0.50:
        return False

    # Reject isolated figure / table labels like 'Fig. 5.' or 'Table 2.'
    if re.match(r'(?i)^(?:fig(?:ure)?\.?|table)\s*\d+[\.:]?$', s):
        return False

    # Reject isolated Roman numeral section headers like 'I. INTRODUCTION' or 'IV. DISCUSSION'
    if re.match(r'^[IVXLCDM]+\.\s+[A-Z\s]{3,}$', s):
        return False

    # Reject bibliographical fragments (e.g. starts with [1], DOI, or page range)
    if re.match(r'^(?:\[\d+\]|doi:\s*|https?://|vol\.\s*\d+|pp\.\s*\d+)', s, flags=re.IGNORECASE):
        return False

    return True


def segment_sentences(text: str) -> List[str]:
    """
    Splits text into coherent sentences without breaking on academic abbreviations
    (vol., pp., fig., i.e., e.g., W., al.), decimals (5.25), or Roman numerals (i., ii.).
    Uses NLTK Punkt tokenizer if installed, falling back to a zero-dependency regex engine.
    Filters out junk fragments and noise.
    """
    if not text or not isinstance(text, str):
        return []

    # Try NLTK's Punkt tokenizer if available
    try:
        import nltk
        try:
            raw_sents = nltk.sent_tokenize(text)
            valid_sents = [s.strip() for s in raw_sents if is_valid_sentence(s)]
            if valid_sents:
                return valid_sents
        except Exception:
            pass
    except ImportError:
        pass

    # Built-in robust regex sentence splitter with academic protection
    placeholder = "<DOT>"
    
    def protect_match(match):
        return match.group(0).replace(".", placeholder)
    
    # 1. Protect academic abbreviations (vol., pp., fig., no., etc.)
    protected = re.sub(ACADEMIC_ABBREVIATIONS, protect_match, text, flags=re.IGNORECASE)

    # 2. Protect month abbreviations (Oct. 25, Jan. 1)
    protected = re.sub(MONTHS_PATTERN, protect_match, protected, flags=re.IGNORECASE)

    # 3. Protect multi-dot acronyms (U.S., Ph.D., a.m., p.m.)
    protected = re.sub(ACRONYMS_PATTERN, protect_match, protected)

    # 4. Protect single-letter capitalized initials and middle names (e.g. W. S. Halsted)
    protected = re.sub(INITIALS_PATTERN, protect_match, protected)

    # 5. Protect Roman numerals with dot (e.g. i., ii., III.)
    protected = re.sub(ROMAN_NUMERALS_PATTERN, protect_match, protected)

    # 6. Protect "et al."
    protected = re.sub(ET_AL_PATTERN, protect_match, protected, flags=re.IGNORECASE)

    # 7. Protect numbers with decimals (e.g. 5.25% or 3.14)
    protected = re.sub(r'(\d+)\.(\d+)', r'\1' + placeholder + r'\2', protected)

    # 8. Sentence boundary split: handles periods followed by spaces and capital letters
    sentence_split_pattern = r'(?:(?<=[.!?])|(?<=[.!?]["\'”’]))\s+(?=["\'“‘]?[A-Z0-9])'
    raw_sentences = re.split(sentence_split_pattern, protected)

    sentences = []
    for s in raw_sentences:
        restored = s.replace(placeholder, ".").strip()
        if is_valid_sentence(restored):
            sentences.append(restored)

    return sentences


def tokenize_words(text: str, lowercase: bool = True) -> List[str]:
    """
    Extracts alphanumeric word tokens from text.
    """
    if not text:
        return []
    if lowercase:
        text = text.lower()
    # Match alphanumeric words, including single hyphens in compound words
    tokens = re.findall(r'\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)?\b', text)
    return tokens


def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Filters out English stopwords from a list of tokens.
    """
    return [t for t in tokens if t.lower() not in ENGLISH_STOPWORDS]


def chunk_text_by_tokens(sentences: List[str], max_tokens_per_chunk: int = 500, overlap_sentences: int = 1) -> List[Dict[str, Any]]:
    """
    Splits a list of sentences into chunks that fit within transformer token budgets.
    Maintains sentence boundaries and optional sentence overlap across chunks.
    
    Args:
        sentences: List of source sentences.
        max_tokens_per_chunk: Approximate maximum word/token count per chunk (e.g., 500 for BART's 1024 limit).
        overlap_sentences: Number of trailing sentences from previous chunk to repeat in the next.
    
    Returns:
        List of dicts containing chunk text, sentence indices, and token count.
    """
    if not sentences:
        return []

    chunks = []
    current_chunk_sentences = []
    current_chunk_indices = []
    current_token_count = 0

    i = 0
    while i < len(sentences):
        sent = sentences[i]
        sent_tokens = tokenize_words(sent)
        sent_len = len(sent_tokens)

        if current_token_count + sent_len > max_tokens_per_chunk and current_chunk_sentences:
            chunks.append({
                "chunk_id": len(chunks),
                "text": " ".join(current_chunk_sentences),
                "sentence_indices": list(current_chunk_indices),
                "token_count": current_token_count,
            })
            # Overlap: keep last `overlap_sentences`
            keep_count = min(overlap_sentences, len(current_chunk_sentences))
            current_chunk_sentences = current_chunk_sentences[-keep_count:] if keep_count > 0 else []
            current_chunk_indices = current_chunk_indices[-keep_count:] if keep_count > 0 else []
            current_token_count = sum(len(tokenize_words(s)) for s in current_chunk_sentences)

        current_chunk_sentences.append(sent)
        current_chunk_indices.append(i)
        current_token_count += sent_len
        i += 1

    if current_chunk_sentences:
        chunks.append({
            "chunk_id": len(chunks),
            "text": " ".join(current_chunk_sentences),
            "sentence_indices": list(current_chunk_indices),
            "token_count": current_token_count,
        })

    return chunks
