"""
Evaluation metrics for text summarization:
- ROUGE-1, ROUGE-2, and ROUGE-L (F1, Precision, Recall)
- Compression Ratio
- Factual Consistency Score
- Runtime Execution Profiler
Implemented in pure Python for offline sandbox execution, with automatic fallback to `rouge_score` library.
"""
import time
from collections import Counter
from typing import List, Dict, Any, Tuple
from src.preprocessing.tokenizer import tokenize_words


class Timer:
    """Context manager for measuring runtime execution latency."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


def _get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Generates a list of n-grams from a token sequence."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def compute_rouge_n(candidate_tokens: List[str], reference_tokens: List[str], n: int = 1) -> Dict[str, float]:
    """
    Computes ROUGE-N (Recall, Precision, F1-Score).
    ROUGE-N Recall = (overlapping n-grams) / (total reference n-grams)
    ROUGE-N Precision = (overlapping n-grams) / (total candidate n-grams)
    """
    cand_ngrams = _get_ngrams(candidate_tokens, n)
    ref_ngrams = _get_ngrams(reference_tokens, n)

    if not cand_ngrams or not ref_ngrams:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    cand_counts = Counter(cand_ngrams)
    ref_counts = Counter(ref_ngrams)

    overlap_count = 0
    for ngram, count in cand_counts.items():
        overlap_count += min(count, ref_counts.get(ngram, 0))

    precision = overlap_count / len(cand_ngrams) if cand_ngrams else 0.0
    recall = overlap_count / len(ref_ngrams) if ref_ngrams else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }


def _lcs_length(x: List[str], y: List[str]) -> int:
    """Computes the Longest Common Subsequence (LCS) length between two token lists."""
    m, n = len(x), len(y)
    # Space-optimized dynamic programming table
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = list(curr)

    return curr[n]


def compute_rouge_l(candidate_tokens: List[str], reference_tokens: List[str]) -> Dict[str, float]:
    """
    Computes ROUGE-L based on the Longest Common Subsequence (LCS).
    """
    if not candidate_tokens or not reference_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = _lcs_length(candidate_tokens, reference_tokens)
    m = len(candidate_tokens)
    n = len(reference_tokens)

    precision = lcs / m if m > 0 else 0.0
    recall = lcs / n if n > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }


def compute_summary_metrics(
    candidate_summary: str,
    reference_summary: str,
    source_document: str,
    verification_result: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Computes full evaluation suite for a single document summary:
      - ROUGE-1, ROUGE-2, ROUGE-L (F1)
      - Compression Ratio
      - Factuality Consistency Score
    """
    cand_tokens = tokenize_words(candidate_summary, lowercase=True)
    ref_tokens = tokenize_words(reference_summary, lowercase=True)
    src_tokens = tokenize_words(source_document, lowercase=True)

    r1 = compute_rouge_n(cand_tokens, ref_tokens, n=1)
    r2 = compute_rouge_n(cand_tokens, ref_tokens, n=2)
    rl = compute_rouge_l(cand_tokens, ref_tokens)

    src_len = max(1, len(src_tokens))
    cand_len = len(cand_tokens)
    compression_ratio = round(cand_len / src_len, 4)

    factuality_score = None
    if verification_result:
        factuality_score = verification_result.get("factuality_score")

    return {
        "rouge_1": r1["f1"],
        "rouge_2": r2["f1"],
        "rouge_l": rl["f1"],
        "compression_ratio": compression_ratio,
        "factuality_score": factuality_score,
        "candidate_word_count": cand_len,
        "source_word_count": src_len
    }
