"""
TF-IDF Extractive Summarizer.
Mathematical sentence scoring based on Term Frequency - Inverse Document Frequency.
Pure Python implementation (requires zero external model weights).
"""
import math
from collections import Counter
from typing import List, Dict, Any, Tuple
from src.preprocessing.cleaner import clean_text
from src.preprocessing.tokenizer import segment_sentences, tokenize_words, remove_stopwords


class TFIDFSummarizer:
    """
    Extractive summarization using Term Frequency - Inverse Document Frequency (TF-IDF).
    Treats each sentence in the document as an individual mini-document to compute IDF.
    """

    def __init__(self, position_bias: bool = True, length_penalty_power: float = 0.8):
        """
        Args:
            position_bias: If True, gives a small bonus to earlier sentences in the document
                           (well-known lead bias in journalism and scientific introductions).
            length_penalty_power: Exponent for length normalization (default 0.8 avoids favoring
                                  either excessively long or very short sentences).
        """
        self.position_bias = position_bias
        self.length_penalty_power = length_penalty_power

    def _compute_frequencies(self, sentences: List[List[str]]) -> Tuple[List[Counter], Counter, int]:
        """
        Computes term frequencies (TF) per sentence and document frequencies (DF) across sentences.
        """
        num_sentences = len(sentences)
        sentence_tfs = [Counter(tokens) for tokens in sentences]
        
        df = Counter()
        for tf in sentence_tfs:
            for term in tf.keys():
                df[term] += 1
                
        return sentence_tfs, df, num_sentences

    def _compute_idf(self, df: Counter, num_sentences: int) -> Dict[str, float]:
        """
        Computes smoothed IDF: ln((N + 1) / (DF + 1)) + 1
        """
        idf = {}
        for term, freq in df.items():
            idf[term] = math.log((num_sentences + 1.0) / (freq + 1.0)) + 1.0
        return idf

    def summarize(self, text: str, num_sentences: int = 3) -> Dict[str, Any]:
        """
        Extracts the most salient sentences using TF-IDF scoring.
        
        Args:
            text: Raw input text.
            num_sentences: Number of sentences to extract for the summary.
            
        Returns:
            Dict containing:
                - summary: Joined summary string
                - selected_sentences: List of extracted sentence strings in original order
                - sentence_scores: List of dicts with (index, sentence, score)
                - total_sentences: Total sentence count in source
        """
        cleaned = clean_text(text)
        raw_sentences = segment_sentences(cleaned)

        if not raw_sentences:
            return {
                "summary": "",
                "selected_sentences": [],
                "sentence_scores": [],
                "total_sentences": 0
            }

        # If document is already shorter than or equal to desired summary length
        if len(raw_sentences) <= num_sentences:
            return {
                "summary": " ".join(raw_sentences),
                "selected_sentences": raw_sentences,
                "sentence_scores": [{"index": i, "sentence": s, "score": 1.0} for i, s in enumerate(raw_sentences)],
                "total_sentences": len(raw_sentences)
            }

        # Tokenize and filter stopwords for scoring
        tokenized_sentences = [
            remove_stopwords(tokenize_words(s, lowercase=True))
            for s in raw_sentences
        ]

        sentence_tfs, df, n = self._compute_frequencies(tokenized_sentences)
        idf_dict = self._compute_idf(df, n)

        scored_sentences = []
        for idx, (raw_sent, tf) in enumerate(zip(raw_sentences, sentence_tfs)):
            token_count = len(tokenized_sentences[idx])
            if token_count == 0:
                score = 0.0
            else:
                # Sum of TF-IDF scores for all content words in the sentence
                tfidf_sum = sum((count / token_count) * idf_dict.get(term, 0.0) for term, count in tf.items())
                
                # Length normalization
                length_norm = max(1, token_count) ** self.length_penalty_power
                score = tfidf_sum / length_norm

                # Optional lead bias: early sentences often contain the main thesis
                if self.position_bias:
                    position_multiplier = 1.0 + (1.0 / math.sqrt(idx + 1)) * 0.15
                    score *= position_multiplier

            scored_sentences.append({
                "index": idx,
                "sentence": raw_sent,
                "score": float(score)
            })

        # Filter candidates by length to avoid isolated context-free snippets
        valid_candidates = [s for s in scored_sentences if len(s["sentence"].split()) >= 10]
        if len(valid_candidates) < num_sentences:
            valid_candidates = scored_sentences

        sorted_cands = sorted(valid_candidates, key=lambda x: x["score"], reverse=True)
        chosen = []
        chosen_word_sets = []
        
        for c in sorted_cands:
            c_words = set(remove_stopwords(tokenize_words(c["sentence"], lowercase=True)))
            # Check Jaccard overlap with already chosen sentences
            is_redundant = False
            for prev_words in chosen_word_sets:
                if c_words and prev_words:
                    overlap = len(c_words.intersection(prev_words)) / len(c_words.union(prev_words))
                    if overlap >= 0.38:
                        is_redundant = True
                        break
            if not is_redundant:
                chosen.append(c)
                chosen_word_sets.append(c_words)
            if len(chosen) >= num_sentences:
                break

        # Fill remaining slots if strict redundancy penalty exhausted candidates
        if len(chosen) < num_sentences:
            for c in sorted_cands:
                if c not in chosen:
                    chosen.append(c)
                if len(chosen) >= num_sentences:
                    break

        # Re-sort selected sentences by original document appearance order to preserve narrative flow
        selected_in_order = sorted(chosen, key=lambda x: x["index"])
        summary_sentences = [item["sentence"] for item in selected_in_order]

        return {
            "summary": " ".join(summary_sentences),
            "selected_sentences": summary_sentences,
            "sentence_scores": scored_sentences,
            "total_sentences": len(raw_sentences)
        }
