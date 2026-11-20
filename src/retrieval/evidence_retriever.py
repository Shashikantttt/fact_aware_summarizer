"""
Evidence Retrieval Engine.
Matches generated summary sentences (claims) to their most relevant source sentences
using semantic sentence embeddings (SentenceTransformers) with a mathematical TF-IDF fallback.
"""
import math
from collections import Counter
from typing import List, Dict, Any, Optional
from src.preprocessing.tokenizer import segment_sentences, tokenize_words, remove_stopwords


class EvidenceRetriever:
    """
    Retrieves Top-K supporting source sentences for each summary claim.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        top_k: int = 2,
        use_fallback_if_unavailable: bool = True,
        mock_mode: bool = False
    ):
        """
        Args:
            model_name: Pretrained SentenceTransformer model identifier.
            device: 'cuda', 'cpu', or None.
            top_k: Number of evidence sentences to retrieve per summary claim.
            use_fallback_if_unavailable: If True, falls back to mathematical TF-IDF vector similarity
                                         when SentenceTransformers is not installed or offline.
            mock_mode: If True, skips neural loading and uses mathematical TF-IDF retrieval.
        """
        self.model_name = model_name
        self.top_k = top_k
        self.device = device
        self.use_fallback = use_fallback_if_unavailable
        self.mock_mode = mock_mode
        self.embedder = None
        self.is_neural = False

        if not self.mock_mode:
            self._load_embedder()

    def _load_embedder(self):
        """Attempts to load SentenceTransformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading sentence embedder ({self.model_name})...")
            self.embedder = SentenceTransformer(self.model_name, device=self.device)
            self.is_neural = True
        except Exception as e:
            print(f"[EvidenceRetriever] SentenceTransformers not loaded ({e}). Using mathematical TF-IDF retriever.")
            self.embedder = None
            self.is_neural = False

    def _compute_tfidf_vectors(self, all_sentences: List[str]):
        """Mathematical fallback: creates sparse TF-IDF vectors for sentences."""
        tokenized = [remove_stopwords(tokenize_words(s, lowercase=True)) for s in all_sentences]
        df = Counter()
        for tokens in tokenized:
            for t in set(tokens):
                df[t] += 1
        n = len(all_sentences)
        idf = {t: math.log((n + 1.0) / (c + 1.0)) + 1.0 for t, c in df.items()}

        vectors = []
        for tokens in tokenized:
            tf = Counter(tokens)
            vec = {t: count * idf.get(t, 1.0) for t, count in tf.items()}
            vectors.append(vec)
        return vectors

    def _sparse_cosine(self, v1: Dict[str, float], v2: Dict[str, float]) -> float:
        """Cosine similarity for sparse term vectors."""
        common = set(v1.keys()).intersection(set(v2.keys()))
        if not common:
            return 0.0
        dot = sum(v1[k] * v2[k] for k in common)
        n1 = math.sqrt(sum(val ** 2 for val in v1.values()))
        n2 = math.sqrt(sum(val ** 2 for val in v2.values()))
        return dot / (n1 * n2) if (n1 > 0 and n2 > 0) else 0.0

    def retrieve_evidence_for_claims(
        self,
        source_text: str,
        summary_sentences: List[str],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves Top-K source sentences for each sentence in the summary.
        
        Args:
            source_text: Raw or cleaned original document text.
            summary_sentences: List of individual summary claim sentences.
            top_k: Override default top_k.
            
        Returns:
            List of dicts per claim, containing:
                - claim: The summary sentence
                - retrieved_evidence: List of {source_index, sentence, similarity_score}
        """
        k = top_k or self.top_k
        source_sentences = segment_sentences(source_text)

        if not source_sentences or not summary_sentences:
            return []

        # Neural path via Sentence Transformers
        if self.is_neural and self.embedder is not None:
            import numpy as np

            source_embeddings = self.embedder.encode(source_sentences, convert_to_numpy=True, normalize_embeddings=True)
            claim_embeddings = self.embedder.encode(summary_sentences, convert_to_numpy=True, normalize_embeddings=True)

            # Cosine similarity matrix: (num_claims, num_source)
            sim_matrix = np.dot(claim_embeddings, source_embeddings.T)

            results = []
            for claim_idx, claim in enumerate(summary_sentences):
                scores = sim_matrix[claim_idx]
                top_indices = np.argsort(scores)[::-1][:k]

                evidence = []
                for idx in top_indices:
                    evidence.append({
                        "source_index": int(idx),
                        "sentence": source_sentences[idx],
                        "similarity_score": float(scores[idx])
                    })

                results.append({
                    "claim_index": claim_idx,
                    "claim": claim,
                    "retrieved_evidence": evidence,
                    "method": "sentence-transformers"
                })

            return results

        # Mathematical TF-IDF fallback path
        all_sents = source_sentences + summary_sentences
        vectors = self._compute_tfidf_vectors(all_sents)
        src_vectors = vectors[:len(source_sentences)]
        claim_vectors = vectors[len(source_sentences):]

        results = []
        for claim_idx, (claim, cv) in enumerate(zip(summary_sentences, claim_vectors)):
            scored = []
            for src_idx, sv in enumerate(src_vectors):
                sim = self._sparse_cosine(cv, sv)
                scored.append((src_idx, sim))

            scored.sort(key=lambda x: x[1], reverse=True)
            top_matches = scored[:k]

            evidence = [
                {
                    "source_index": idx,
                    "sentence": source_sentences[idx],
                    "similarity_score": round(score, 4)
                }
                for idx, score in top_matches
            ]

            results.append({
                "claim_index": claim_idx,
                "claim": claim,
                "retrieved_evidence": evidence,
                "method": "tfidf-cosine"
            })

        return results
