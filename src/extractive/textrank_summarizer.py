"""
TextRank Extractive Summarizer.
Graph-based sentence ranking using PageRank algorithm with a damping factor.
Implemented from mathematical foundations without requiring external graph libraries.
"""
import math
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
from src.preprocessing.cleaner import clean_text
from src.preprocessing.tokenizer import segment_sentences, tokenize_words, remove_stopwords


class TextRankSummarizer:
    """
    Implements the TextRank graph algorithm for extractive summarization.
    
    1. Nodes: Sentences in the document.
    2. Edges: Semantic similarity (cosine similarity of TF-IDF vectors or word overlap).
    3. Graph Ranking: PageRank power iteration with damping factor (d=0.85).
    """

    def __init__(
        self,
        damping_factor: float = 0.85,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-5,
        similarity_threshold: float = 0.05
    ):
        """
        Args:
            damping_factor: Probability of continuing the random walk (standard PageRank d=0.85).
                            1 - d is the teleportation probability to any random sentence.
            max_iterations: Maximum number of power iterations.
            convergence_threshold: Epsilon threshold for L1 convergence of PageRank vector.
            similarity_threshold: Minimum edge weight to keep; lower weights are zeroed to prune noise.
        """
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.similarity_threshold = similarity_threshold

    def _build_sentence_vectors(self, tokenized_sentences: List[List[str]]) -> Tuple[List[Dict[str, float]], List[str]]:
        """
        Builds sparse TF-IDF vectors for each sentence.
        """
        num_sentences = len(tokenized_sentences)
        sentence_tfs = [Counter(tokens) for tokens in tokenized_sentences]

        df = Counter()
        for tf in sentence_tfs:
            for term in tf.keys():
                df[term] += 1

        vocabulary = list(df.keys())
        idf = {term: math.log((num_sentences + 1.0) / (freq + 1.0)) + 1.0 for term, freq in df.items()}

        vectors = []
        for tf in sentence_tfs:
            vec = {}
            for term, count in tf.items():
                vec[term] = count * idf[term]
            vectors.append(vec)

        return vectors, vocabulary

    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """
        Computes cosine similarity between two sparse term vectors:
        CosineSim(u, v) = (u . v) / (||u|| * ||v||)
        """
        intersection = set(vec1.keys()).intersection(set(vec2.keys()))
        if not intersection:
            return 0.0

        dot_product = sum(vec1[w] * vec2[w] for w in intersection)
        norm1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
        norm2 = math.sqrt(sum(val ** 2 for val in vec2.values()))

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _build_similarity_matrix(self, vectors: List[Dict[str, float]]) -> List[List[float]]:
        """
        Constructs the undirected weighted adjacency matrix where A[i][j] = similarity(S_i, S_j).
        Self-loops A[i][i] are set to 0.0.
        """
        n = len(vectors)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                sim = self._cosine_similarity(vectors[i], vectors[j])
                if sim >= self.similarity_threshold:
                    matrix[i][j] = sim
                    matrix[j][i] = sim

        return matrix

    def _run_pagerank(self, matrix: List[List[float]]) -> List[float]:
        """
        Executes PageRank power iteration:
        PR(i) = (1 - d) / N + d * SUM_{j in neighbors} (PR(j) * W_ji / degree(j))
        """
        n = len(matrix)
        if n == 0:
            return []
        if n == 1:
            return [1.0]

        # Compute out-degree (sum of edge weights) for each node
        degrees = [sum(row) for row in matrix]

        # Initialize uniform probability distribution
        scores = [1.0 / n] * n

        teleport = (1.0 - self.damping_factor) / n

        for iteration in range(self.max_iterations):
            new_scores = [0.0] * n
            
            # Handle dangling nodes (nodes with 0 degree)
            dangling_weight = sum(scores[j] for j in range(n) if degrees[j] == 0)

            for i in range(n):
                inbound_sum = 0.0
                for j in range(n):
                    if degrees[j] > 0 and matrix[j][i] > 0:
                        inbound_sum += (scores[j] * matrix[j][i]) / degrees[j]

                # Incorporate damping factor, inbound graph flow, and dangling node redistribution
                new_scores[i] = teleport + self.damping_factor * (inbound_sum + (dangling_weight / n))

            # Check convergence via L1 norm difference
            diff = sum(abs(new_scores[i] - scores[i]) for i in range(n))
            scores = new_scores

            if diff < self.convergence_threshold:
                break

        return scores

    def _select_mmr(
        self,
        scored_sentences: List[Dict[str, Any]],
        vectors: List[Dict[str, float]],
        num_sentences: int,
        diversity_lambda: float = 0.65,
        sim_threshold: float = 0.40,
        min_words: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Selects sentences using Maximal Marginal Relevance (MMR):
        MMR = argmax_{s_i in R \ S} [ lambda * Score(s_i) - (1 - lambda) * max_{s_j in S} Sim(s_i, s_j) ]
        
        Args:
            scored_sentences: Sentences with computed PageRank scores.
            vectors: TF-IDF sparse vectors for cosine similarity computation.
            num_sentences: Target sentence count.
            diversity_lambda: Weight on centrality (1.0 = pure PageRank, 0.0 = pure diversity).
            sim_threshold: Sentences with similarity >= this to any chosen sentence are skipped.
            min_words: Minimum words required for an extractive candidate (drops context-free snippets).
        """
        # Filter out short or isolated sentences that lack standalone context
        valid_candidates = [s for s in scored_sentences if len(s["sentence"].split()) >= min_words]
        if len(valid_candidates) < num_sentences:
            # Fallback if text is unusually terse
            valid_candidates = [s for s in scored_sentences if len(s["sentence"].split()) >= 6]
        if not valid_candidates:
            valid_candidates = scored_sentences

        max_sc = max(s["score"] for s in valid_candidates)
        min_sc = min(s["score"] for s in valid_candidates)
        denom = (max_sc - min_sc) if (max_sc - min_sc) > 0 else 1.0

        # Pick first sentence: highest PageRank score
        first_pick = max(valid_candidates, key=lambda x: x["score"])
        selected = [first_pick]
        remaining = [s for s in valid_candidates if s["index"] != first_pick["index"]]

        while len(selected) < num_sentences and remaining:
            best_cand = None
            best_mmr = -float("inf")

            for cand in remaining:
                cand_vec = vectors[cand["index"]]
                
                # Compute maximum similarity to any already-selected sentence
                max_sim = max(
                    self._cosine_similarity(cand_vec, vectors[sel["index"]])
                    for sel in selected
                )

                # Skip sentence if too similar to an already selected sentence (redundancy penalty)
                if max_sim >= sim_threshold:
                    continue

                norm_score = (cand["score"] - min_sc) / denom
                mmr_score = (diversity_lambda * norm_score) - ((1.0 - diversity_lambda) * max_sim)

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_cand = cand

            # If all candidates exceeded similarity threshold, relax threshold to pick next best
            if best_cand is None and remaining:
                best_cand = max(remaining, key=lambda x: x["score"])

            if best_cand:
                selected.append(best_cand)
                remaining = [s for s in remaining if s["index"] != best_cand["index"]]

        return selected

    def summarize(
        self,
        text: str,
        num_sentences: int = 3,
        sections: Optional[List[Dict[str, Any]]] = None,
        diversity_lambda: float = 0.65,
        min_words: int = 10
    ) -> Dict[str, Any]:
        """
        Generates an extractive summary using TextRank graph centrality + MMR redundancy filtering.
        If sections are provided, applies Section-Aware Stratified Extraction for balanced coverage.
        
        Args:
            text: Raw document text.
            num_sentences: Desired sentence count in the final summary.
            sections: Optional list of structured paper sections (title, text).
            diversity_lambda: MMR diversity trade-off (default 0.65).
            min_words: Minimum words per sentence to avoid context-free fragments.
            
        Returns:
            Dict containing summary string, selected sentences, and metadata.
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

        if len(raw_sentences) <= num_sentences:
            return {
                "summary": " ".join(raw_sentences),
                "selected_sentences": raw_sentences,
                "sentence_scores": [{"index": i, "sentence": s, "score": 1.0} for i, s in enumerate(raw_sentences)],
                "total_sentences": len(raw_sentences)
            }

        # Build sentence TF-IDF vectors for PageRank and MMR
        tokenized_sentences = [
            remove_stopwords(tokenize_words(s, lowercase=True))
            for s in raw_sentences
        ]
        vectors, _ = self._build_sentence_vectors(tokenized_sentences)
        sim_matrix = self._build_similarity_matrix(vectors)
        pagerank_scores = self._run_pagerank(sim_matrix)

        scored_sentences = []
        for idx, (sent, pr) in enumerate(zip(raw_sentences, pagerank_scores)):
            scored_sentences.append({
                "index": idx,
                "sentence": sent,
                "score": float(pr)
            })

        # --- Section-Aware Stratified Path ---
        if sections and len(sections) >= 2 and num_sentences >= len(sections):
            chosen_sentences = []
            per_section_quota = max(1, num_sentences // len(sections))
            
            for sec in sections:
                sec_sents = set(segment_sentences(sec["text"]))
                sec_scored = [s for s in scored_sentences if s["sentence"] in sec_sents]
                if sec_scored:
                    sec_selected = self._select_mmr(
                        scored_sentences=sec_scored,
                        vectors=vectors,
                        num_sentences=per_section_quota,
                        diversity_lambda=diversity_lambda,
                        sim_threshold=0.45,
                        min_words=min_words
                    )
                    chosen_sentences.extend(sec_selected)

            # Fill any remaining slots up to num_sentences
            if len(chosen_sentences) < num_sentences:
                chosen_indices = {s["index"] for s in chosen_sentences}
                leftover = [s for s in scored_sentences if s["index"] not in chosen_indices]
                needed = num_sentences - len(chosen_sentences)
                fillers = self._select_mmr(
                    scored_sentences=leftover,
                    vectors=vectors,
                    num_sentences=needed,
                    diversity_lambda=diversity_lambda,
                    sim_threshold=0.45,
                    min_words=min_words
                )
                chosen_sentences.extend(fillers)

            top_ranked = chosen_sentences[:num_sentences]
        else:
            # --- Standard Global MMR Path ---
            top_ranked = self._select_mmr(
                scored_sentences=scored_sentences,
                vectors=vectors,
                num_sentences=num_sentences,
                diversity_lambda=diversity_lambda,
                sim_threshold=0.40,
                min_words=min_words
            )

        # Re-sort into chronological appearance order
        selected_in_order = sorted(top_ranked, key=lambda x: x["index"])
        summary_sentences = [item["sentence"] for item in selected_in_order]

        return {
            "summary": " ".join(summary_sentences),
            "selected_sentences": summary_sentences,
            "sentence_scores": scored_sentences,
            "total_sentences": len(raw_sentences)
        }
