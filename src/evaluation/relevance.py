"""
Topic Relevance and Quality Scorer.
Distinguishes FAITHFULNESS from RELEVANCE:
A claim can be 100% faithful to the source (supported by NLI) but completely irrelevant
(e.g., figure captions, license lines, author acknowledgements, or peripheral metadata).

This module measures:
1. Topic Salience: Semantic & lexical alignment with the paper's Abstract / Title / Core Topic.
2. Content Density: Ratio of informative keywords vs boilerplate.
3. 2x2 Evaluation Quadrant:
   - Supported & Relevant (🟢 High Quality)
   - Supported but Peripheral (🟡 Noise / Low Quality - Factually true, but trivial)
   - Contradicted (🔴 Hallucination)
   - Uncertain (⚪ Unsubstantiated)
"""
import re
import math
from collections import Counter
from typing import List, Dict, Any, Optional
from src.preprocessing.tokenizer import tokenize_words, remove_stopwords


# Common boilerplate / noise phrases in academic or news texts
BOILERPLATE_PATTERNS = [
    r'(?i)\bfig(?:ure)?\.?\s*\d+\b',
    r'(?i)\btable\s*\d+\b',
    r'(?i)\bauthor(?:s)?\b.*?\bthank\b',
    r'(?i)\bconflict of interest\b',
    r'(?i)\bcorrespondence to\b',
    r'(?i)\bvolume\s+\d+\b',
    r'(?i)\bjournal of\b',
    r'(?i)\bdoi:\b',
    r'(?i)\bhttps?://\b'
]


class RelevanceScorer:
    """
    Evaluates how central a summary claim is to the primary subject matter of the document.
    """

    def __init__(self, high_threshold: float = 0.35, marginal_threshold: float = 0.18):
        self.high_threshold = high_threshold
        self.marginal_threshold = marginal_threshold

    def _extract_topic_profile(self, topic_context: str) -> Counter:
        """Extracts frequency profile of salient content words in topic anchor (abstract/lead)."""
        tokens = remove_stopwords(tokenize_words(topic_context, lowercase=True))
        # Filter short tokens
        content_tokens = [t for t in tokens if len(t) >= 3 and not t.isdigit()]
        return Counter(content_tokens)

    def score_claim(
        self,
        claim: str,
        topic_context: str,
        nli_status: str = "supported"
    ) -> Dict[str, Any]:
        """
        Scores a single claim for relevance against the document's core topic.
        
        Args:
            claim: Summary statement.
            topic_context: Reference abstract, title, or lead document text.
            nli_status: 'supported', 'contradicted', or 'uncertain' from NLI.
            
        Returns:
            Dict containing relevance score, category, and 2x2 quadrant.
        """
        claim_tokens = remove_stopwords(tokenize_words(claim, lowercase=True))
        claim_content = [t for t in claim_tokens if len(t) >= 3 and not t.isdigit()]
        
        if not claim_content or not topic_context:
            return {
                "claim": claim,
                "relevance_score": 0.0,
                "relevance_label": "peripheral_noise",
                "is_relevant": False,
                "quadrant": "Supported but Peripheral" if nli_status == "supported" else nli_status.capitalize(),
                "boilerplate_flags": []
            }

        topic_counter = self._extract_topic_profile(topic_context)
        total_topic_words = sum(topic_counter.values()) or 1

        # 1. Topic Keyword Overlap (Weighted Salience)
        overlap_weight = 0.0
        overlap_words = []
        for word in claim_content:
            if word in topic_counter:
                # TF-weight in topic context
                weight = 1.0 + math.log(1.0 + topic_counter[word])
                overlap_weight += weight
                overlap_words.append(word)

        # Normalized overlap score (0.0 to 1.0)
        norm_score = overlap_weight / (len(claim_content) * 2.0)
        relevance_score = min(1.0, round(norm_score, 3))

        # 2. Check for boilerplate / noise indicators
        boilerplate_flags = []
        for pat in BOILERPLATE_PATTERNS:
            if re.search(pat, claim):
                boilerplate_flags.append(pat)
                relevance_score = max(0.0, relevance_score - 0.25)

        # 3. Label category
        if relevance_score >= self.high_threshold:
            relevance_label = "high_relevance"
            is_relevant = True
        elif relevance_score >= self.marginal_threshold:
            relevance_label = "marginal"
            is_relevant = True
        else:
            relevance_label = "peripheral_noise"
            is_relevant = False

        # 4. Map to 2x2 Evaluation Quadrant
        if nli_status == "contradicted":
            quadrant = "Contradicted (Hallucination)"
        elif nli_status == "uncertain":
            quadrant = "Uncertain / Speculative"
        elif is_relevant:
            quadrant = "Supported & Relevant (High Quality)"
        else:
            quadrant = "Supported but Peripheral (Noise)"

        return {
            "claim": claim,
            "relevance_score": relevance_score,
            "relevance_label": relevance_label,
            "is_relevant": is_relevant,
            "quadrant": quadrant,
            "overlap_keywords": overlap_words[:6],
            "boilerplate_flags": boilerplate_flags
        }

    def evaluate_summary(
        self,
        claims_with_nli: List[Dict[str, Any]],
        topic_context: str
    ) -> Dict[str, Any]:
        """
        Evaluates an entire summary across both FAITHFULNESS and RELEVANCE.
        """
        scored_claims = []
        high_count = 0
        marginal_count = 0
        noise_count = 0
        quadrant_counts = {
            "supported_and_relevant": 0,
            "supported_but_peripheral": 0,
            "contradicted": 0,
            "uncertain": 0
        }

        for item in claims_with_nli:
            claim_text = item.get("claim", "")
            nli_status = item.get("status", "uncertain").lower()

            rel_data = self.score_claim(claim_text, topic_context, nli_status=nli_status)
            merged = {**item, **rel_data}
            scored_claims.append(merged)

            if rel_data["relevance_label"] == "high_relevance":
                high_count += 1
            elif rel_data["relevance_label"] == "marginal":
                marginal_count += 1
            else:
                noise_count += 1

            if nli_status == "contradicted":
                quadrant_counts["contradicted"] += 1
            elif nli_status == "uncertain":
                quadrant_counts["uncertain"] += 1
            elif rel_data["is_relevant"]:
                quadrant_counts["supported_and_relevant"] += 1
            else:
                quadrant_counts["supported_but_peripheral"] += 1

        total = len(scored_claims) or 1
        overall_relevance = round((high_count + 0.5 * marginal_count) / total, 3)

        return {
            "claims": scored_claims,
            "overall_relevance_score": overall_relevance,
            "counts": {
                "high_relevance": high_count,
                "marginal": marginal_count,
                "peripheral_noise": noise_count,
                "total": len(scored_claims)
            },
            "quadrant_counts": quadrant_counts,
            "relevance_percentage": int(overall_relevance * 100),
            "noise_percentage": int((noise_count / total) * 100)
        }
