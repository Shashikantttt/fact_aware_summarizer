"""
Natural Language Inference (NLI) Factual Verifier.
Verifies claims in abstractive summaries against retrieved source evidence.
Classifies each claim as:
  - SUPPORTED (Entailment)
  - CONTRADICTED (Hallucination / Factual Conflict)
  - UNCERTAIN (Neutral / Unsubstantiated Claim)
Includes both neural NLI inference and offline heuristic verification for sandbox environments.
"""
import re
from typing import List, Dict, Any, Optional, Tuple


class NLIVerifier:
    """
    NLI Verifier that treats retrieved evidence as PREMISE and summary sentence as HYPOTHESIS.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        entailment_threshold: float = 0.50,
        contradiction_threshold: float = 0.40,
        device: Optional[str] = None,
        mock_mode: bool = False
    ):
        """
        Args:
            model_name: Pretrained NLI model identifier.
            entailment_threshold: Probability threshold above which a claim is tagged SUPPORTED.
            contradiction_threshold: Probability threshold above which a claim is flagged CONTRADICTED.
            device: 'cuda', 'cpu', or None.
            mock_mode: If True, uses offline heuristic checking for sandbox testing.
        """
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.device = device
        self.mock_mode = mock_mode
        self.pipeline = None

        if not self.mock_mode:
            self._load_pipeline()

    def _load_pipeline(self):
        """Attempts to load transformers NLI pipeline."""
        try:
            import torch
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

            device_id = 0 if (torch.cuda.is_available() and self.device != "cpu") else -1
            print(f"Loading NLI verifier ({self.model_name}) on device={device_id}...")

            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=device_id,
                top_k=None  # return all label probabilities
            )
        except Exception as e:
            print(f"[NLIVerifier] Neural NLI model not loaded ({e}). Using heuristic verifier.")
            self.mock_mode = True

    def _heuristic_check(self, premise: str, hypothesis: str) -> Tuple[str, Dict[str, float]]:
        """
        Offline rule-based fallback for sandbox testing:
        Detects numerical contradictions, polarity reversals, and entity alignment.
        """
        prem_lower = premise.lower()
        hyp_lower = hypothesis.lower()

        # 1. Number discrepancy check
        prem_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', prem_lower))
        hyp_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', hyp_lower))

        # If hypothesis introduces numbers not in premise (e.g. 14 million vs 40 million)
        if hyp_numbers and not hyp_numbers.issubset(prem_numbers):
            return "contradiction", {"entailment": 0.05, "neutral": 0.15, "contradiction": 0.80}

        # 2. Polar negation reversal check
        neg_words = {"not", "never", "declined", "refused", "denied", "failed", "decreased", "dropped", "fell"}
        prem_has_neg = any(w in prem_lower.split() for w in neg_words)
        hyp_has_neg = any(w in hyp_lower.split() for w in neg_words)

        if prem_has_neg != hyp_has_neg and any(w in prem_lower for w in ["increase", "approve", "agree", "accept", "rose"]):
            return "contradiction", {"entailment": 0.08, "neutral": 0.17, "contradiction": 0.75}

        # 3. Word overlap heuristic
        prem_words = set(re.findall(r'\b[a-z]{4,}\b', prem_lower))
        hyp_words = set(re.findall(r'\b[a-z]{4,}\b', hyp_lower))
        if not hyp_words:
            return "neutral", {"entailment": 0.20, "neutral": 0.60, "contradiction": 0.20}

        overlap = len(hyp_words.intersection(prem_words)) / len(hyp_words)
        if overlap >= 0.65:
            return "entailment", {"entailment": 0.85, "neutral": 0.10, "contradiction": 0.05}
        elif overlap >= 0.35:
            return "neutral", {"entailment": 0.30, "neutral": 0.55, "contradiction": 0.15}
        else:
            return "neutral", {"entailment": 0.10, "neutral": 0.70, "contradiction": 0.20}

    def verify_claim(self, claim: str, evidence_sentences: List[str]) -> Dict[str, Any]:
        """
        Verifies a single claim hypothesis against premise evidence.
        
        Args:
            claim: The summary sentence to verify (Hypothesis).
            evidence_sentences: List of supporting source sentences (Premise).
            
        Returns:
            Dict containing predicted_label, probabilities, confidence, and premise used.
        """
        premise = " ".join(evidence_sentences).strip()
        if not premise:
            return {
                "claim": claim,
                "premise": "",
                "label": "uncertain",
                "status": "unsupported",
                "probabilities": {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0},
                "confidence": 1.0,
                "badge": "yellow"
            }

        if self.mock_mode or self.pipeline is None:
            pred_label, probs = self._heuristic_check(premise, claim)
        else:
            # Neural NLI model format
            formatted_input = f"premise: {premise} hypothesis: {claim}"
            raw_scores = self.pipeline(formatted_input)[0]
            probs = {}
            for item in raw_scores:
                lbl = item["label"].lower()
                if "entail" in lbl:
                    probs["entailment"] = float(item["score"])
                elif "contra" in lbl:
                    probs["contradiction"] = float(item["score"])
                else:
                    probs["neutral"] = float(item["score"])

            # Determine decision using calibrated thresholds
            p_contra = probs.get("contradiction", 0.0)
            p_entail = probs.get("entailment", 0.0)

            if p_contra >= self.contradiction_threshold:
                pred_label = "contradiction"
            elif p_entail >= self.entailment_threshold:
                pred_label = "entailment"
            else:
                pred_label = "neutral"

        # Map to user-friendly status and colors
        if pred_label == "entailment":
            status = "supported"
            badge = "green"
            confidence = probs.get("entailment", 0.8)
        elif pred_label == "contradiction":
            status = "contradicted"
            badge = "red"
            confidence = probs.get("contradiction", 0.8)
        else:
            status = "uncertain"
            badge = "yellow"
            confidence = probs.get("neutral", 0.6)

        return {
            "claim": claim,
            "premise": premise,
            "label": pred_label,
            "status": status,
            "probabilities": probs,
            "confidence": round(confidence, 4),
            "badge": badge
        }

    def verify_summary(self, evidence_retrieval_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verifies all claims in an evidence retrieval payload.
        
        Args:
            evidence_retrieval_results: Output from EvidenceRetriever.retrieve_evidence_for_claims()
            
        Returns:
            Dict containing detailed claim reports and overall summary factuality ratio.
        """
        claims_report = []
        counts = {"supported": 0, "contradicted": 0, "uncertain": 0}

        for item in evidence_retrieval_results:
            claim = item["claim"]
            evidence_texts = [e["sentence"] for e in item.get("retrieved_evidence", [])]
            res = self.verify_claim(claim, evidence_texts)
            res["evidence_sources"] = item.get("retrieved_evidence", [])
            claims_report.append(res)
            counts[res["status"]] += 1

        total_claims = len(claims_report)
        factuality_score = (counts["supported"] / total_claims) if total_claims > 0 else 0.0

        return {
            "claims": claims_report,
            "counts": counts,
            "total_claims": total_claims,
            "factuality_score": round(factuality_score, 4),
            "has_hallucinations": counts["contradicted"] > 0
        }
