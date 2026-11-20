"""
Unified Summarization and Fact-Verification Pipeline.
Coordinates Preprocessing, Extractive Baselines, Abstractive Transformers,
Evidence Retrieval, and NLI Claim Verification.
"""
from typing import Dict, Any, Optional
from src.preprocessing.cleaner import clean_text
from src.preprocessing.academic_cleaner import clean_academic_paper
from src.preprocessing.tokenizer import segment_sentences
from src.extractive.tfidf_summarizer import TFIDFSummarizer
from src.extractive.textrank_summarizer import TextRankSummarizer
from src.abstractive.transformer_summarizer import TransformerSummarizer
from src.retrieval.evidence_retriever import EvidenceRetriever
from src.verification.nli_verifier import NLIVerifier
from src.evaluation.metrics import Timer, compute_summary_metrics
from src.evaluation.relevance import RelevanceScorer
from src.remote.colab_client import ColabClient, ColabClientError


class ExplainableSummarizationPipeline:
    """
    End-to-End Orchestrator for Explainable, Fact-Aware Text Summarization.
    Combines:
      - Academic & General Preprocessing
      - Extractive Graph Centrality (TextRank)
      - Abstractive Generation (BART/T5) with Map-Reduce Chunking
      - Remote GPU Inference via Google Colab T4 (FastAPI + ngrok)
      - Semantic Evidence Grounding (MiniLM)
      - Factuality Verification (NLI)
      - Topic Relevance & Quality Scoring (Separating Truth from Relevance)
    """

    def __init__(
        self,
        abstractive_model: str = "facebook/bart-large-cnn",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        nli_model: str = "cross-encoder/nli-deberta-v3-base",
        device: Optional[str] = None,
        mock_mode: bool = False,
        model_path: Optional[str] = None,
        backend: str = "local_cpu",
        colab_url: Optional[str] = None
    ):
        print(f"Initializing Explainable Summarization Pipeline (backend={backend})...")
        self.mock_mode = mock_mode
        self.backend = backend
        self.colab_url = colab_url
        self.colab_client = ColabClient(base_url=colab_url) if (colab_url and backend == "colab_gpu") else None
        self.device = device
        self.abstractive_model = abstractive_model

        self.tfidf_summarizer = TFIDFSummarizer(position_bias=True)
        self.textrank_summarizer = TextRankSummarizer(damping_factor=0.85)
        
        # Local transformer summarizer (always available for local execution or fallback)
        self.transformer_summarizer = TransformerSummarizer(
            model_name=abstractive_model,
            device=device,
            mock_mode=mock_mode,
            model_path=model_path
        )
        self.evidence_retriever = EvidenceRetriever(
            model_name=embedding_model,
            device=device,
            top_k=2,
            mock_mode=mock_mode
        )
        self.nli_verifier = NLIVerifier(
            model_name=nli_model,
            device=device,
            mock_mode=mock_mode
        )
        self.relevance_scorer = RelevanceScorer()

    def process(
        self,
        text: str,
        extractive_sentences: int = 3,
        num_beams: int = 4,
        top_k_evidence: int = 2,
        reference_abstract: Optional[str] = None,
        is_academic_paper: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executes the full pipeline on input text:
          1. Clean and segment text (academic paper noise removal if applicable)
          2. Generate TF-IDF baseline summary
          3. Generate Abstractive transformer summary (with hierarchical consolidation)
          4. Retrieve top-k source evidence for each abstractive claim
          5. Verify claims via NLI (Entailed, Contradicted, Neutral)
          6. Score Topic Relevance (distinguish true support from irrelevant noise)
          7. Benchmark against author Abstract (ROUGE evaluation) if available
        """
        extracted_abstract = reference_abstract
        sections = None

        # Check for academic paper signatures
        if is_academic_paper or "REFERENCES" in text or "References" in text or "Abstract—" in text or "Abstract -" in text:
            academic_data = clean_academic_paper(text)
            if not extracted_abstract:
                extracted_abstract = academic_data.get("abstract")
            sections = academic_data.get("sections", [])
            
            # ZERO DATA LEAKAGE: If an abstract is isolated and used as reference summary,
            # feed ONLY the body text (excluding the abstract) into summarization.
            # This ensures ROUGE rigorously evaluates whether the model synthesizes the paper's body.
            if extracted_abstract:
                cleaned_text = academic_data["body_text"]
            else:
                cleaned_text = academic_data["clean_full_text"]
        else:
            cleaned_text = clean_text(text)

        source_sentences = segment_sentences(cleaned_text)

        if not source_sentences:
            return {"error": "Input text contains no valid sentences."}

        # 1. Extractive Baselines (with Section-Aware Stratification & MMR)
        with Timer() as t_tfidf:
            tfidf_result = self.tfidf_summarizer.summarize(cleaned_text, num_sentences=extractive_sentences)
            
        with Timer() as t_tr:
            textrank_result = self.textrank_summarizer.summarize(
                cleaned_text,
                num_sentences=extractive_sentences,
                sections=sections,
                diversity_lambda=0.65,
                min_words=12
            )

        # 2. Abstractive Summarization (Colab T4 GPU or Local CPU with fallback)
        abs_result = None
        colab_failure_reason = None

        if self.backend == "colab_gpu" and not self.mock_mode:
            if not self.colab_client or not self.colab_client.is_configured():
                colab_failure_reason = "Colab backend URL was not configured"
            else:
                try:
                    with Timer() as t_abs:
                        colab_resp = self.colab_client.summarize(
                            cleaned_text,
                            max_length=140,
                            min_length=40,
                            num_beams=num_beams
                        )
                    summary_text = colab_resp["summary"]
                    abs_result = {
                        "summary": summary_text,
                        "summary_sentences": segment_sentences(summary_text),
                        "num_chunks": 1,
                        "mode": "colab_bart",
                        "backend": "colab_gpu",
                        "label": "Neural Abstractive (BART on Colab T4 GPU)",
                        "device": colab_resp.get("device", "cuda"),
                        "model_name": f"{colab_resp.get('model_name', 'facebook/bart-large-cnn')} (Colab T4 GPU)",
                        "is_abstractive": True,
                        "failure_reason": None,
                        "generation_time_sec": colab_resp.get("generation_time_sec", round(t_abs.interval, 3)),
                        "runtime_sec": round(t_abs.interval, 3)
                    }
                except Exception as e:
                    colab_failure_reason = f"Colab GPU unreachable ({str(e)})"
                    print(f"[Pipeline] {colab_failure_reason}. Triggering fallback...")

        # If Colab was not used or failed, run local path
        if abs_result is None:
            if colab_failure_reason:
                # Step 3 requirement: If URL is unreachable, fall back to local distilbart-cnn-12-6 (or extractive)
                try:
                    print("Attempting local distilbart-cnn-12-6 fallback...")
                    fallback_summarizer = TransformerSummarizer(
                        model_name="sshleifer/distilbart-cnn-12-6",
                        device=self.device,
                        mock_mode=False
                    )
                    with Timer() as t_abs:
                        abs_result = fallback_summarizer.summarize(
                            cleaned_text,
                            num_beams=num_beams,
                            target_sentences=extractive_sentences
                        )
                    abs_result["backend"] = "local_distilbart_fallback"
                    abs_result["label"] = f"Local DistilBART Fallback ({colab_failure_reason})"
                    abs_result["failure_reason"] = colab_failure_reason
                except Exception as local_err:
                    print(f"Local DistilBART failed ({local_err}). Using extractive fallback...")
                    fallback_summarizer = TransformerSummarizer(mock_mode=True)
                    with Timer() as t_abs:
                        abs_result = fallback_summarizer.summarize(
                            cleaned_text,
                            num_beams=num_beams,
                            target_sentences=extractive_sentences
                        )
                    abs_result["backend"] = "extractive_fallback"
                    abs_result["label"] = "Extractive fallback (not abstractive)"
                    abs_result["failure_reason"] = f"{colab_failure_reason} -> Extractive fallback ({local_err})"
            else:
                # Normal local CPU execution
                with Timer() as t_abs:
                    abs_result = self.transformer_summarizer.summarize(
                        cleaned_text,
                        num_beams=num_beams,
                        target_sentences=extractive_sentences
                    )
                abs_result["backend"] = "local_cpu"

        summary_claims = abs_result["summary_sentences"]

        # 3. Evidence Retrieval
        with Timer() as t_retrieval:
            evidence_results = self.evidence_retriever.retrieve_evidence_for_claims(
                cleaned_text,
                summary_claims,
                top_k=top_k_evidence
            )

        # 4. Factual Verification (Faithfulness)
        verification_result = None
        if abs_result.get("backend") == "colab_gpu" and self.colab_client:
            try:
                with Timer() as t_verify:
                    colab_v = self.colab_client.verify(
                        summary_sentences=summary_claims,
                        source_sentences=source_sentences,
                        top_k=top_k_evidence
                    )
                verification_result = colab_v
            except Exception as e:
                print(f"[Pipeline] Colab remote verify failed ({e}). Using local verifier...")

        if verification_result is None:
            with Timer() as t_verify:
                verification_result = self.nli_verifier.verify_summary(evidence_results)

        # 5. Topic Relevance & Quality Scoring (Faithfulness != Relevance)
        topic_anchor = extracted_abstract if extracted_abstract else " ".join(source_sentences[:3])
        relevance_eval = self.relevance_scorer.evaluate_summary(
            verification_result["claims"],
            topic_context=topic_anchor
        )

        # 6. Optional Benchmark Against Gold-Standard Abstract (ROUGE)
        abstract_benchmark = None
        if extracted_abstract:
            tfidf_rouge = compute_summary_metrics(
                candidate_summary=tfidf_result["summary"],
                reference_summary=extracted_abstract,
                source_document=cleaned_text
            )
            abs_rouge = compute_summary_metrics(
                candidate_summary=abs_result["summary"],
                reference_summary=extracted_abstract,
                source_document=cleaned_text
            )
            abstract_benchmark = {
                "reference_abstract": extracted_abstract,
                "extractive_rouge": {
                    "rouge_1": tfidf_rouge["rouge_1"],
                    "rouge_2": tfidf_rouge["rouge_2"],
                    "rouge_l": tfidf_rouge["rouge_l"]
                },
                "tfidf_rouge": {
                    "rouge_1": tfidf_rouge["rouge_1"],
                    "rouge_2": tfidf_rouge["rouge_2"],
                    "rouge_l": tfidf_rouge["rouge_l"]
                },
                "textrank_rouge": {
                    "rouge_1": tfidf_rouge["rouge_1"],
                    "rouge_2": tfidf_rouge["rouge_2"],
                    "rouge_l": tfidf_rouge["rouge_l"]
                },
                "abstractive_rouge": {
                    "rouge_1": abs_rouge["rouge_1"],
                    "rouge_2": abs_rouge["rouge_2"],
                    "rouge_l": abs_rouge["rouge_l"]
                }
            }

        return {
            "source": {
                "text": cleaned_text,
                "sentences": source_sentences,
                "total_sentences": len(source_sentences),
                "reference_abstract": extracted_abstract
            },
            "extractive": {
                "summary": tfidf_result["summary"],
                "sentences": tfidf_result["selected_sentences"],
                "runtime_sec": round(t_tfidf.interval, 3),
                "tfidf": {
                    "summary": tfidf_result["summary"],
                    "sentences": tfidf_result["selected_sentences"],
                    "runtime_sec": round(t_tfidf.interval, 3)
                },
                "textrank": {
                    "summary": tfidf_result["summary"],
                    "sentences": tfidf_result["selected_sentences"],
                    "runtime_sec": round(t_tfidf.interval, 3)
                }
            },
            "abstractive": {
                "summary": abs_result["summary"],
                "sentences": summary_claims,
                "num_chunks": abs_result.get("num_chunks", 1),
                "mode": abs_result.get("mode", "neural_bart"),
                "backend": abs_result.get("backend", self.backend),
                "label": abs_result.get("label", "Abstractive Summary"),
                "device": abs_result.get("device", "cpu"),
                "model_name": abs_result.get("model_name", self.transformer_summarizer.model_name),
                "is_abstractive": abs_result.get("is_abstractive", True),
                "failure_reason": abs_result.get("failure_reason"),
                "generation_time_sec": abs_result.get("generation_time_sec", round(t_abs.interval, 3)),
                "runtime_sec": round(t_abs.interval, 3)
            },
            "verification": {
                "claims": relevance_eval["claims"],
                "counts": verification_result["counts"],
                "factuality_score": verification_result["factuality_score"],
                "has_hallucinations": verification_result["has_hallucinations"],
                "retrieval_runtime_sec": round(t_retrieval.interval, 3),
                "verification_runtime_sec": round(t_verify.interval, 3)
            },
            "relevance": {
                "overall_relevance_score": relevance_eval["overall_relevance_score"],
                "relevance_percentage": relevance_eval["relevance_percentage"],
                "noise_percentage": relevance_eval["noise_percentage"],
                "counts": relevance_eval["counts"],
                "quadrant_counts": relevance_eval["quadrant_counts"]
            },
            "abstract_benchmark": abstract_benchmark,
            "total_runtime_sec": round(t_tfidf.interval + t_tr.interval + t_abs.interval + t_retrieval.interval + t_verify.interval, 3)
        }
