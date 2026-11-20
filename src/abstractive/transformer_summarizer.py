"""
Abstractive Transformer Summarizer (BART / T5).
Handles token limits via intelligent chunk-and-combine for long documents / research papers.
Includes fallback/offline execution mode for environments without PyTorch or HuggingFace weights.
"""
import os
import re
import math
import time
import traceback
from collections import Counter
from typing import List, Dict, Any, Optional
from src.preprocessing.cleaner import clean_text
from src.preprocessing.tokenizer import segment_sentences, chunk_text_by_tokens, tokenize_words, remove_stopwords


class Seq2SeqPipelineWrapper:
    """
    Lightweight callable pipeline wrapper around AutoModelForSeq2SeqLM & AutoTokenizer.
    Guarantees compatibility across transformers versions (including v5.x where 'summarization'
    pipeline task string was removed) while matching standard pipeline return format.
    """
    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def __call__(
        self,
        text: str,
        max_length: int = 140,
        min_length: int = 40,
        num_beams: int = 4,
        **kwargs
    ) -> List[Dict[str, str]]:
        import torch

        # Tokenize with BART's 1024 token limit
        inputs = self.tokenizer(
            text,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Ensure valid length bounds
        actual_max_length = max(max_length, 25)
        actual_min_length = min(min_length, actual_max_length - 5)
        actual_min_length = max(actual_min_length, 1)

        with torch.no_grad():
            summary_ids = self.model.generate(
                **inputs,
                max_length=actual_max_length,
                min_length=actual_min_length,
                num_beams=num_beams,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3
            )

        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
        return [{"summary_text": summary_text}]


class TransformerSummarizer:
    """
    Abstractive summarizer using pretrained encoder-decoder models (BART or T5).
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        device: Optional[str] = None,
        max_chunk_tokens: int = 650,
        mock_mode: bool = False,
        model_path: Optional[str] = None
    ):
        """
        Args:
            model_name: HuggingFace model identifier (e.g. 'facebook/bart-large-cnn' or 'sshleifer/distilbart-cnn-12-6').
            device: 'cuda', 'cpu', or None (auto-detects CUDA).
            max_chunk_tokens: Max words per chunk to guarantee staying below BART's 1024 token limit.
            mock_mode: If True, operates without loading heavy PyTorch weights (useful for dry runs/tests).
            model_path: Optional local directory path for offline execution with pre-downloaded weights.
        """
        self.model_name = model_name
        self.model_path = model_path
        self.effective_model_path = model_path if (model_path and os.path.exists(model_path)) else model_name
        self.max_chunk_tokens = max_chunk_tokens
        self.mock_mode = mock_mode
        self.pipeline = None
        self.model = None
        self.tokenizer = None
        self.is_neural = False
        self.device_name = "cpu"
        self.load_error: Optional[str] = None
        self.load_traceback: Optional[str] = None
        self.last_generation_time_sec: float = 0.0

        if not self.mock_mode:
            self._initialize_pipeline(device)

    def _initialize_pipeline(self, device: Optional[str] = None):
        """Attempts to load transformers model and tokenizer with auto device & precision selection."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            # Automatic device selection: cuda if available, otherwise cpu
            if device is None:
                self.device_name = "cuda" if torch.cuda.is_available() else "cpu"
            elif device == "cuda":
                self.device_name = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device_name = "cpu"

            # Use float16 on GPU for memory efficiency and speed; float32 on CPU
            torch_dtype = torch.float16 if self.device_name == "cuda" else torch.float32

            print(f"Loading transformer summarizer ({self.effective_model_path}) on device={self.device_name} (dtype={torch_dtype})...")
            if not os.path.exists(self.effective_model_path):
                print(f"Downloading {self.effective_model_path}, ~1.6 GB, first run only (or loading from cache)...")

            # Load tokenizer and seq2seq model directly
            self.tokenizer = AutoTokenizer.from_pretrained(self.effective_model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.effective_model_path,
                torch_dtype=torch_dtype
            ).to(self.device_name)
            self.model.eval()

            # Wrap in pipeline-compatible callable
            self.pipeline = Seq2SeqPipelineWrapper(self.model, self.tokenizer, self.device_name)
            self.is_neural = True
            self.mock_mode = False
            self.load_error = None
            self.load_traceback = None
            print(f"[TransformerSummarizer] Successfully loaded {self.effective_model_path} on {self.device_name}.")
        except Exception as e:
            self.load_error = f"{type(e).__name__}: {str(e)}"
            self.load_traceback = traceback.format_exc()
            print(f"[TransformerSummarizer] Failed to load model weights ({self.effective_model_path}): {self.load_error}")
            print("Full Traceback:")
            traceback.print_exc()
            self.pipeline = None
            self.model = None
            self.tokenizer = None
            self.is_neural = False
            self.mock_mode = True

    def _extractive_fallback_chunk(self, text: str) -> str:
        """Intelligent fallback: select highest TF-IDF scored sentence from this chunk."""
        sentences = segment_sentences(text)
        if not sentences:
            return ""
        if len(sentences) == 1:
            return sentences[0].strip()
        tokenized = [remove_stopwords(tokenize_words(s, lowercase=True)) for s in sentences]
        df = Counter()
        for toks in tokenized:
            for t in set(toks):
                df[t] += 1
        n = len(sentences)
        best_idx, best_score = 0, -1.0
        for i, toks in enumerate(tokenized):
            if not toks:
                continue
            score = sum(math.log((n + 1.0) / (df[t] + 1.0)) + 1.0 for t in toks) / len(toks)
            if score > best_score:
                best_score = score
                best_idx = i
        return sentences[best_idx].strip()

    def _summarize_single_chunk(
        self,
        text: str,
        max_length: int = 140,
        min_length: int = 40,
        num_beams: int = 4
    ) -> str:
        """Summarizes a single text block within token limits."""
        if self.mock_mode or self.pipeline is None or not self.is_neural:
            return self._extractive_fallback_chunk(text)

        try:
            outputs = self.pipeline(
                text,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams
            )
            return outputs[0]["summary_text"].strip()
        except Exception as e:
            print(f"[TransformerSummarizer] Error generating summary for chunk ({type(e).__name__}: {e}). Traceback:")
            traceback.print_exc()
            self.load_error = f"Inference Error: {type(e).__name__}: {str(e)}"
            self.load_traceback = traceback.format_exc()
            return self._extractive_fallback_chunk(text)

    def summarize(
        self,
        text: str,
        max_length: int = 140,
        min_length: int = 40,
        num_beams: int = 4,
        target_sentences: int = 4,
        remove_academic_citations: bool = True
    ) -> Dict[str, Any]:
        """
        Generates an abstractive summary.
        If document exceeds max_chunk_tokens, applies Hierarchical Map-Reduce:
        1. Map: Summarizes each chunk.
        2. Reduce: Combines and synthesizes a concise, executive 3-5 sentence summary.
        
        Args:
            text: Raw input text (news or research paper).
            max_length: Target max tokens in generated summary.
            min_length: Minimum tokens in generated summary.
            num_beams: Beam search beam count (default 4 per BART paper).
            target_sentences: Target sentence count for consolidated summaries.
            remove_academic_citations: Strips academic citations [1] before feeding to model.
            
        Returns:
            Dict containing summary string, chunks used, and metadata.
        """
        start_time = time.time()
        cleaned = clean_text(text, remove_academic_citations=remove_academic_citations)
        sentences = segment_sentences(cleaned)
        word_count = sum(len(tokenize_words(s)) for s in sentences)

        is_real_neural = (not self.mock_mode and self.pipeline is not None and self.is_neural)
        mode_str = "neural_bart" if is_real_neural else "extractive_fallback"
        label_str = f"Neural Abstractive ({self.model_name})" if is_real_neural else "Extractive fallback (not abstractive)"

        # If document fits within single chunk token budget
        if word_count <= self.max_chunk_tokens:
            summary = self._summarize_single_chunk(
                cleaned,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams
            )
            runtime = round(time.time() - start_time, 3)
            self.last_generation_time_sec = runtime
            # Re-check is_real_neural in case chunk inference failed
            is_real_neural = (not self.mock_mode and self.pipeline is not None and self.is_neural)
            mode_str = "neural_bart" if is_real_neural else "extractive_fallback"
            label_str = f"Neural Abstractive ({self.model_name})" if is_real_neural else "Extractive fallback (not abstractive)"

            return {
                "summary": summary,
                "summary_sentences": segment_sentences(summary),
                "num_chunks": 1,
                "word_count": word_count,
                "model_name": self.model_name,
                "device": self.device_name,
                "mode": mode_str,
                "label": label_str,
                "failure_reason": self.load_error if not is_real_neural else None,
                "runtime_sec": runtime,
                "generation_time_sec": runtime,
                "is_abstractive": is_real_neural
            }

        # Long document path: Hierarchical Map-Reduce
        chunks = chunk_text_by_tokens(sentences, max_tokens_per_chunk=self.max_chunk_tokens, overlap_sentences=1)
        chunk_summaries = []
        for c in chunks:
            chunk_sum = self._summarize_single_chunk(
                c["text"],
                max_length=max(min_length + 10, max_length // len(chunks) + 30),
                min_length=min_length // 2,
                num_beams=num_beams
            )
            if chunk_sum:
                chunk_summaries.append(chunk_sum)

        combined_intermediate = " ".join(chunk_summaries)
        
        # Second Pass (Reduce): Consolidate intermediate chunk summaries into an executive summary
        if not self.mock_mode and self.pipeline is not None and self.is_neural:
            # Neural second pass
            final_summary = self._summarize_single_chunk(
                combined_intermediate,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams
            )
        else:
            # Fallback second pass: MMR-based deduplication across chunk summaries
            inter_sents = segment_sentences(combined_intermediate)
            if len(inter_sents) <= target_sentences:
                final_summary = combined_intermediate
            else:
                tokenized = [remove_stopwords(tokenize_words(s, lowercase=True)) for s in inter_sents]
                df = Counter()
                for toks in tokenized:
                    for t in set(toks):
                        df[t] += 1
                n_sents = len(inter_sents)

                def _tfidf_vec(toks):
                    tf = Counter(toks)
                    return {t: tf[t] * (math.log((n_sents + 1.0) / (df[t] + 1.0)) + 1.0) for t in toks}

                def _cosine(v1, v2):
                    shared = set(v1) & set(v2)
                    if not shared:
                        return 0.0
                    dot = sum(v1[k] * v2[k] for k in shared)
                    n1 = math.sqrt(sum(x * x for x in v1.values()))
                    n2 = math.sqrt(sum(x * x for x in v2.values()))
                    return dot / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0

                vecs = [_tfidf_vec(t) for t in tokenized]
                scores = [sum(v.values()) / max(1, len(v)) for v in vecs]

                selected_idx = []
                remaining = list(range(n_sents))
                while len(selected_idx) < target_sentences and remaining:
                    if not selected_idx:
                        best = max(remaining, key=lambda i: scores[i])
                    else:
                        def mmr(i):
                            max_sim = max(_cosine(vecs[i], vecs[j]) for j in selected_idx)
                            return 0.65 * scores[i] - 0.35 * max_sim
                        best = max(remaining, key=mmr)
                    selected_idx.append(best)
                    remaining.remove(best)

                selected_idx.sort()
                final_summary = " ".join(inter_sents[i] for i in selected_idx)

        runtime = round(time.time() - start_time, 3)
        self.last_generation_time_sec = runtime
        is_real_neural = (not self.mock_mode and self.pipeline is not None and self.is_neural)
        mode_str = "neural_bart" if is_real_neural else "extractive_fallback"
        label_str = f"Neural Abstractive ({self.model_name})" if is_real_neural else "Extractive fallback (not abstractive)"

        return {
            "summary": final_summary,
            "summary_sentences": segment_sentences(final_summary),
            "num_chunks": len(chunks),
            "word_count": word_count,
            "model_name": self.model_name,
            "device": self.device_name,
            "mode": mode_str,
            "label": label_str,
            "failure_reason": self.load_error if not is_real_neural else None,
            "runtime_sec": runtime,
            "generation_time_sec": runtime,
            "is_abstractive": is_real_neural
        }
