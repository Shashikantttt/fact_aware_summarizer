"""
Classical extractive summarization baselines (TF-IDF & TextRank).
Pure mathematical implementations running without external model weights.
"""
from src.extractive.tfidf_summarizer import TFIDFSummarizer
from src.extractive.textrank_summarizer import TextRankSummarizer

__all__ = ["TFIDFSummarizer", "TextRankSummarizer"]
