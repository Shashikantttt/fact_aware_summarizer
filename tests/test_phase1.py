"""
Verification test for Phase 1: Classical NLP Pipeline (Preprocessing, TF-IDF, TextRank).
Runs locally with zero external network or model weight requirements.
"""
import json
import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.cleaner import clean_text, remove_citations
from src.preprocessing.tokenizer import (
    segment_sentences,
    tokenize_words,
    remove_stopwords,
    chunk_text_by_tokens
)
from src.extractive.tfidf_summarizer import TFIDFSummarizer
from src.extractive.textrank_summarizer import TextRankSummarizer


def test_cleaner():
    print("[1/5] Testing text cleaner & citation remover...")
    sample_text = "NLP systems [1, 2] revolutionized text (Vaswani et al., 2017). They use trans-\nformer mechanisms."
    cleaned = clean_text(sample_text, remove_academic_citations=True)
    assert "[1, 2]" not in cleaned, "Bracket citation should be removed"
    assert "Vaswani" not in cleaned, "Author citation should be removed"
    assert "transformer" in cleaned, "Broken hyphenation should be joined"
    print("  -> Passed cleaner tests.")


def test_tokenizer():
    print("[2/5] Testing sentence segmentation and tokenization...")
    text = "Dr. Smith presented findings at the U.S. summit on Oct. 25. The inflation rate reached 5.25% in 2023! Did it help? Yes, it did."
    sentences = segment_sentences(text)
    assert len(sentences) == 4, f"Expected 4 sentences, got {len(sentences)}: {sentences}"
    assert "Dr. Smith" in sentences[0], "Dr. should not cause false sentence split"
    assert "U.S." in sentences[0], "U.S. should not cause false sentence split"
    assert "5.25%" in sentences[1], "Decimals should not cause false sentence split"
    
    words = tokenize_words(sentences[0])
    no_stops = remove_stopwords(words)
    assert "dr" in words and "smith" in words
    assert "the" not in no_stops and "at" not in no_stops
    print("  -> Passed tokenizer tests.")


def test_chunker():
    print("[3/5] Testing long document chunking...")
    dummy_sentences = [f"This is sentence number {i} discussing academic concepts in NLP." for i in range(25)]
    chunks = chunk_text_by_tokens(dummy_sentences, max_tokens_per_chunk=50, overlap_sentences=1)
    assert len(chunks) > 1, "Should produce multiple chunks for long texts"
    assert chunks[0]["chunk_id"] == 0
    print(f"  -> Generated {len(chunks)} chunks with token budgeting.")


def test_tfidf_summarizer():
    print("[4/5] Testing TF-IDF Summarizer on CNN/DailyMail sample...")
    samples_path = os.path.join(os.path.dirname(__file__), "..", "data", "samples", "cnn_dm_samples.json")
    with open(samples_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    article = data[0]["article"]
    summarizer = TFIDFSummarizer(position_bias=True)
    result = summarizer.summarize(article, num_sentences=2)

    assert len(result["selected_sentences"]) == 2, "Expected 2 summary sentences"
    assert result["total_sentences"] >= 4
    assert len(result["summary"]) > 0
    print(f"  -> TF-IDF Summary:\n     \"{result['summary']}\"")


def test_textrank_summarizer():
    print("[5/5] Testing TextRank Summarizer (PageRank graph ranking)...")
    samples_path = os.path.join(os.path.dirname(__file__), "..", "data", "samples", "cnn_dm_samples.json")
    with open(samples_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    article = data[1]["article"]
    summarizer = TextRankSummarizer(damping_factor=0.85)
    result = summarizer.summarize(article, num_sentences=2)

    assert len(result["selected_sentences"]) == 2, "Expected 2 summary sentences"
    assert len(result["sentence_scores"]) == result["total_sentences"]
    # Check that scores sum to roughly 1.0 (PageRank probability distribution)
    total_pr = sum(item["score"] for item in result["sentence_scores"])
    assert abs(total_pr - 1.0) < 0.05, f"PageRank distribution should sum close to 1.0, got {total_pr}"
    print(f"  -> TextRank Summary:\n     \"{result['summary']}\"")


def test_user_news_article():
    print("[Extra] Testing minister terrorism article with dialogue and quotes...")
    user_text = (
        'The external affairs minister also flagged terrorism as another major threat to global peace, '
        'saying its use as an "instrument of statecraft" was "simply unacceptable" and could neither be justified nor rationalised.\n\n'
        '"Terrorism is another deadly threat to peace," he said, adding that the threat continued because not all countries displayed zero tolerance towards it.\n\n'
        '"Despite history holding so many lessons, some still allow political calculations to take precedence," Jaishankar said, calling for greater global awareness and stronger collective action to counter terrorism in all its forms.\n\n'
        'Linking global peace to economic development, Jaishankar said progress and prosperity were equally interconnected, particularly in a world dependent on global supply chains, markets and shared resources.\n\n'
        '"Progress and prosperity are the flip side of conflict and violence," the minister said.\n\n'
        '"The world cannot advance half prosperous and half denied in an era of intricate supply chains, of globalized markets, and of common resources. This is simply a prescription for continuous instability," he added.'
    )
    sents = segment_sentences(clean_text(user_text))
    assert len(sents) >= 5, f"Expected at least 5-6 sentences, got {len(sents)}"
    
    tr = TextRankSummarizer().summarize(user_text, num_sentences=2)
    assert len(tr["selected_sentences"]) == 2, f"Expected 2 sentences in summary, got {len(tr['selected_sentences'])}"
    assert len(tr["summary"].split()) < len(user_text.split()) * 0.5, "Summary should reduce text length by > 50%"
    print(f"  -> Segmented into {len(sents)} sentences.")
    print(f"  -> Compressed from {len(user_text.split())} words to {len(tr['summary'].split())} words ({int((1 - len(tr['summary'].split())/len(user_text.split()))*100)}% reduction).")
    print(f"  -> Extractive Summary (2 sentences):\n     \"{tr['summary']}\"")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING PHASE 1 VERIFICATION (Classical NLP Pipeline)")
    print("=" * 60)
    test_cleaner()
    test_tokenizer()
    test_chunker()
    test_tfidf_summarizer()
    test_textrank_summarizer()
    test_user_news_article()
    print("=" * 60)
    print("ALL PHASE 1 TESTS PASSED SUCCESSFULLY! (Zero external models needed)")
    print("=" * 60)
