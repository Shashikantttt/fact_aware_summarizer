"""
Verification test for:
1. PDF Hyphenation Bug (rejoining broken words vs preserving real hyphenated terms).
2. TransformerSummarizer abstractive generation verification (asserting it doesn't silently return extractive output).
3. Honest fallback metadata verification.
"""
import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.cleaner import clean_text
from src.preprocessing.academic_cleaner import unwrap_pdf_lines
from src.abstractive.transformer_summarizer import TransformerSummarizer


def test_rejoin_broken_words_at_line_breaks():
    print("[1/6] Testing broken words at line breaks...")
    # Specific problem cases requested by user
    assert clean_text("sta-\nkeholder") == "stakeholder"
    assert clean_text("pro-\ncess") == "process"
    assert clean_text("trans-\nformer") == "transformer"
    assert clean_text("inter-\nconnected") == "interconnected"

    # Windows CRLF line endings
    assert clean_text("sta-\r\nkeholder") == "stakeholder"

    # Spaces around newline
    assert clean_text("sta-   \n   keholder") == "stakeholder"
    print("  -> Passed broken words test.")


def test_preserve_real_hyphenated_terms():
    print("[2/6] Testing preservation of real hyphenated terms...")
    assert clean_text("state-of-the-art") == "state-of-the-art"
    assert clean_text("cost-effective") == "cost-effective"
    assert clean_text("fact-aware") == "fact-aware"
    assert clean_text("end-to-end") == "end-to-end"
    assert clean_text("twenty-one") == "twenty-one"
    print("  -> Passed real hyphenated terms preservation test.")


def test_mixed_sentence_with_both_broken_and_real_hyphenated():
    print("[3/6] Testing mixed text with broken words and real hyphenated terms...")
    raw = "The sta-\nkeholder reviewed the state-of-the-art pro-\ncess using a cost-effective method."
    cleaned = clean_text(raw)
    assert "stakeholder" in cleaned
    assert "process" in cleaned
    assert "state-of-the-art" in cleaned
    assert "cost-effective" in cleaned
    assert "sta-keholder" not in cleaned
    assert "pro-cess" not in cleaned
    print("  -> Passed mixed text test.")


def test_academic_cleaner_unwrap_pdf_lines():
    print("[4/6] Testing unwrap_pdf_lines in academic_cleaner...")
    assert "stakeholder" in unwrap_pdf_lines("sta-\nkeholder")
    assert "process" in unwrap_pdf_lines("pro-\ncess")
    assert "state-of-the-art" in unwrap_pdf_lines("state-of-the-art")
    assert "sta-keholder" not in unwrap_pdf_lines("sta-\nkeholder")
    assert "pro-cess" not in unwrap_pdf_lines("pro-\ncess")
    print("  -> Passed unwrap_pdf_lines test.")


def test_abstractive_path_fails_if_silently_extractive():
    print("[5/6] Testing abstractive path (verifying it does not silently return extractive output)...")
    summarizer = TransformerSummarizer(
        model_name="facebook/bart-large-cnn",
        mock_mode=False
    )

    # Must have loaded the neural weights
    assert summarizer.is_neural is True, f"Neural model failed to load: {summarizer.load_error}"
    assert summarizer.pipeline is not None, "Pipeline wrapper is None"
    assert summarizer.mock_mode is False, "TransformerSummarizer unexpectedly switched to mock_mode"

    sample_article = (
        "The Federal Reserve decided to hold interest rates steady on Wednesday following a two-day "
        "policy meeting in Washington. Officials cited slowing consumer spending and moderating inflation "
        "as key reasons for maintaining the benchmark rate in its current range of 5.25 to 5.5 percent. "
        "Federal Reserve Chairman Jerome Powell noted during a news conference that while inflation has declined "
        "from its peak forty-year highs, the central bank remains vigilant against future economic shocks."
    )

    result = summarizer.summarize(sample_article, max_length=60, min_length=20)

    # 1. Output mode must be neural_bart, not extractive_fallback
    assert result["mode"] == "neural_bart", (
        f"Abstractive path silently returned fallback mode: {result.get('mode')} "
        f"(Reason: {result.get('failure_reason')})"
    )
    assert result["is_abstractive"] is True, "is_abstractive flag is False"

    # 2. Output must NOT be an extractive verbatim copy of any source sentence
    source_sentences = [
        "The Federal Reserve decided to hold interest rates steady on Wednesday following a two-day policy meeting in Washington.",
        "Officials cited slowing consumer spending and moderating inflation as key reasons for maintaining the benchmark rate in its current range of 5.25 to 5.5 percent.",
        "Federal Reserve Chairman Jerome Powell noted during a news conference that while inflation has declined from its peak forty-year highs, the central bank remains vigilant against future economic shocks."
    ]

    generated_summary = result["summary"].strip()
    assert len(generated_summary) > 0, "Generated summary is empty"
    for src_sent in source_sentences:
        assert generated_summary != src_sent, (
            "Abstractive path silently returned an extractive sentence verbatim!"
        )

    print(f"  -> Generated Summary: {generated_summary}")
    print(f"  -> Model: {result['model_name']} | Device: {result['device']} | Time: {result['generation_time_sec']}s")
    print("  -> Passed abstractive non-extractive verification test.")


def test_mock_mode_honestly_labels_fallback():
    print("[6/6] Testing honest labeling of fallback when mock_mode=True...")
    summarizer = TransformerSummarizer(
        model_name="facebook/bart-large-cnn",
        mock_mode=True
    )
    result = summarizer.summarize("This is a simple test sentence for dry-run verification.")
    assert result["mode"] == "extractive_fallback"
    assert result["is_abstractive"] is False
    assert "Extractive fallback (not abstractive)" in result["label"]
    print("  -> Passed fallback honest labeling test.")


if __name__ == "__main__":
    print("=== Running Hyphenation & Abstractive Verification Tests ===\n")
    test_rejoin_broken_words_at_line_breaks()
    test_preserve_real_hyphenated_terms()
    test_mixed_sentence_with_both_broken_and_real_hyphenated()
    test_academic_cleaner_unwrap_pdf_lines()
    test_mock_mode_honestly_labels_fallback()
    test_abstractive_path_fails_if_silently_extractive()
    print("\n=== ALL TESTS PASSED SUCCESSFULLY! ===")
